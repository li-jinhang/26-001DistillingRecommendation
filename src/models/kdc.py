# coding: utf-8
#基本Hinge Encrypt蒸馏
#增加了动态置信度
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender
from utils_package.utils import get_model
import numpy as np
import scipy.sparse as sp
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, degree
from common.loss import BPRLoss, EmbLoss


class KDC(GeneralRecommender):
    def __init__(self, config, dataset):
        super(KDC, self).__init__(config, dataset)

        # ==========================================
        # 1. 蒸馏相关的超参数 (新增动态权重)
        # ==========================================
        self.temp = config.get('temp', 0.2)
        self.reg_weight = config.get('reg_weight', 1e-4)
        self.alpha_hinge = config.get('alpha_hinge', 1.0)
        self.beta_ce = config.get('beta_ce', 1.0)

        # ==========================================
        # 2. 学生模型参数定义
        # ==========================================
        self.num_user = self.n_users
        self.num_item = self.n_items
        self.feat_embed_dim = config.get('feat_embed_dim', 64)
        self.n_layers = config.get('n_mm_layers', 1)
        self.knn_k = config.get('knn_k', 10)
        self.mm_image_weight = config.get('mm_image_weight', 0.5)
        self.aggr_mode = 'add'
        self.dim_latent = config['embedding_size']

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        mm_adj_file = os.path.join(dataset_path, 'mm_adj_{}.pt'.format(self.knn_k))

        # 加载模态特征 (移除了冗余且未使用的 image_trs 和 text_trs)
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)

        # 构建 KNN 模态图
        if os.path.exists(mm_adj_file):
            self.mm_adj = torch.load(mm_adj_file, map_location=self.device)
        else:
            if self.v_feat is not None:
                indices, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
                self.mm_adj = image_adj
            if self.t_feat is not None:
                indices, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
                self.mm_adj = text_adj
            if self.v_feat is not None and self.t_feat is not None:
                self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
                del text_adj
                del image_adj
            torch.save(self.mm_adj, mm_adj_file)

        # 构建图的边索引
        train_interactions = dataset.inter_matrix(form='coo').astype(np.float32)
        rows = train_interactions.row
        cols = train_interactions.col + self.num_user
        edge_index = np.column_stack((rows, cols))
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)

        # 仿照 MENTOR 风格初始化用户模态融合权重，直接存放于对应设备
        self.weight_u = nn.Parameter(nn.init.xavier_normal_(
            torch.tensor(np.random.randn(self.num_user, 2, 1), dtype=torch.float32, requires_grad=True).to(
                self.device)))

        # 初始化各模态学生 GCN
        if self.v_feat is not None:
            self.v_gcn = GCN(self.num_user, self.num_item, self.aggr_mode, dim_latent=self.dim_latent,
                             device=self.device, features=self.v_feat)
        if self.t_feat is not None:
            self.t_gcn = GCN(self.num_user, self.num_item, self.aggr_mode, dim_latent=self.dim_latent,
                             device=self.device, features=self.t_feat)

        self.result_embed = None

        # ==========================================
        # 3. 教师模型的加载与冻结
        # ==========================================
        # --- 教师 1 ---
        teacher1_name = config['teacher1_model']
        teacher1_weight = config['teacher1_weight']
        self.teacher1 = get_model(teacher1_name)(config, dataset)
        teacher1_path = os.path.abspath(os.path.join(os.getcwd(), 'saved', teacher1_weight))
        self.teacher1.load_state_dict(torch.load(teacher1_path, map_location=self.device), strict=False)
        self.teacher1.eval()
        for param in self.teacher1.parameters():
            param.requires_grad = False

        # --- 教师 2 ---
        teacher2_name = config['teacher2_model']
        teacher2_weight = config['teacher2_weight']
        self.teacher2 = get_model(teacher2_name)(config, dataset)
        teacher2_path = os.path.abspath(os.path.join(os.getcwd(), 'saved', teacher2_weight))
        self.teacher2.load_state_dict(torch.load(teacher2_path, map_location=self.device), strict=False)
        self.teacher2.eval()
        for param in self.teacher2.parameters():
            param.requires_grad = False

    def get_knn_adj_mat(self, mm_embeddings):
        # 仿照 MENTOR 释放相似度矩阵，缓解大规模物品的 OOM 问题
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim  # 及时清理内存

        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def forward(self, interaction):
        user_nodes = interaction[0].clone()
        pos_item_nodes = interaction[1] + self.num_user  # 这里使用了 '+'，会自动创建新张量，是安全的
        neg_item_nodes = interaction[2] + self.num_user

        # 1. 基础多模态表示学习
        self.v_rep, self.v_preference = self.v_gcn(self.edge_index, self.v_feat)
        self.t_rep, self.t_preference = self.t_gcn(self.edge_index, self.t_feat)

        # 2. 拼接多模态表示
        representation = torch.cat((self.v_rep, self.t_rep), dim=1)

        # 3. 仿照 MENTOR 的干净拼接方式聚合用户表示
        w = F.softmax(self.weight_u, dim=1).transpose(1, 2)  # [num_user, 1, 2]
        u_v_weighted = w[:, :, 0] * self.v_rep[:self.num_user]
        u_t_weighted = w[:, :, 1] * self.t_rep[:self.num_user]
        user_rep = torch.cat((u_v_weighted, u_t_weighted), dim=1)

        # 4. 物品图传播
        item_rep = representation[self.num_user:]
        h = self.buildItemGraph(item_rep)
        item_rep = item_rep + h

        # 5. 保存并构建全局表征
        self.user_rep, self.item_rep = user_rep, item_rep
        self.result_embed = torch.cat((user_rep, item_rep), dim=0)

        # 6. 计算内积得分
        user_tensor = self.result_embed[user_nodes]
        pos_item_tensor = self.result_embed[pos_item_nodes]
        neg_item_tensor = self.result_embed[neg_item_nodes]

        pos_scores = torch.sum(user_tensor * pos_item_tensor, dim=1)
        neg_scores = torch.sum(user_tensor * neg_item_tensor, dim=1)

        return pos_scores, neg_scores

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        values = r_inv_sqrt[indices[0]] * r_inv_sqrt[indices[1]]
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def buildItemGraph(self, h):
        for i in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, h)
        return h

    def full_sort_predict(self, interaction):
        # 移除了重复的变量分配，使用与前向传播一致的逻辑确保评价时表示的更新
        self.v_rep, _ = self.v_gcn(self.edge_index, self.v_feat)
        self.t_rep, _ = self.t_gcn(self.edge_index, self.t_feat)

        representation = torch.cat((self.v_rep, self.t_rep), dim=1)

        w = F.softmax(self.weight_u, dim=1).transpose(1, 2)
        u_v_weighted = w[:, :, 0] * self.v_rep[:self.num_user]
        u_t_weighted = w[:, :, 1] * self.t_rep[:self.num_user]
        user_rep = torch.cat((u_v_weighted, u_t_weighted), dim=1)

        item_rep = representation[self.num_user:]
        h = self.buildItemGraph(item_rep)
        item_rep = item_rep + h

        self.result_embed = torch.cat((user_rep, item_rep), dim=0)

        temp_user_tensor = user_rep[interaction[0], :]
        return torch.matmul(temp_user_tensor, item_rep.t())

    def calculate_loss(self, interaction):
        user = interaction[0]
        pos_scores, neg_scores = self.forward(interaction)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))

        # 1. 教师模型前向传播 (使用克隆对象)
        with torch.no_grad():
            # 使用 interaction.clone() 确保教师的操作不影响原始数据
            t1_pos, t1_neg = self.teacher1.forward(interaction.clone())
            t2_pos, t2_neg = self.teacher2.forward(interaction.clone())

        s_diff = pos_scores - neg_scores
        t1_diff = t1_pos - t1_neg
        t2_diff = t2_pos - t2_neg

        # ==========================================
        # 2. 动态权重计算：基于教师的样本级 BPR Loss
        # ==========================================
        # 计算教师模型在每个样本上的 BPR 损失（不取平均）
        t1_loss_sample = -F.logsigmoid(t1_diff)
        t2_loss_sample = -F.logsigmoid(t2_diff)

        # 在 batch 维度拼接负损失（负号使得损失越小的模型，在后续 Softmax 后的值越大）
        # 形状: [batch_size, 2]
        stacked_teacher_losses = torch.stack([-t1_loss_sample, -t2_loss_sample], dim=1)

        # 计算样本级别的动态教师权重
        teacher_weights = F.softmax(stacked_teacher_losses, dim=1)
        w1 = teacher_weights[:, 0]  # 教师1的样本级权重
        w2 = teacher_weights[:, 1]  # 教师2的样本级权重

        # ==========================================
        # 3. 动态加权的 Hinge 蒸馏损失
        # ==========================================
        # 计算样本级 Hinge loss
        hinge_loss_1_sample = torch.clamp(t1_diff - s_diff, min=0.0)
        hinge_loss_2_sample = torch.clamp(t2_diff - s_diff, min=0.0)

        # 结合权重求均值
        hinge_loss = self.alpha_hinge * torch.mean(w1 * hinge_loss_1_sample + w2 * hinge_loss_2_sample)

        # ==========================================
        # 4. 动态加权的 Cross Entropy 蒸馏损失
        # ==========================================
        t1_bar = torch.sigmoid(t1_diff / self.temp)
        t2_bar = torch.sigmoid(t2_diff / self.temp)
        s_bar = torch.sigmoid(s_diff / self.temp)
        eps = 1e-8

        # 计算样本级 CE loss
        ce_loss_1_sample = -(t1_bar * torch.log(s_bar + eps) + (1 - t1_bar) * torch.log(1 - s_bar + eps))
        ce_loss_2_sample = -(t2_bar * torch.log(s_bar + eps) + (1 - t2_bar) * torch.log(1 - s_bar + eps))

        # 结合权重求均值
        ce_loss = self.beta_ce * torch.mean(w1 * ce_loss_1_sample + w2 * ce_loss_2_sample)

        # 5. 正则化损失
        reg_loss = self.reg_weight * (
                (self.v_preference[user] ** 2).mean() +
                (self.t_preference[user] ** 2).mean() +
                (self.weight_u ** 2).mean()
        )

        return bpr_loss + hinge_loss + ce_loss + reg_loss


class GCN(torch.nn.Module):
    def __init__(self, num_user, num_item, aggr_mode, dim_latent=64, device=None, features=None):
        super(GCN, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.device = device
        self.dim_feat = features.size(1)
        self.dim_latent = dim_latent

        # 仿照 MENTOR 风格：将 Parameter 初始化时直接赋予 device，避免 forward 时造成冗余显存分配
        self.preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
            np.random.randn(num_user, self.dim_latent), dtype=torch.float32, requires_grad=True),
            gain=1).to(self.device))

        # 特征投影层
        self.MLP = nn.Linear(self.dim_feat, 4 * self.dim_latent)
        self.MLP_1 = nn.Linear(4 * self.dim_latent, self.dim_latent)

        # 基础 GCN 层
        self.conv_embed_1 = Base_gcn(self.dim_latent, self.dim_latent, aggr=aggr_mode)

    def forward(self, edge_index, features):
        # 投影模态特征
        features = F.normalize(features, p=2, dim=1)
        temp_features = self.MLP_1(F.leaky_relu(self.MLP(features)))

        # 移除了原版 x.to(self.device) 的冗余操作
        x = torch.cat((self.preference, temp_features), dim=0)
        x = F.normalize(x)

        # 多层演化
        h1 = self.conv_embed_1(x, edge_index)
        h2 = self.conv_embed_1(h1, edge_index)

        x_hat = x + h1 + h2
        return x_hat, self.preference


class Base_gcn(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr='add', **kwargs):
        super(Base_gcn, self).__init__(aggr=aggr, **kwargs)
        self.aggr = aggr

    def forward(self, x, edge_index):
        edge_index, _ = remove_self_loops(edge_index)
        return self.propagate(edge_index, x=x, size=(x.size(0), x.size(0)))

    def message(self, x_j, edge_index, size):
        if self.aggr == 'add':
            row, col = edge_index
            deg = degree(row, size[0], dtype=x_j.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            return norm.view(-1, 1) * x_j
        return x_j

    def update(self, aggr_out):
        return aggr_out
