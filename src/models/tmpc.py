# coding: utf-8
#互学习
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
import torch_geometric

# 假设这些是你所在框架的通用组件，保持原样导入
from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss
from common.init import xavier_uniform_initialization


# =====================================================================
# 1. 基础图卷积组件 (供 TMPA 和 TMPB 共享使用)
# =====================================================================
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


class GCN(torch.nn.Module):
    def __init__(self, num_user, num_item, aggr_mode, dim_latent=64, device=None, features=None):
        super(GCN, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.device = device
        self.dim_feat = features.size(1)
        self.dim_latent = dim_latent

        # 用户偏好向量 (ID-based)
        self.preference = nn.Parameter(nn.init.xavier_normal_(
            torch.tensor(np.random.randn(num_user, self.dim_latent), dtype=torch.float32, requires_grad=True)))

        # 特征投影层
        self.MLP = nn.Linear(self.dim_feat, 4 * self.dim_latent)
        self.MLP_1 = nn.Linear(4 * self.dim_latent, self.dim_latent)

        # 基础 GCN 层
        self.conv_embed_1 = Base_gcn(self.dim_latent, self.dim_latent, aggr=aggr_mode)

    def forward(self, edge_index, features):
        # 投影模态特征
        temp_features = self.MLP_1(F.leaky_relu(self.MLP(features)))
        x = torch.cat((self.preference, temp_features), dim=0).to(self.device)
        x = F.normalize(x).to(self.device)

        # 多层演化
        h1 = self.conv_embed_1(x, edge_index)
        h2 = self.conv_embed_1(h1, edge_index)

        x_hat = x + h1 + h2
        return x_hat, self.preference


# =====================================================================
# 2. 模型A：TMPA (基于宏观高斯分布对齐)
# =====================================================================
class TMPA(GeneralRecommender):
    def __init__(self, config, dataset):
        super(TMPA, self).__init__(config, dataset)

        num_user = self.n_users
        num_item = self.n_items
        dim_x = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_mm_layers']
        self.knn_k = config['knn_k']
        self.mm_image_weight = config['mm_image_weight']

        self.num_user = num_user
        self.num_item = num_item
        self.k = 40
        self.aggr_mode = 'add'
        self.dataset = dataset
        self.dropout = config['dropout']
        self.reg_weight = config['reg_weight']
        self.align_weight = config['align_weight']
        self.temp = config['temp']

        self.v_rep = None
        self.t_rep = None
        self.id_rep = None
        self.dim_latent = 64
        self.mm_adj = None

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        self.user_graph_dict = np.load(os.path.join(dataset_path, config['user_graph_dict_file']),
                                       allow_pickle=True).item()

        mm_adj_file = os.path.join(dataset_path, 'mm_adj_{}.pt'.format(self.knn_k))

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)

        if os.path.exists(mm_adj_file):
            self.mm_adj = torch.load(mm_adj_file)
        else:
            if self.v_feat is not None:
                indices, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
                self.mm_adj = image_adj
            if self.t_feat is not None:
                indices, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
                self.mm_adj = text_adj
            if self.v_feat is not None and self.t_feat is not None:
                self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
            torch.save(self.mm_adj, mm_adj_file)

        # 训练交互矩阵转为 edge_index
        train_interactions = dataset.inter_matrix(form='coo').astype(np.float32)
        edge_index = self.pack_edge_index(train_interactions)
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)

        # 模态权重参数
        self.weight_u = nn.Parameter(nn.init.xavier_normal_(
            torch.tensor(np.random.randn(self.num_user, 2, 1), dtype=torch.float32, requires_grad=True)))

        # 初始化各模态 GCN
        if self.v_feat is not None:
            self.v_gcn = GCN(self.num_user, self.num_item, self.aggr_mode, dim_latent=64,
                             device=self.device, features=self.v_feat)
        if self.t_feat is not None:
            self.t_gcn = GCN(self.num_user, self.num_item, self.aggr_mode, dim_latent=64,
                             device=self.device, features=self.t_feat)

        self.id_feat = nn.Parameter(
            nn.init.xavier_normal_(torch.tensor(np.random.randn(self.n_items, self.dim_latent), dtype=torch.float32,
                                                requires_grad=True), gain=1).to(self.device))
        self.id_gcn = GCN(self.num_user, self.num_item, self.aggr_mode,
                          dim_latent=64, device=self.device, features=self.id_feat)

        # 结果 Embedding 初始化
        self.result_embed = nn.Parameter(
            nn.init.xavier_normal_(torch.tensor(np.random.randn(num_user + num_item, dim_x)))).to(self.device)

    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device).unsqueeze(1).expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        values = r_inv_sqrt[indices[0]] * r_inv_sqrt[indices[1]]
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        return np.column_stack((rows, cols))

    def forward(self, interaction):
        # 【修正点】: 不直接对 interaction[1] 和 interaction[2] 进行 += 操作，避免修改原始数据
        user_nodes = interaction[0]
        pos_item_nodes = interaction[1] + self.n_users
        neg_item_nodes = interaction[2] + self.n_users

        # 基础模态表示学习
        self.v_rep, self.v_preference = self.v_gcn(self.edge_index, self.v_feat)
        self.t_rep, self.t_preference = self.t_gcn(self.edge_index, self.t_feat)
        self.id_rep, _ = self.id_gcn(self.edge_index, self.id_feat)

        # 拼接表示
        representation = torch.cat((self.v_rep, self.t_rep), dim=1)
        guide_representation = torch.cat((self.id_rep, self.id_rep), dim=1)
        v_representation = torch.cat((self.v_rep, self.v_rep), dim=1)
        t_representation = torch.cat((self.t_rep, self.t_rep), dim=1)

        # 用户聚合逻辑
        u_v, u_t = self.v_rep[:self.num_user], self.t_rep[:self.num_user]
        w = F.softmax(self.weight_u, dim=1).transpose(1, 2)
        user_rep = torch.cat((w[:, :, 0] * u_v, w[:, :, 1] * u_t), dim=1)

        guide_user_rep = torch.cat((self.id_rep[:self.num_user], self.id_rep[:self.num_user]), dim=1)
        v_user_rep = torch.cat((u_v, u_v), dim=1)
        t_user_rep = torch.cat((u_t, u_t), dim=1)

        # 物品图传播
        item_rep = representation[self.num_user:] + self.buildItemGraph(representation[self.num_user:])
        guide_item_rep = guide_representation[self.num_user:] + self.buildItemGraph(
            guide_representation[self.num_user:])
        v_item_rep = v_representation[self.num_user:] + self.buildItemGraph(v_representation[self.num_user:])
        t_item_rep = t_representation[self.num_user:] + self.buildItemGraph(t_representation[self.num_user:])

        # 最终 Embedding 拼接
        self.user_rep, self.item_rep = user_rep, item_rep
        self.result_embed = torch.cat((user_rep, item_rep), dim=0)

        self.result_embed_guide = torch.cat((guide_user_rep, guide_item_rep), dim=0)
        self.result_embed_v = torch.cat((v_user_rep, v_item_rep), dim=0)
        self.result_embed_t = torch.cat((t_user_rep, t_item_rep), dim=0)

        # 计算得分
        user_tensor = self.result_embed[user_nodes]
        pos_item_tensor = self.result_embed[pos_item_nodes]
        neg_item_tensor = self.result_embed[neg_item_nodes]
        pos_scores = torch.sum(user_tensor * pos_item_tensor, dim=1)
        neg_scores = torch.sum(user_tensor * neg_item_tensor, dim=1)
        return pos_scores, neg_scores

    def buildItemGraph(self, h):
        for i in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, h)
        return h

    def fit_Gaussian_dis(self):
        return (torch.var(self.result_embed), torch.mean(self.result_embed),
                torch.var(self.result_embed_guide), torch.mean(self.result_embed_guide),
                torch.var(self.result_embed_v), torch.mean(self.result_embed_v),
                torch.var(self.result_embed_t), torch.mean(self.result_embed_t))

    def calculate_loss(self, interaction):
        user = interaction[0]
        pos_scores, neg_scores = self.forward(interaction)

        loss_value = -torch.mean(torch.log2(torch.sigmoid(pos_scores - neg_scores) + 1e-8))

        reg_loss = self.reg_weight * ((self.v_preference[user] ** 2).mean() +
                                      (self.t_preference[user] ** 2).mean() +
                                      (self.weight_u ** 2).mean())

        r_var, r_mean, g_var, g_mean, v_var, v_mean, t_var, t_mean = self.fit_Gaussian_dis()
        align_loss = ((torch.abs(g_var - r_var) + torch.abs(g_mean - r_mean)) +
                      (torch.abs(g_var - v_var) + torch.abs(g_mean - v_mean)) +
                      (torch.abs(g_var - t_var) + torch.abs(g_mean - t_mean)) +
                      (torch.abs(r_var - v_var) + torch.abs(r_mean - v_mean)) +
                      (torch.abs(r_var - t_var) + torch.abs(r_mean - t_mean)) +
                      (torch.abs(v_var - t_var) + torch.abs(v_mean - t_mean))).mean()

        return loss_value + reg_loss + align_loss * self.align_weight

    def full_sort_predict(self, interaction):
        user_tensor = self.result_embed[:self.n_users]
        item_tensor = self.result_embed[self.n_users:]
        temp_user_tensor = user_tensor[interaction[0], :]
        return torch.matmul(temp_user_tensor, item_tensor.t())


# =====================================================================
# 3. 模型B：TMPB (基于微观 InfoNCE 细粒度对齐)
# =====================================================================
class TMPB(GeneralRecommender):
    def __init__(self, config, dataset):
        super(TMPB, self).__init__(config, dataset)

        num_user = self.n_users
        num_item = self.n_items
        dim_x = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_mm_layers']
        self.knn_k = config['knn_k']
        self.mm_image_weight = config['mm_image_weight']

        self.num_user = num_user
        self.num_item = num_item
        self.k = 40
        self.aggr_mode = 'add'
        self.dataset = dataset
        self.dropout = config['dropout']
        self.reg_weight = config['reg_weight']
        self.align_weight = config['align_weight']
        self.temp = config['temp']

        self.v_rep = None
        self.t_rep = None
        self.id_rep = None
        self.dim_latent = 64
        self.mm_adj = None

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        self.user_graph_dict = np.load(os.path.join(dataset_path, config['user_graph_dict_file']),
                                       allow_pickle=True).item()

        mm_adj_file = os.path.join(dataset_path, 'mm_adj_{}.pt'.format(self.knn_k))

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)

        if os.path.exists(mm_adj_file):
            self.mm_adj = torch.load(mm_adj_file)
        else:
            if self.v_feat is not None:
                indices, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
                self.mm_adj = image_adj
            if self.t_feat is not None:
                indices, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
                self.mm_adj = text_adj
            if self.v_feat is not None and self.t_feat is not None:
                self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
            torch.save(self.mm_adj, mm_adj_file)

        train_interactions = dataset.inter_matrix(form='coo').astype(np.float32)
        edge_index = self.pack_edge_index(train_interactions)
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)

        self.weight_u = nn.Parameter(nn.init.xavier_normal_(
            torch.tensor(np.random.randn(self.num_user, 2, 1), dtype=torch.float32, requires_grad=True)))

        if self.v_feat is not None:
            self.v_gcn = GCN(self.num_user, self.num_item, self.aggr_mode, dim_latent=64,
                             device=self.device, features=self.v_feat)
        if self.t_feat is not None:
            self.t_gcn = GCN(self.num_user, self.num_item, self.aggr_mode, dim_latent=64,
                             device=self.device, features=self.t_feat)

        self.id_feat = nn.Parameter(
            nn.init.xavier_normal_(torch.tensor(np.random.randn(self.n_items, self.dim_latent), dtype=torch.float32,
                                                requires_grad=True), gain=1).to(self.device))
        self.id_gcn = GCN(self.num_user, self.num_item, self.aggr_mode,
                          dim_latent=64, device=self.device, features=self.id_feat)

        self.result_embed = nn.Parameter(
            nn.init.xavier_normal_(torch.tensor(np.random.randn(num_user + num_item, dim_x)))).to(self.device)

    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device).unsqueeze(1).expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        values = r_inv_sqrt[indices[0]] * r_inv_sqrt[indices[1]]
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        return np.column_stack((rows, cols))

    def forward(self, interaction):
        # 【修正点】: 不直接对 interaction[1] 和 interaction[2] 进行 += 操作
        user_nodes = interaction[0]
        pos_item_nodes = interaction[1] + self.n_users
        neg_item_nodes = interaction[2] + self.n_users

        # 基础模态表示学习
        self.v_rep, self.v_preference = self.v_gcn(self.edge_index, self.v_feat)
        self.t_rep, self.t_preference = self.t_gcn(self.edge_index, self.t_feat)
        self.id_rep, _ = self.id_gcn(self.edge_index, self.id_feat)

        # 拼接表示
        representation = torch.cat((self.v_rep, self.t_rep), dim=1)
        guide_representation = torch.cat((self.id_rep, self.id_rep), dim=1)
        v_representation = torch.cat((self.v_rep, self.v_rep), dim=1)
        t_representation = torch.cat((self.t_rep, self.t_rep), dim=1)

        # 用户聚合逻辑
        u_v, u_t = self.v_rep[:self.num_user], self.t_rep[:self.num_user]
        w = F.softmax(self.weight_u, dim=1).transpose(1, 2)
        user_rep = torch.cat((w[:, :, 0] * u_v, w[:, :, 1] * u_t), dim=1)

        guide_user_rep = torch.cat((self.id_rep[:self.num_user], self.id_rep[:self.num_user]), dim=1)
        v_user_rep = torch.cat((u_v, u_v), dim=1)
        t_user_rep = torch.cat((u_t, u_t), dim=1)

        # 物品图传播
        item_rep = representation[self.num_user:] + self.buildItemGraph(representation[self.num_user:])
        guide_item_rep = guide_representation[self.num_user:] + self.buildItemGraph(
            guide_representation[self.num_user:])
        v_item_rep = v_representation[self.num_user:] + self.buildItemGraph(v_representation[self.num_user:])
        t_item_rep = t_representation[self.num_user:] + self.buildItemGraph(t_representation[self.num_user:])

        # 最终 Embedding 拼接
        self.user_rep, self.item_rep = user_rep, item_rep
        self.result_embed = torch.cat((user_rep, item_rep), dim=0)

        self.result_embed_guide = torch.cat((guide_user_rep, guide_item_rep), dim=0)
        self.result_embed_v = torch.cat((v_user_rep, v_item_rep), dim=0)
        self.result_embed_t = torch.cat((t_user_rep, t_item_rep), dim=0)

        # 计算得分
        user_tensor = self.result_embed[user_nodes]
        pos_item_tensor = self.result_embed[pos_item_nodes]
        neg_item_tensor = self.result_embed[neg_item_nodes]
        pos_scores = torch.sum(user_tensor * pos_item_tensor, dim=1)
        neg_scores = torch.sum(user_tensor * neg_item_tensor, dim=1)
        return pos_scores, neg_scores

    def buildItemGraph(self, h):
        for i in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, h)
        return h

    def calc_infonce_loss(self, embed1, embed2, nodes):
        z1 = F.normalize(embed1[nodes], p=2, dim=1)
        z2 = F.normalize(embed2[nodes], p=2, dim=1)
        pos_sim = torch.sum(z1 * z2, dim=-1) / self.temp
        sim_matrix = torch.matmul(z1, z2.T) / self.temp
        loss = -torch.log(torch.exp(pos_sim) / torch.sum(torch.exp(sim_matrix), dim=1)).mean()
        return loss

    def calculate_loss(self, interaction):
        user = interaction[0]
        # 【修正点】: 此处同样使用非原地操作
        pos_item = interaction[1] + self.n_users

        pos_scores, neg_scores = self.forward(interaction)

        loss_value = -torch.mean(torch.log2(torch.sigmoid(pos_scores - neg_scores) + 1e-8))

        reg_loss = self.reg_weight * ((self.v_preference[user] ** 2).mean() +
                                      (self.t_preference[user] ** 2).mean() +
                                      (self.weight_u ** 2).mean())

        batch_nodes = torch.unique(torch.cat([user, pos_item]))

        align_loss_v_id = self.calc_infonce_loss(self.result_embed_v, self.result_embed_guide, batch_nodes)
        align_loss_t_id = self.calc_infonce_loss(self.result_embed_t, self.result_embed_guide, batch_nodes)
        align_loss_v_t = self.calc_infonce_loss(self.result_embed_v, self.result_embed_t, batch_nodes)

        align_loss = (align_loss_v_id + align_loss_t_id + align_loss_v_t) / 3.0

        return loss_value + reg_loss + align_loss * self.align_weight

    def full_sort_predict(self, interaction):
        user_tensor = self.result_embed[:self.n_users]
        item_tensor = self.result_embed[self.n_users:]
        temp_user_tensor = user_tensor[interaction[0], :]
        return torch.matmul(temp_user_tensor, item_tensor.t())


# =====================================================================
# 4. 集成互学习模型：双向蒸馏对齐
# =====================================================================
class TMPC(GeneralRecommender):
    def __init__(self, config, dataset):
        super(TMPC, self).__init__(config, dataset)

        # 1. 实例化双分支模型
        self.model_a = TMPA(config, dataset)
        self.model_b = TMPB(config, dataset)

        # 2. 互学习超参数
        self.kd_weight = config['kd_weight']
        self.kd_temp = config['kd_temp']

    def forward(self, interaction):
        pos_scores_a, neg_scores_a = self.model_a(interaction)
        pos_scores_b, neg_scores_b = self.model_b(interaction)
        return pos_scores_a, neg_scores_a, pos_scores_b, neg_scores_b

    def calculate_loss(self, interaction):
        user = interaction[0]
        pos_item = interaction[1] + self.n_users

        pos_a, neg_a, pos_b, neg_b = self.forward(interaction)

        # Model A Loss
        bpr_loss_a = -torch.mean(torch.log2(torch.sigmoid(pos_a - neg_a) + 1e-8))
        reg_loss_a = self.model_a.reg_weight * ((self.model_a.v_preference[user] ** 2).mean() +
                                                (self.model_a.t_preference[user] ** 2).mean() +
                                                (self.model_a.weight_u ** 2).mean())

        r_var, r_mean, g_var, g_mean, v_var, v_mean, t_var, t_mean = self.model_a.fit_Gaussian_dis()
        align_loss_a = ((torch.abs(g_var - r_var) + torch.abs(g_mean - r_mean)) +
                        (torch.abs(g_var - v_var) + torch.abs(g_mean - v_mean)) +
                        (torch.abs(g_var - t_var) + torch.abs(g_mean - t_mean)) +
                        (torch.abs(r_var - v_var) + torch.abs(r_mean - v_mean)) +
                        (torch.abs(r_var - t_var) + torch.abs(r_mean - t_mean)) +
                        (torch.abs(v_var - t_var) + torch.abs(v_mean - t_mean))).mean()

        loss_a = bpr_loss_a + reg_loss_a + align_loss_a * self.model_a.align_weight

        # Model B Loss
        bpr_loss_b = -torch.mean(torch.log2(torch.sigmoid(pos_b - neg_b) + 1e-8))
        reg_loss_b = self.model_b.reg_weight * ((self.model_b.v_preference[user] ** 2).mean() +
                                                (self.model_b.t_preference[user] ** 2).mean() +
                                                (self.model_b.weight_u ** 2).mean())

        batch_nodes = torch.unique(torch.cat([user, pos_item]))
        align_loss_v_id = self.model_b.calc_infonce_loss(self.model_b.result_embed_v, self.model_b.result_embed_guide,
                                                         batch_nodes)
        align_loss_t_id = self.model_b.calc_infonce_loss(self.model_b.result_embed_t, self.model_b.result_embed_guide,
                                                         batch_nodes)
        align_loss_v_t = self.model_b.calc_infonce_loss(self.model_b.result_embed_v, self.model_b.result_embed_t,
                                                        batch_nodes)
        align_loss_b = (align_loss_v_id + align_loss_t_id + align_loss_v_t) / 3.0

        loss_b = bpr_loss_b + reg_loss_b + align_loss_b * self.model_b.align_weight

        # Mutual Learning (KD)
        logits_a = (pos_a - neg_a) / self.kd_temp
        logits_b = (pos_b - neg_b) / self.kd_temp

        p_a = torch.sigmoid(logits_a)
        p_b = torch.sigmoid(logits_b)

        dist_a = torch.stack([p_a, 1.0 - p_a], dim=1)
        dist_b = torch.stack([p_b, 1.0 - p_b], dim=1)

        log_dist_a = torch.log(dist_a + 1e-8)
        log_dist_b = torch.log(dist_b + 1e-8)

        kl_loss_a = F.kl_div(log_dist_a, dist_b.detach(), reduction='batchmean')
        kl_loss_b = F.kl_div(log_dist_b, dist_a.detach(), reduction='batchmean')

        mutual_loss = kl_loss_a + kl_loss_b

        total_loss = loss_a + loss_b + self.kd_weight * mutual_loss

        return total_loss

    def full_sort_predict(self, interaction):
        # 推断时集成两者的结果
        user_tensor_a = self.model_a.result_embed[:self.n_users]
        item_tensor_a = self.model_a.result_embed[self.n_users:]
        temp_user_a = user_tensor_a[interaction[0], :]
        scores_a = torch.matmul(temp_user_a, item_tensor_a.t())

        user_tensor_b = self.model_b.result_embed[:self.n_users]
        item_tensor_b = self.model_b.result_embed[self.n_users:]
        temp_user_b = user_tensor_b[interaction[0], :]
        scores_b = torch.matmul(temp_user_b, item_tensor_b.t())

        return (scores_a + scores_b) / 2.0