# coding: utf-8
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, degree

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss


class MADA(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MADA, self).__init__(config, dataset)

        # 参数初始化
        self.num_user = self.n_users
        self.num_item = self.n_items
        self.dim_latent = config.get('embedding_size', 64)
        self.feat_embed_dim = config.get('feat_embed_dim', 64)
        self.n_layers = config.get('n_mm_layers', 1)
        self.knn_k = config.get('knn_k', 10)
        self.mm_image_weight = config.get('mm_image_weight', 0.5)
        self.aggr_mode = 'add'
        self.dataset = dataset
        self.reg_weight = config.get('reg_weight', 1e-4)
        self.align_weight = config.get('align_weight', 1e-2)

        # 一、初始化与数据准备 (Initialization)

        # 1. 多模态特征嵌入：加载特征并投影到统一隐空间
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)

        # 2. 构建物品-物品相似度图 (kNN Graph)
        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        mm_adj_file = os.path.join(dataset_path, f'mm_adj_{self.knn_k}.pt')
        if os.path.exists(mm_adj_file):
            self.mm_adj = torch.load(mm_adj_file).to(self.device)
        else:
            if self.v_feat is not None:
                _, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
                self.mm_adj = image_adj
            if self.t_feat is not None:
                _, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
                self.mm_adj = text_adj
            if self.v_feat is not None and self.t_feat is not None:
                self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
            torch.save(self.mm_adj, mm_adj_file)

        # 3. 构建用户-物品交互图 edge_index
        train_interactions = dataset.inter_matrix(form='coo').astype(np.float32)
        edge_index = self.pack_edge_index(train_interactions)
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)  # 双向边

        # 建立用于 Sum Pooling 的稀疏矩阵 (User x Item)
        self.A_sum = self.build_sum_pooling_adj(train_interactions)

        # 结构/ID特征初始化
        self.id_feat = nn.Parameter(nn.init.xavier_normal_(
            torch.empty(self.n_items, self.dim_latent, dtype=torch.float32, requires_grad=True)))

        # 二、多模态表示学习 (Multimodal Representation Learning)
        # 独立的 GCN 演化路径
        if self.v_feat is not None:
            self.v_gcn = GCN(self.dataset, self.num_user, self.num_item, self.aggr_mode, dim_latent=self.dim_latent,
                             device=self.device, features=self.v_feat)
        if self.t_feat is not None:
            self.t_gcn = GCN(self.dataset, self.num_user, self.num_item, self.aggr_mode, dim_latent=self.dim_latent,
                             device=self.device, features=self.t_feat)
        self.id_gcn = GCN(self.dataset, self.num_user, self.num_item, self.aggr_mode, dim_latent=self.dim_latent,
                          device=self.device, features=self.id_feat)

        # 三、跨模态粗粒度对齐 (Coarse-grained Alignment) 相关 MLP
        # 将聚合后的特征投影到一个公共的统一隐向量空间
        self.align_mlp_id = nn.Sequential(nn.Linear(self.dim_latent, self.dim_latent), nn.LeakyReLU(),
                                          nn.Linear(self.dim_latent, self.dim_latent))
        self.align_mlp_v = nn.Sequential(nn.Linear(self.dim_latent, self.dim_latent), nn.LeakyReLU(),
                                         nn.Linear(self.dim_latent, self.dim_latent))
        self.align_mlp_t = nn.Sequential(nn.Linear(self.dim_latent, self.dim_latent), nn.LeakyReLU(),
                                         nn.Linear(self.dim_latent, self.dim_latent))

        # 四、多模态融合与传播 (Fusion & Propagation) 相关参数
        # 1. 用户侧：注意力权重融合
        self.weight_u = nn.Parameter(nn.init.xavier_normal_(
            torch.empty(self.num_user, 2, 1, dtype=torch.float32, requires_grad=True)))

        # 2. 物品侧：拼接后降维的 MLP
        self.item_fusion_mlp = nn.Linear(self.dim_latent * 3, self.dim_latent)

        # 最终存储 embedding 用于 inference
        self.result_embed = nn.Parameter(
            nn.init.xavier_normal_(torch.empty(self.num_user + self.num_item, self.dim_latent))).to(self.device)

    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        return np.column_stack((rows, cols))

    def build_sum_pooling_adj(self, inter_mat):
        """构建用于计算用户所有交互物品 Sum Pooling 的稀疏邻接矩阵"""
        rows = torch.tensor(inter_mat.row, dtype=torch.long)
        cols = torch.tensor(inter_mat.col, dtype=torch.long)
        vals = torch.ones_like(rows, dtype=torch.float32)
        A_sum = torch.sparse.FloatTensor(
            torch.stack([rows, cols]), vals, torch.Size([self.n_users, self.n_items])
        ).to(self.device)
        return A_sum

    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device).unsqueeze(1).expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)

        # Laplacian Normalization
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        values = r_inv_sqrt[indices[0]] * r_inv_sqrt[indices[1]]
        return indices, torch.sparse.FloatTensor(indices, values, adj_size)

    def forward(self, interaction):
        user_nodes, pos_item_nodes, neg_item_nodes = interaction[0], interaction[1], interaction[2]
        pos_item_nodes += self.n_users
        neg_item_nodes += self.n_users

        # ---------------------------------------------------------
        # 第二阶段：多模态表示学习 (GCN with Residuals)
        # ---------------------------------------------------------
        self.v_rep, self.v_pref = self.v_gcn(self.edge_index, self.v_feat)
        self.t_rep, self.t_pref = self.t_gcn(self.edge_index, self.t_feat)
        self.id_rep, self.id_pref = self.id_gcn(self.edge_index, self.id_feat)

        user_v, item_v = self.v_rep[:self.num_user], self.v_rep[self.num_user:]
        user_t, item_t = self.t_rep[:self.num_user], self.t_rep[self.num_user:]
        user_id, item_id = self.id_rep[:self.num_user], self.id_rep[self.num_user:]

        # ---------------------------------------------------------
        # 第三阶段：跨模态粗粒度对齐 (Coarse-grained Alignment)
        # ---------------------------------------------------------
        # 1. Sum Pooling (聚合用户交互过的所有物品)
        self.pool_v = torch.sparse.mm(self.A_sum, item_v)
        self.pool_t = torch.sparse.mm(self.A_sum, item_t)
        self.pool_id = torch.sparse.mm(self.A_sum, item_id)

        # 2. 多模态空间投影
        self.proj_v = self.align_mlp_v(self.pool_v)
        self.proj_t = self.align_mlp_t(self.pool_t)
        self.proj_id = self.align_mlp_id(self.pool_id)

        # ---------------------------------------------------------
        # 第四阶段：多模态融合与传播 (Fusion & Propagation)
        # ---------------------------------------------------------
        # 1. 用户侧：动态注意力权重融合
        weight_u = F.softmax(self.weight_u, dim=1)  # [num_user, 2, 1]
        user_vt_fused = weight_u[:, 0] * user_v + weight_u[:, 1] * user_t
        final_user_rep = user_id + user_vt_fused  # 加上结构 ID 的特征

        # 2. 物品侧：模态图传播 (Item-Item)
        # 拼接不同模态表示
        item_concat = torch.cat((item_id, item_v, item_t), dim=1)

        # 在 mm_adj 上进行图传播
        h_item = item_concat
        for _ in range(self.n_layers):
            h_item = torch.sparse.mm(self.mm_adj, h_item)

        # 投影回共享特征维度
        final_item_rep = self.item_fusion_mlp(h_item)

        # 保存全局节点表示供 Inference 排序使用
        self.result_embed = torch.cat((final_user_rep, final_item_rep), dim=0)

        # 计算批次内正负样本得分
        u_tensor = self.result_embed[user_nodes]
        pos_tensor = self.result_embed[pos_item_nodes]
        neg_tensor = self.result_embed[neg_item_nodes]

        pos_scores = torch.sum(u_tensor * pos_tensor, dim=1)
        neg_scores = torch.sum(u_tensor * neg_tensor, dim=1)

        return pos_scores, neg_scores

    def calculate_loss(self, interaction):
        user = interaction[0]
        pos_scores, neg_scores = self.forward(interaction)

        # 1. BPR Loss
        bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8))

        # 2. 粗粒度一致性对齐损失 (Alignment Loss) - 仿照 mentor 均值/方差方式
        # 计算经过 Sum Pooling 并投影到同一空间的表示分布差异
        id_mean, id_var = torch.mean(self.proj_id), torch.var(self.proj_id)
        v_mean, v_var = torch.mean(self.proj_v), torch.var(self.proj_v)
        t_mean, t_var = torch.mean(self.proj_t), torch.var(self.proj_t)

        align_loss = (
                             (torch.abs(id_var - v_var) + torch.abs(id_mean - v_mean)) +
                             (torch.abs(id_var - t_var) + torch.abs(id_mean - t_mean)) +
                             (torch.abs(v_var - t_var) + torch.abs(v_mean - t_mean))
                     ).mean() * self.align_weight

        # 3. 正则化损失 (Reg Loss)
        reg_loss = self.reg_weight * (
                (self.v_pref[user] ** 2).mean() +
                (self.t_pref[user] ** 2).mean() +
                (self.id_pref[user] ** 2).mean() +
                (self.weight_u ** 2).mean()
        )

        return bpr_loss + align_loss + reg_loss

    def full_sort_predict(self, interaction):
        # 第六阶段：预测与全排序
        user_tensor = self.result_embed[:self.n_users]
        item_tensor = self.result_embed[self.n_users:]

        temp_user_tensor = user_tensor[interaction[0], :]
        score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())
        return score_matrix


class GCN(torch.nn.Module):
    """
    底层特征多层卷积表示学习网络，自带特征投影(MLP)与残差连接
    """

    def __init__(self, datasets, num_user, num_item, aggr_mode, dim_latent=None, device=None, features=None):
        super(GCN, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.dim_feat = features.size(1)
        self.dim_latent = dim_latent
        self.aggr_mode = aggr_mode
        self.device = device

        self.preference = nn.Parameter(nn.init.xavier_normal_(
            torch.empty(num_user, self.dim_latent, dtype=torch.float32, requires_grad=True)).to(self.device))

        # 特征投影 (将图像/文本维度转换到统一潜在维度)
        self.MLP = nn.Linear(self.dim_feat, 4 * self.dim_latent)
        self.MLP_1 = nn.Linear(4 * self.dim_latent, self.dim_latent)

        self.conv_embed_1 = Base_gcn(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)

    def forward(self, edge_index, features):
        # 1. 特征投影与拼接
        temp_features = self.MLP_1(F.leaky_relu(self.MLP(features))) if self.dim_latent else features
        x = torch.cat((self.preference, temp_features), dim=0).to(self.device)
        x = F.normalize(x).to(self.device)

        # 2. 多层图卷积 (带有残差连接 x + h1 + h2)
        h = self.conv_embed_1(x, edge_index)
        h_1 = self.conv_embed_1(h, edge_index)

        x_hat = x + h + h_1
        return x_hat, self.preference


class Base_gcn(MessagePassing):
    """拉普拉斯归一化消息传递网络"""

    def __init__(self, in_channels, out_channels, normalize=True, bias=True, aggr='add', **kwargs):
        super(Base_gcn, self).__init__(aggr=aggr, **kwargs)
        self.aggr = aggr

    def forward(self, x, edge_index, size=None):
        if size is None:
            edge_index, _ = remove_self_loops(edge_index)
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

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