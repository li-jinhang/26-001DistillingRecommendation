# coding: utf-8
# 尝试融合用户对齐与MENTOR对齐
# 用bool控制两种对齐
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, degree
from torch_scatter import scatter_max

from common.abstract_recommender import GeneralRecommender
from common.init import xavier_uniform_initialization


class ALLB(GeneralRecommender):
    def __init__(self, config, dataset):
        super(ALLB, self).__init__(config, dataset)

        # 基础参数
        self.num_user = self.n_users
        self.num_item = self.n_items
        self.dim_latent = config.get('embedding_size', 64)
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_mm_layers']
        self.knn_k = config['knn_k']
        self.mm_image_weight = config['mm_image_weight']

        # 损失权重
        self.reg_weight = config['reg_weight']
        self.align_weight = config.get('align_weight', 0.1)
        # 对齐任务控制开关 (默认开启，保证向下兼容)
        self.use_global_align = config.get('use_global_align', True)
        self.use_local_align = config.get('use_local_align', True)
        self.aggr_mode = 'add'
        self.device = config['device']

        # 加载/生成物品相似度图 (Item-Item kNN Graph)
        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        mm_adj_file = os.path.join(dataset_path, 'mm_adj_{}.pt'.format(self.knn_k))

        if os.path.exists(mm_adj_file):
            self.mm_adj = torch.load(mm_adj_file, map_location=self.device)
        else:
            # 简化处理：基于输入特征构建邻接矩阵
            self.mm_adj = self._build_knn_graph()
            torch.save(self.mm_adj, mm_adj_file)

        # 训练交互矩阵转为 edge_index (User-Item Graph)
        train_interactions = dataset.inter_matrix(form='coo').astype(np.float32)
        self.ui_rows_tensor = torch.LongTensor(train_interactions.row).to(self.device)
        self.ui_cols_tensor = torch.LongTensor(train_interactions.col).to(self.device)

        edge_index = self.pack_edge_index(train_interactions)
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)

        # 模态参数初始化
        self.weight_u = nn.Parameter(nn.init.xavier_normal_(
            torch.empty(self.num_user, 2, 1)))

        # 物品纯 ID 特征 (用于引导)
        self.id_feat = nn.Parameter(nn.init.xavier_normal_(
            torch.empty(self.num_item, self.dim_latent)))

        # 初始化各模态 GCN
        if self.v_feat is not None:
            self.v_gcn = GCN(self.num_user, self.num_item, self.aggr_mode, self.dim_latent, self.device, self.v_feat)
        if self.t_feat is not None:
            self.t_gcn = GCN(self.num_user, self.num_item, self.aggr_mode, self.dim_latent, self.device, self.t_feat)

        self.id_gcn = GCN(self.num_user, self.num_item, self.aggr_mode, self.dim_latent, self.device, self.id_feat)

        self.result_embed = None

    def _build_knn_graph(self):
        # 内部逻辑结合 TMPA 的构建方式
        def get_adj(feat):
            fn = F.normalize(feat, p=2, dim=1)
            sim = torch.mm(fn, fn.t())
            _, indices = torch.topk(sim, self.knn_k, dim=-1)
            row = torch.arange(feat.shape[0], device=self.device).view(-1, 1).expand(-1, self.knn_k).flatten()
            col = indices.flatten()
            idx = torch.stack([row, col], dim=0)
            return self.compute_normalized_laplacian(idx, (feat.shape[0], feat.shape[0]))

        v_adj = get_adj(self.v_feat) if self.v_feat is not None else 0
        t_adj = get_adj(self.t_feat) if self.t_feat is not None else 0
        return self.mm_image_weight * v_adj + (1.0 - self.mm_image_weight) * t_adj

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        values = r_inv_sqrt[indices[0]] * r_inv_sqrt[indices[1]]
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.num_user
        return np.column_stack((rows, cols))

    def forward(self, interaction):
        user_nodes = interaction[0]
        pos_item_nodes = interaction[1] + self.num_user
        neg_item_nodes = interaction[2] + self.num_user

        # 1. 模态表示学习 (User-Item GCN)
        self.v_rep, self.v_pref = self.v_gcn(self.edge_index, self.v_feat)
        self.t_rep, self.t_pref = self.t_gcn(self.edge_index, self.t_feat)
        self.id_rep, _ = self.id_gcn(self.edge_index, self.id_feat)

        # 2. 用户自适应融合
        u_v, u_t = self.v_rep[:self.num_user], self.t_rep[:self.num_user]
        w = F.softmax(self.weight_u, dim=1).transpose(1, 2)
        user_rep = torch.cat((w[:, :, 0] * u_v, w[:, :, 1] * u_t), dim=1)

        # 3. 物品跨模态图增强 (Item-Item Graph)
        item_base = torch.cat((self.v_rep[self.num_user:], self.t_rep[self.num_user:]), dim=1)
        item_rep = item_base + self.buildItemGraph(item_base)

        # 存储结果用于 Loss
        self.user_rep, self.item_rep = user_rep, item_rep
        self.result_embed = torch.cat((user_rep, item_rep), dim=0)

        # 计算得分
        pos_scores = torch.sum(self.result_embed[user_nodes] * self.result_embed[pos_item_nodes], dim=1)
        neg_scores = torch.sum(self.result_embed[user_nodes] * self.result_embed[neg_item_nodes], dim=1)
        return pos_scores, neg_scores

    def buildItemGraph(self, h):
        for i in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, h)
        return h

    def calculate_loss(self, interaction):
        user = interaction[0]
        pos_scores, neg_scores = self.forward(interaction)

        # 1. BPR Loss
        loss_bpr = -torch.mean(F.logsigmoid(pos_scores - neg_scores))

        # 2. Regularization
        reg_loss = self.reg_weight * ((self.v_pref[user] ** 2).mean() +
                                      (self.t_pref[user] ** 2).mean() +
                                      (self.weight_u ** 2).mean())

        # 3. 混合对齐损失 (Hybrid Alignment)
        global_align = 0.0
        local_align = 0.0

        # A. 全局分布对齐 (控制开关)
        if self.use_global_align:
            r_var, r_mean = torch.var(self.result_embed), torch.mean(self.result_embed)
            id_var, id_mean = torch.var(self.id_rep), torch.mean(self.id_rep)
            global_align = torch.abs(r_var - id_var) + torch.abs(r_mean - id_mean)

        # B. 细粒度兴趣对齐 (控制开关)
        if self.use_local_align:
            v_item = self.v_rep[self.num_user:]
            t_item = self.t_rep[self.num_user:]

            # Max Pooling: 用户交互过的物品特征
            pool_v_all, _ = scatter_max(v_item[self.ui_cols_tensor], self.ui_rows_tensor, dim=0, dim_size=self.num_user)
            pool_t_all, _ = scatter_max(t_item[self.ui_cols_tensor], self.ui_rows_tensor, dim=0, dim_size=self.num_user)

            u_id_pref = self.id_gcn.preference[user]
            u_v_pool = pool_v_all[user]
            u_t_pool = pool_t_all[user]

            local_align = (torch.abs(torch.var(u_id_pref) - torch.var(u_v_pool)) +
                           torch.abs(torch.mean(u_id_pref) - torch.mean(u_v_pool)) +
                           torch.abs(torch.var(u_id_pref) - torch.var(u_t_pool)) +
                           torch.abs(torch.mean(u_id_pref) - torch.mean(u_t_pool)))

        return loss_bpr + reg_loss + self.align_weight * (global_align + local_align)

    def full_sort_predict(self, interaction):
        user_tensor = self.result_embed[:self.num_user]
        item_tensor = self.result_embed[self.num_user:]
        return torch.matmul(user_tensor[interaction[0]], item_tensor.t())


# 辅助类：GCN 保持一致
class GCN(nn.Module):
    def __init__(self, num_user, num_item, aggr_mode, dim_latent, device, features):
        super(GCN, self).__init__()
        self.device = device
        self.dim_latent = dim_latent
        self.preference = nn.Parameter(nn.init.xavier_normal_(torch.empty(num_user, dim_latent)))

        self.MLP = nn.Sequential(
            nn.Linear(features.shape[1], 4 * dim_latent),
            nn.LeakyReLU(),
            nn.Linear(4 * dim_latent, dim_latent)
        ).to(device)

        self.conv = Base_gcn(dim_latent, dim_latent, aggr=aggr_mode)

    def forward(self, edge_index, features):
        # 归一化输入特征
        features = F.normalize(features, p=2, dim=1)
        item_embed = self.MLP(features)
        x = torch.cat((self.preference, item_embed), dim=0).to(self.device)
        x = F.normalize(x, p=2, dim=1)

        # 两层卷积
        h1 = self.conv(x, edge_index)
        h2 = self.conv(h1, edge_index)
        return x + h1 + h2, self.preference


class Base_gcn(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr='add'):
        super(Base_gcn, self).__init__(aggr=aggr)

    def forward(self, x, edge_index):
        edge_index, _ = remove_self_loops(edge_index)
        return self.propagate(edge_index, x=x, size=(x.size(0), x.size(0)))

    def message(self, x_j, edge_index, size):
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return norm.view(-1, 1) * x_j