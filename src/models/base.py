# coding: utf-8

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree

# 假设这些是你框架中定义的基础类
from common.abstract_recommender import GeneralRecommender


class LightGCNConv(MessagePassing):
    """用于交互图的轻量级图卷积层 (LightGCN风格)"""

    def __init__(self):
        super(LightGCNConv, self).__init__(aggr='add')

    def forward(self, x, edge_index):
        # 计算归一化系数 1 / sqrt(d_i * d_j)
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class BASE(GeneralRecommender):
    def __init__(self, config, dataset):
        super(BASE, self).__init__(config, dataset)

        self.num_user = self.n_users
        self.num_item = self.n_items
        self.dim_latent = config['embedding_size']

        # 超参数
        self.knn_k = config.get('knn_k', 10)
        self.alpha_v = config.get('alpha_v', 0.5)  # 视觉模态图融合权重
        self.lambda_mm = config.get('lambda_mm', 0.1)  # 多模态损失权重
        self.reg_weight = config.get('reg_weight', 1e-4)
        self.L_ii = config.get('L_ii', 2)  # 物品图传播层数
        self.L_ui = config.get('L_ui', 3)  # 交互图传播层数

        # 一、 初始化与数据准备
        # 1. ID 嵌入初始化
        self.user_embedding = nn.Embedding(self.num_user, self.dim_latent)
        self.item_embedding = nn.Embedding(self.num_item, self.dim_latent)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # 2. 多模态投影 (单层 MLP 映射到 ID 隐向量空间)
        if self.v_feat is not None and self.t_feat is not None:
            self.v_feat = self.v_feat.to(self.device)
            self.t_feat = self.t_feat.to(self.device)
            self.mm_mlp = nn.Linear(self.v_feat.shape[1] + self.t_feat.shape[1], self.dim_latent)

        # 3. 构建并冻结物品-物品相似度图 S
        self.frozen_item_graph = self._build_frozen_item_graph().to(self.device)

        # 4. 交互数据准备
        train_interactions = dataset.inter_matrix(form='coo').astype(np.float32)
        edge_index = self._pack_edge_index(train_interactions)
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)
        # 构建对称邻接矩阵所需的边
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)

        # GCN 卷积层
        self.ui_conv = LightGCNConv()

    def _pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.num_user
        return np.column_stack((rows, cols))

    def _get_knn_adj(self, features, k):
        """计算 kNN 图并返回无权稀疏矩阵"""
        # 1. 计算相似度并获取 Top-k 索引
        features_norm = F.normalize(features, p=2, dim=1)
        sim = torch.mm(features_norm, features_norm.t())
        _, knn_ind = torch.topk(sim, k, dim=-1)

        # 2. 构建初始稀疏矩阵
        row = torch.arange(knn_ind.shape[0]).view(-1, 1).expand(-1, k).flatten().to(self.device)
        col = knn_ind.flatten().to(self.device)
        indices = torch.stack([row, col])
        values = torch.ones_like(row, dtype=torch.float32)

        adj = torch.sparse_coo_tensor(indices, values, (self.num_item, self.num_item)).coalesce()

        # 3. 保证对称性 (adj + adj.t)
        # 注意：相加后权重可能变成 2 (如果 i->j 和 j->i 同时存在)
        adj = (adj + adj.t()).coalesce()

        # 4. 离散化处理：将所有连接权重重置为 1 (根据流程文档：连接为 1，否则为 0)
        # 此时 adj 已经是稀疏格式，我们只需要取其索引，重新生成全 1 的 values
        final_indices = adj.indices()
        final_values = torch.ones_like(adj.values(), dtype=torch.float32)

        return torch.sparse_coo_tensor(final_indices, final_values, (self.num_item, self.num_item)).coalesce()

    def _build_frozen_item_graph(self):
        """构建冻结的多模态物品图 S"""
        adj_v = self._get_knn_adj(self.v_feat, self.knn_k)
        adj_t = self._get_knn_adj(self.t_feat, self.knn_k)

        # 模态聚合加权
        S_adj = self.alpha_v * adj_v + (1.0 - self.alpha_v) * adj_t
        S_adj = S_adj.coalesce()

        # 拉普拉斯归一化 (Laplacian Normalization)
        row_sum = torch.sparse.sum(S_adj, dim=1).to_dense()
        d_inv_sqrt = torch.pow(row_sum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.

        indices = S_adj.indices()
        values = S_adj.values()
        norm_values = d_inv_sqrt[indices[0]] * values * d_inv_sqrt[indices[1]]

        return torch.sparse_coo_tensor(indices, norm_values, (self.num_item, self.num_item)).coalesce()

    def degree_sensitive_prune(self, edge_index):
        """三、 用户-物品图结构去噪 (动态生成子图)"""
        row, col = edge_index

        # 计算节点度数
        deg_u = degree(row, num_nodes=self.num_user + self.num_item)
        deg_i = degree(col, num_nodes=self.num_user + self.num_item)

        # 计算保留概率 p_ui = 1 / sqrt(w_u * w_i)
        p_ui = 1.0 / (torch.sqrt(deg_u[row]) * torch.sqrt(deg_i[col]))
        p_ui = torch.clamp(p_ui, max=1.0)  # 确保概率 <= 1

        # 概率采样剪枝
        keep_mask = torch.bernoulli(p_ui).bool()
        pruned_edge_index = edge_index[:, keep_mask]

        return pruned_edge_index

    def forward(self, edge_index):
        # 初始 ID 表示
        all_emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)

        # ---- 四、 多模态融合与双图传播 ----
        # 1. 交互视图 (在去噪后的二分图上进行传播)
        ui_embs = [all_emb]
        for _ in range(self.L_ui):
            all_emb = self.ui_conv(all_emb, edge_index)
            ui_embs.append(all_emb)
        ui_embs = torch.stack(ui_embs, dim=1).mean(dim=1)
        h_u, h_i_int = torch.split(ui_embs, [self.num_user, self.num_item])

        # 2. 语义视图 (在冻结的物品图上进行传播)
        h_i_sem = self.item_embedding.weight
        sem_embs = [h_i_sem]
        for _ in range(self.L_ii):
            h_i_sem = torch.sparse.mm(self.frozen_item_graph, h_i_sem)
            sem_embs.append(h_i_sem)
        h_i_sem = torch.stack(sem_embs, dim=1).mean(dim=1)

        # 最终物品表示 = 语义视图 + 交互视图
        h_i = h_i_sem + h_i_int

        return h_u, h_i

    def calculate_loss(self, interaction):
        user = interaction[0]
        pos_item = interaction[1]
        neg_item = interaction[2]

        # 动态去噪交互图
        pruned_edge_index = self.degree_sensitive_prune(self.edge_index)

        # 获取传播后的特征
        h_u, h_i = self.forward(pruned_edge_index)

        u_emb = h_u[user]
        pos_emb = h_i[pos_item]
        neg_emb = h_i[neg_item]

        # ---- 五、 联合损失优化 ----
        # 1. 主 BPR 损失
        pos_scores = torch.sum(u_emb * pos_emb, dim=1)
        neg_scores = torch.sum(u_emb * neg_emb, dim=1)
        bpr_loss = -torch.mean(torch.nn.functional.logsigmoid(pos_scores - neg_scores))

        # 2. 多模态重构损失
        # 投影多模态特征
        concat_mm_feat = torch.cat([self.v_feat, self.t_feat], dim=-1)
        h_i_m = self.mm_mlp(concat_mm_feat)

        # 原始用户 ID 嵌入
        u_emb_0 = self.user_embedding.weight[user]
        pos_m_emb = h_i_m[pos_item]
        neg_m_emb = h_i_m[neg_item]

        pos_m_scores = torch.sum(u_emb_0 * pos_m_emb, dim=1)
        neg_m_scores = torch.sum(u_emb_0 * neg_m_emb, dim=1)
        mm_loss = -torch.mean(torch.nn.functional.logsigmoid(pos_m_scores - neg_m_scores))

        # 正则化损失
        reg_loss = self.reg_weight * (u_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2)) / float(
            len(user))

        # 联合总损失
        total_loss = bpr_loss + self.lambda_mm * mm_loss + reg_loss
        return total_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]

        # 推理阶段使用完整的图（不剪枝）
        h_u, h_i = self.forward(self.edge_index)

        # ---- 六、 预测与排序 ----
        # 使用 ID 加总的综合表示进行打分，不再依赖训练时的多模态原始特征
        user_tensor = h_u[user, :]
        score_matrix = torch.matmul(user_tensor, h_i.t())

        return score_matrix