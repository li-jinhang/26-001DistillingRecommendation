#注意力机制聚合尝试  对齐优化InfoNCE
# coding: utf-8
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
from torch_geometric.utils import softmax as pyg_softmax
import torch_geometric

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss
from common.init import xavier_uniform_initialization


class MADE(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MADE, self).__init__(config, dataset)

        num_user = self.n_users
        num_item = self.n_items
        dim_x = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_mm_layers']
        self.knn_k = config['knn_k']
        self.mm_image_weight = config['mm_image_weight']

        # 对齐损失的超参数权重
        self.align_weight = config.get('align_weight', 0.01)

        self.num_user = num_user
        self.num_item = num_item
        self.k = 40
        self.aggr_mode = 'add'
        self.dataset = dataset
        self.dropout = config['dropout']
        self.reg_weight = config['reg_weight']
        self.temp = config['temp']

        self.v_rep = None
        self.t_rep = None
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

        # ========= ID 特征与初始化 =========
        # 为物品初始化纯 ID Embedding
        self.id_feat = nn.Parameter(nn.init.xavier_normal_(
            torch.empty(self.num_item, self.dim_latent))).to(self.device)

        # 初始化各模态及 ID 的 GCN
        if self.v_feat is not None:
            self.v_gcn = GCN(self.num_user, self.num_item, self.aggr_mode, dim_latent=self.dim_latent,
                             device=self.device, features=self.v_feat)
        if self.t_feat is not None:
            self.t_gcn = GCN(self.num_user, self.num_item, self.aggr_mode, dim_latent=self.dim_latent,
                             device=self.device, features=self.t_feat)

        self.id_gcn = GCN(self.num_user, self.num_item, self.aggr_mode, dim_latent=self.dim_latent,
                          device=self.device, features=self.id_feat)

        # ========= 注意力网络初始化 =========
        ui_rows = train_interactions.row
        ui_cols = train_interactions.col
        ui_indices = torch.LongTensor(np.vstack((ui_rows, ui_cols)))

        # 保存索引用于动态稀疏注意力计算
        self.ui_indices = ui_indices.to(self.device)

        # 定义计算注意力得分的 MLP 网络
        self.attn_mlp_id = nn.Sequential(
            nn.Linear(self.dim_latent * 2, self.dim_latent),
            nn.LeakyReLU(0.2),
            nn.Linear(self.dim_latent, 1)
        ).to(self.device)

        self.attn_mlp_v = nn.Sequential(
            nn.Linear(self.dim_latent * 2, self.dim_latent),
            nn.LeakyReLU(0.2),
            nn.Linear(self.dim_latent, 1)
        ).to(self.device)

        self.attn_mlp_t = nn.Sequential(
            nn.Linear(self.dim_latent * 2, self.dim_latent),
            nn.LeakyReLU(0.2),
            nn.Linear(self.dim_latent, 1)
        ).to(self.device)

        # 结果 Embedding 初始化
        self.result_embed = nn.Parameter(
            nn.init.xavier_normal_(torch.tensor(np.random.randn(num_user + num_item, dim_x)))).to(self.device)

    def sparse_attention_pooling(self, user_embs, item_embs, ui_indices, attn_network):
        """
        基于注意力机制的稀疏加权聚合
        """
        users = ui_indices[0]
        items = ui_indices[1]

        # 1. 获取存在交互边的 user 和 item 的特征
        u_features = user_embs[users]
        i_features = item_embs[items]

        # 2. 拼接特征并计算未归一化的注意力得分 (e_ui)
        cat_features = torch.cat([u_features, i_features], dim=-1)
        attn_scores = attn_network(cat_features).squeeze(-1)  # [num_edges]

        # 3. 对每个用户的局部邻居节点内部进行 softmax 归一化 (alpha_ui)
        attn_weights = pyg_softmax(attn_scores, index=users, num_nodes=self.num_user)

        # 4. 构建带注意力权重的稀疏邻接矩阵
        size = torch.Size([self.num_user, self.num_item])
        attn_adj = torch.sparse.FloatTensor(ui_indices, attn_weights, size)

        # 5. 利用稀疏矩阵乘法，高效完成所有用户的加权聚合
        pooled_embs = torch.sparse.mm(attn_adj, item_embs)

        return pooled_embs

    # (原有的辅助函数保持不变)
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

    def info_nce_loss(self, z1, z2):
        """
        计算两个视图（模态）表示之间的 InfoNCE 损失 (对比学习)
        拉近同一个用户的不同模态表示（正样本），推开不同用户的模态表示（负样本）
        """
        # L2 归一化
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)

        # 计算余弦相似度矩阵 (内积) / temperature
        sim_matrix = torch.matmul(z1, z2.t()) / self.temp

        # 标签：对角线上的元素互为正样本（同一用户）
        labels = torch.arange(sim_matrix.size(0)).long().to(self.device)

        # 对称的对比损失 (z1 -> z2 和 z2 -> z1)
        loss_1 = F.cross_entropy(sim_matrix, labels)
        loss_2 = F.cross_entropy(sim_matrix.t(), labels)

        return (loss_1 + loss_2) * 0.5

    def calculate_alignment_loss(self, users):
        # 1. 经过多个GCN实例分别处理特征 (获取全量传播后的嵌入)
        id_rep, _ = self.id_gcn(self.edge_index, self.id_feat)
        v_rep, _ = self.v_gcn(self.edge_index, self.v_feat)
        t_rep, _ = self.t_gcn(self.edge_index, self.t_feat)

        # 分离 user 和 item 的表征
        id_user, id_item = id_rep[:self.num_user], id_rep[self.num_user:]
        v_user, v_item = v_rep[:self.num_user], v_rep[self.num_user:]
        t_user, t_item = t_rep[:self.num_user], t_rep[self.num_user:]

        # 2. 使用动态稀疏注意力池化
        # 针对每个模态，计算用户对不同物品的注意力权重并加权聚合
        pool_id_all = self.sparse_attention_pooling(id_user, id_item, self.ui_indices, self.attn_mlp_id)
        pool_v_all = self.sparse_attention_pooling(v_user, v_item, self.ui_indices, self.attn_mlp_v)
        pool_t_all = self.sparse_attention_pooling(t_user, t_item, self.ui_indices, self.attn_mlp_t)

        # 提取当前批次用户的聚合表示
        pool_id = pool_id_all[users]
        pool_v = pool_v_all[users]
        pool_t = pool_t_all[users]

        # 3. 执行对比学习对齐 (InfoNCE Loss)
        # 消除单一模态带来的偏好偏差，拉近 ID、视觉、文本 在同一个用户下的表征距离

        # (1) ID 空间与 视觉感知 对齐
        loss_id_v = self.info_nce_loss(pool_id, pool_v)

        # (2) ID 空间与 文本理解 对齐
        loss_id_t = self.info_nce_loss(pool_id, pool_t)

        # (3) 视觉 与 文本 相互对齐，保证逻辑一致性
        loss_v_t = self.info_nce_loss(pool_v, pool_t)

        # 汇总对齐损失
        align_loss = loss_id_v + loss_id_t + loss_v_t
        return self.align_weight * align_loss

    def forward(self, interaction):
        user_nodes, pos_item_nodes, neg_item_nodes = interaction[0], interaction[1], interaction[2]
        pos_item_nodes += self.n_users
        neg_item_nodes += self.n_users

        # 基础模态表示学习 (使用完整 edge_index)
        self.v_rep, self.v_preference = self.v_gcn(self.edge_index, self.v_feat)
        self.t_rep, self.t_preference = self.t_gcn(self.edge_index, self.t_feat)

        # 拼接表示
        representation = torch.cat((self.v_rep, self.t_rep), dim=1)

        # 用户聚合逻辑 (权重融合)
        u_v, u_t = self.v_rep[:self.num_user], self.t_rep[:self.num_user]
        w = F.softmax(self.weight_u, dim=1).transpose(1, 2)
        user_rep = torch.cat((w[:, :, 0] * u_v, w[:, :, 1] * u_t), dim=1)

        # 物品图传播 (Item-Item)
        item_rep = representation[self.num_user:] + self.buildItemGraph(representation[self.num_user:])

        # 最终 Embedding 拼接
        self.user_rep, self.item_rep = user_rep, item_rep
        self.result_embed = torch.cat((user_rep, item_rep), dim=0)

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

    def calculate_loss(self, interaction):
        user = interaction[0]
        pos_scores, neg_scores = self.forward(interaction)

        # BPR 损失
        loss_value = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        # 正则化损失
        reg_loss = self.reg_weight * ((self.v_preference[user] ** 2).mean() +
                                      (self.t_preference[user] ** 2).mean() +
                                      (self.weight_u ** 2).mean())

        # ========= 新增：计算并叠加对齐损失 =========
        align_loss = self.calculate_alignment_loss(user)

        return loss_value + reg_loss + align_loss

    def full_sort_predict(self, interaction):
        user_tensor = self.result_embed[:self.n_users]
        item_tensor = self.result_embed[self.n_users:]
        temp_user_tensor = user_tensor[interaction[0], :]
        return torch.matmul(temp_user_tensor, item_tensor.t())

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
        features = F.normalize(features, p=2, dim=1)
        temp_features = self.MLP_1(F.leaky_relu(self.MLP(features)))
        x = torch.cat((self.preference, temp_features), dim=0).to(self.device)
        x = F.normalize(x).to(self.device)

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