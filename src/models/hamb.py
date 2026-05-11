# coding: utf-8
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, degree

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss


class HAMB(GeneralRecommender):
    def __init__(self, config, dataset):
        super(HAMB, self).__init__(config, dataset)

        num_user = self.n_users
        num_item = self.n_items
        batch_size = config['train_batch_size']
        dim_x = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_mm_layers']
        self.knn_k = config['knn_k']
        self.mm_image_weight = config.get('mm_image_weight', 0.5)
        self.temp = config.get('temp', 0.2)

        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.aggr_mode = 'add'
        self.dataset = dataset
        self.dropout = config.get('dropout', 0.1)
        self.reg_weight = config.get('reg_weight', 1e-4)
        self.align_weight = config.get('align_weight', 0.1)  # 超边对齐的权重
        self.dim_latent = 64
        self.dim_feat = 128

        # ================= 一、 初始化与数据准备 =================
        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])

        # 1. 多模态特征嵌入
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)

        # 2. 构建物品-物品相似度图 (kNN Graph)
        mm_adj_file = os.path.join(dataset_path, 'mm_adj_{}.pt'.format(self.knn_k))
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

        # 3. 构建用户-物品交互图 (用于GCN)
        train_interactions = dataset.inter_matrix(form='coo').astype(np.float32)
        edge_index = self.pack_edge_index(train_interactions)
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)

        # 4. 构建超图关联矩阵与拉普拉斯 (用于HGNN)
        self.build_hypergraph(train_interactions)

        # ================= 二、多模态表示学习 GCN模块 =================
        if self.v_feat is not None:
            self.v_gcn = GCN(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode,
                             dim_latent=self.dim_latent,
                             device=self.device, features=self.v_feat)
        if self.t_feat is not None:
            self.t_gcn = GCN(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode,
                             dim_latent=self.dim_latent,
                             device=self.device, features=self.t_feat)

        self.id_feat = nn.Parameter(
            nn.init.xavier_normal_(torch.tensor(np.random.randn(self.n_items, self.dim_latent), dtype=torch.float32,
                                                requires_grad=True), gain=1).to(self.device))
        self.id_gcn = GCN(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode,
                          dim_latent=self.dim_latent, device=self.device, features=self.id_feat)

        # 注意力权重
        self.weight_u = nn.Parameter(nn.init.xavier_normal_(
            torch.tensor(np.random.randn(self.num_user, 2, 1), dtype=torch.float32, requires_grad=True)))

        self.result_embed = nn.Parameter(
            nn.init.xavier_normal_(torch.tensor(np.random.randn(num_user + num_item, dim_x)))).to(self.device)

    def build_hypergraph(self, inter_mat):
        """构建模态超图结构, 计算关联矩阵 H 及度矩阵"""
        # H 的形状为 (|I|, |U|)
        row = torch.tensor(inter_mat.col, dtype=torch.long)  # Items
        col = torch.tensor(inter_mat.row, dtype=torch.long)  # Users (超边)
        values = torch.ones(len(row), dtype=torch.float32)

        # 稀疏关联矩阵 H
        self.H_sparse = torch.sparse_coo_tensor(torch.stack([row, col]), values, (self.num_item, self.num_user)).to(
            self.device)
        self.H_T_sparse = self.H_sparse.transpose(0, 1).coalesce()  # (|U|, |I|)

        # 节点度 D_v 和 超边度 D_e
        D_v_deg = torch.sparse.sum(self.H_sparse, dim=1).to_dense()
        D_e_deg = torch.sparse.sum(self.H_sparse, dim=0).to_dense()

        # 保存归一化因子
        self.D_v_inv_sqrt = torch.pow(D_v_deg + 1e-8, -0.5).view(-1, 1).to(self.device)
        self.D_e_inv = torch.pow(D_e_deg + 1e-8, -1.0).view(-1, 1).to(self.device)

    def get_hyperedge_embed(self, item_rep):
        """提取超边特征 (Mean Pooling): z_u = D_e^{-1} H^T X_i"""
        sum_pool = torch.sparse.mm(self.H_T_sparse, item_rep)  # (|U|, dim)
        return self.D_e_inv * sum_pool

    def hgnn_layer(self, item_rep):
        """超图卷积: \mathcal{L} * X = D_v^{-1/2} H D_e^{-1} H^T D_v^{-1/2} X"""
        temp1 = self.D_v_inv_sqrt * item_rep
        temp2 = torch.sparse.mm(self.H_T_sparse, temp1)
        temp3 = self.D_e_inv * temp2
        temp4 = torch.sparse.mm(self.H_sparse, temp3)
        out = self.D_v_inv_sqrt * temp4
        return out

    # [与 MENTOR 保持一致的方法省略展开：get_knn_adj_mat, compute_normalized_laplacian, pack_edge_index]
    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1).expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        return np.column_stack((rows, cols))

    def InfoNCE(self, view1, view2, temp):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temp)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temp).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score + 1e-8)
        return torch.mean(cl_loss)

    def buildItemGraph(self, h):
        for i in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, h)
        return h

    def forward(self, interaction):
        user_nodes, pos_item_nodes, neg_item_nodes = interaction[0], interaction[1], interaction[2]
        pos_item_nodes += self.n_users
        neg_item_nodes += self.n_users

        # ================= 二、多模态表示学习 (独立GCN演化) =================
        self.v_rep, self.v_preference = self.v_gcn(self.edge_index, self.edge_index, self.v_feat)
        self.t_rep, self.t_preference = self.t_gcn(self.edge_index, self.edge_index, self.t_feat)
        self.id_rep, self.id_preference = self.id_gcn(self.edge_index, self.edge_index, self.id_feat)

        v_user_rep, v_item_rep = self.v_rep[:self.num_user], self.v_rep[self.num_user:]
        t_user_rep, t_item_rep = self.t_rep[:self.num_user], self.t_rep[self.num_user:]
        id_user_rep, id_item_rep = self.id_rep[:self.num_user], self.id_rep[self.num_user:]

        # ================= 超图表示提取与卷积 =================
        # 同步输出视觉超边表示 $z_{u,v}$ 和文本超边表示 $z_{u,t}$
        z_u_v = self.get_hyperedge_embed(v_item_rep)
        z_u_t = self.get_hyperedge_embed(t_item_rep)

        # 物品节点在超图上进行消息传递 (捕捉模态内高阶特征关联)
        v_item_rep = v_item_rep + self.hgnn_layer(v_item_rep)
        t_item_rep = t_item_rep + self.hgnn_layer(t_item_rep)

        # ================= 三、多模态融合与传播 =================
        # 1. 用户侧：注意力权重融合
        v_u = v_user_rep.unsqueeze(1)
        t_u = t_user_rep.unsqueeze(1)
        user_cat = torch.cat((v_u, t_u), dim=1)  # (N, 2, dim)
        weight_u_softmax = F.softmax(self.weight_u, dim=1)  # (N, 2, 1)
        user_fused = (user_cat * weight_u_softmax).sum(dim=1)

        self.user_rep = user_fused + id_user_rep

        # 2. 物品侧：模态图传播 (Item-Item propagation)
        h_v = self.buildItemGraph(v_item_rep)
        h_t = self.buildItemGraph(t_item_rep)

        v_item_fused = v_item_rep + h_v
        t_item_fused = t_item_rep + h_t
        self.item_rep = id_item_rep + v_item_fused + t_item_fused

        self.result_embed = torch.cat((self.user_rep, self.item_rep), dim=0)

        # ================= 六、预测与得分计算 =================
        user_tensor = self.result_embed[user_nodes]
        pos_item_tensor = self.result_embed[pos_item_nodes]
        neg_item_tensor = self.result_embed[neg_item_nodes]

        pos_scores = torch.sum(user_tensor * pos_item_tensor, dim=1)
        neg_scores = torch.sum(user_tensor * neg_item_tensor, dim=1)

        return pos_scores, neg_scores, z_u_v, z_u_t

    def calculate_loss(self, interaction):
        user = interaction[0]
        pos_scores, neg_scores, z_u_v, z_u_t = self.forward(interaction)

        # 1. BPR Loss
        loss_value = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8))

        # 2. 正则化损失 (Reg Loss)
        reg_embedding_loss_v = (self.v_preference[user] ** 2).mean() if self.v_preference is not None else 0.0
        reg_embedding_loss_t = (self.t_preference[user] ** 2).mean() if self.t_preference is not None else 0.0
        reg_loss = self.reg_weight * (reg_embedding_loss_v + reg_embedding_loss_t)
        reg_loss += self.reg_weight * (self.weight_u ** 2).mean()

        # ================= 四、细粒度对齐与对比学习 =================
        # 取同一用户在不同模态下的整体偏好（超边表示）作为正样本，Batch内其他用户作为负样本
        batch_z_u_v = z_u_v[user]
        batch_z_u_t = z_u_t[user]
        align_loss = self.InfoNCE(batch_z_u_v, batch_z_u_t, self.temp) * self.align_weight

        # ================= 五、联合损失优化 =================
        return loss_value + reg_loss + align_loss

    def full_sort_predict(self, interaction):
        user_tensor = self.result_embed[:self.n_users]
        item_tensor = self.result_embed[self.n_users:]

        temp_user_tensor = user_tensor[interaction[0], :]
        score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())
        return score_matrix


class GCN(torch.nn.Module):
    # 与原框架保持一致，执行两层GCN卷积 + 初始残差连接 x + h + h_1
    def __init__(self, datasets, batch_size, num_user, num_item, dim_id, aggr_mode,
                 dim_latent=None, device=None, features=None):
        super(GCN, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.datasets = datasets
        self.dim_id = dim_id
        self.dim_feat = features.size(1)
        self.dim_latent = dim_latent
        self.aggr_mode = aggr_mode
        self.device = device

        if self.dim_latent:
            self.preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
                np.random.randn(num_user, self.dim_latent), dtype=torch.float32, requires_grad=True),
                gain=1).to(self.device))
            self.MLP = nn.Linear(self.dim_feat, 4 * self.dim_latent)
            self.MLP_1 = nn.Linear(4 * self.dim_latent, self.dim_latent)
            self.conv_embed_1 = Base_gcn(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)

        else:
            self.preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
                np.random.randn(num_user, self.dim_feat), dtype=torch.float32, requires_grad=True),
                gain=1).to(self.device))
            self.conv_embed_1 = Base_gcn(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)

    def forward(self, edge_index_drop, edge_index, features, perturbed=False):
        temp_features = self.MLP_1(F.leaky_relu(self.MLP(features))) if self.dim_latent else features
        x = torch.cat((self.preference, temp_features), dim=0).to(self.device)
        x = F.normalize(x).to(self.device)

        h = self.conv_embed_1(x, edge_index)
        h_1 = self.conv_embed_1(h, edge_index)

        x_hat = x + h + h_1
        return x_hat, self.preference


class Base_gcn(MessagePassing):
    # 与原框架保持一致，底层基于 torch_geometric 的消息传递
    def __init__(self, in_channels, out_channels, normalize=True, bias=True, aggr='add', **kwargs):
        super(Base_gcn, self).__init__(aggr=aggr, **kwargs)
        self.aggr = aggr
        self.in_channels = in_channels
        self.out_channels = out_channels

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