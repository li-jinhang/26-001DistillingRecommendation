# coding: utf-8
# 包含 TMPA (粗粒度对齐), TMPB (细粒度对齐) 以及 TMP_MKD (基于注意力的多教师知识蒸馏)

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, degree

from common.abstract_recommender import GeneralRecommender


# ==========================================
# 1. 注意力融合模块 (用于知识蒸馏)
# ==========================================
class AttentionFusion(nn.Module):
    def __init__(self, dim):
        super(AttentionFusion, self).__init__()
        # 使用一个简单的 MLP 计算注意力权重
        self.attn = nn.Sequential(
            nn.Linear(dim * 2, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1)
        )

    def forward(self, stu_emb, teacher_embs):
        """
        stu_emb: [N, D] 学生模型的节点表示
        teacher_embs: list of [N, D], 包含不同教师模型的节点表示
        """
        weights = []
        for t_emb in teacher_embs:
            # 拼接学生和教师的表示来计算当前教师的重要性
            w = self.attn(torch.cat([stu_emb, t_emb], dim=1))  # [N, 1]
            weights.append(w)

        # Softmax 归一化注意力权重
        weights = torch.cat(weights, dim=1)  # [N, num_teachers]
        weights = F.softmax(weights, dim=1)

        # 加权融合教师表示
        fused_emb = 0
        for i, t_emb in enumerate(teacher_embs):
            fused_emb += weights[:, i:i + 1] * t_emb

        return fused_emb


# ==========================================
# 2. 多教师知识蒸馏模型 (TMP_MKD)
# ==========================================
class TMPD(GeneralRecommender):
    def __init__(self, config, dataset):
        super(TMPD, self).__init__(config, dataset)

        self.num_user = self.n_users
        self.num_item = self.n_items
        self.dim_x = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_mm_layers']
        self.knn_k = config['knn_k']
        self.mm_image_weight = config['mm_image_weight']

        self.aggr_mode = 'add'
        self.dataset = dataset
        self.dropout = config['dropout']
        self.reg_weight = config['reg_weight']
        self.align_weight = config['align_weight']
        self.temp = config['temp']
        self.kd_weight = config['kd_weight']  # 新增：知识蒸馏损失的权重
        self.dim_latent = 64

        # --- 共享的图和多模态特征初始化 ---
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

        # --- 初始化三大分支参数 (Teacher A, Teacher B, Student) ---
        self.gcns = nn.ModuleDict()
        self.id_feats = nn.ParameterDict()
        self.weight_us = nn.ParameterDict()

        for branch in ['A', 'B', 'S']:
            if self.v_feat is not None:
                self.gcns[f'{branch}_v'] = GCN(self.num_user, self.num_item, self.aggr_mode, dim_latent=64,
                                               device=self.device, features=self.v_feat)
            if self.t_feat is not None:
                self.gcns[f'{branch}_t'] = GCN(self.num_user, self.num_item, self.aggr_mode, dim_latent=64,
                                               device=self.device, features=self.t_feat)

            id_tensor = torch.tensor(np.random.randn(self.n_items, self.dim_latent), dtype=torch.float32,
                                     requires_grad=True).to(self.device)
            self.id_feats[branch] = nn.Parameter(nn.init.xavier_normal_(id_tensor, gain=1))
            self.gcns[f'{branch}_id'] = GCN(self.num_user, self.num_item, self.aggr_mode, dim_latent=64,
                                            device=self.device, features=self.id_feats[branch])

            w_tensor = torch.tensor(np.random.randn(self.num_user, 2, 1), dtype=torch.float32, requires_grad=True).to(
                self.device)
            self.weight_us[branch] = nn.Parameter(nn.init.xavier_normal_(w_tensor))

        # 蒸馏注意力模块
        actual_embed_dim = self.dim_latent * 2
        self.attn_fusion = AttentionFusion(actual_embed_dim).to(self.device)

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

    def buildItemGraph(self, h):
        for i in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, h)
        return h

    # 核心前向传播逻辑 (支持多分支复用)
    def forward_branch(self, interaction, branch):
        v_gcn = self.gcns[f'{branch}_v']
        t_gcn = self.gcns[f'{branch}_t']
        id_gcn = self.gcns[f'{branch}_id']
        id_feat = self.id_feats[branch]
        weight_u = self.weight_us[branch]

        # 无论是训练还是评估阶段，都至少会包含用户节点
        user_nodes = interaction[0]

        v_rep, v_pref = v_gcn(self.edge_index, self.v_feat)
        t_rep, t_pref = t_gcn(self.edge_index, self.t_feat)
        id_rep, _ = id_gcn(self.edge_index, id_feat)

        representation = torch.cat((v_rep, t_rep), dim=1)
        guide_representation = torch.cat((id_rep, id_rep), dim=1)
        v_representation = torch.cat((v_rep, v_rep), dim=1)
        t_representation = torch.cat((t_rep, t_rep), dim=1)

        u_v, u_t = v_rep[:self.num_user], t_rep[:self.num_user]
        w = F.softmax(weight_u, dim=1).transpose(1, 2)
        user_rep = torch.cat((w[:, :, 0] * u_v, w[:, :, 1] * u_t), dim=1)

        guide_user_rep = torch.cat((id_rep[:self.num_user], id_rep[:self.num_user]), dim=1)
        v_user_rep = torch.cat((u_v, u_v), dim=1)
        t_user_rep = torch.cat((u_t, u_t), dim=1)

        item_rep = representation[self.num_user:] + self.buildItemGraph(representation[self.num_user:])
        guide_item_rep = guide_representation[self.num_user:] + self.buildItemGraph(
            guide_representation[self.num_user:])
        v_item_rep = v_representation[self.num_user:] + self.buildItemGraph(v_representation[self.num_user:])
        t_item_rep = t_representation[self.num_user:] + self.buildItemGraph(t_representation[self.num_user:])

        result_embed = torch.cat((user_rep, item_rep), dim=0)
        result_embed_guide = torch.cat((guide_user_rep, guide_item_rep), dim=0)
        result_embed_v = torch.cat((v_user_rep, v_item_rep), dim=0)
        result_embed_t = torch.cat((t_user_rep, t_item_rep), dim=0)

        out_dict = {
            'result_embed': result_embed,
            'result_embed_guide': result_embed_guide,
            'result_embed_v': result_embed_v,
            'result_embed_t': result_embed_t,
            'v_pref': v_pref,
            't_pref': t_pref,
            'weight_u': weight_u
        }

        # 容错处理：仅在训练阶段（有正负样本时）才计算 pos_scores 和 neg_scores
        try:
            pos_item_nodes, neg_item_nodes = interaction[1], interaction[2]
            pos_item_nodes_offset = pos_item_nodes + self.n_users
            neg_item_nodes_offset = neg_item_nodes + self.n_users

            user_tensor = result_embed[user_nodes]
            pos_item_tensor = result_embed[pos_item_nodes_offset]
            neg_item_tensor = result_embed[neg_item_nodes_offset]

            out_dict['pos_scores'] = torch.sum(user_tensor * pos_item_tensor, dim=1)
            out_dict['neg_scores'] = torch.sum(user_tensor * neg_item_tensor, dim=1)
        except IndexError:
            # 评估阶段 interaction 长度较短，直接跳过得分计算
            pass

        return out_dict

    # 教师A 的粗粒度对齐损失
    def calc_coarse_align(self, out):
        r_var, r_mean = torch.var(out['result_embed']), torch.mean(out['result_embed'])
        g_var, g_mean = torch.var(out['result_embed_guide']), torch.mean(out['result_embed_guide'])
        v_var, v_mean = torch.var(out['result_embed_v']), torch.mean(out['result_embed_v'])
        t_var, t_mean = torch.var(out['result_embed_t']), torch.mean(out['result_embed_t'])

        loss = ((torch.abs(g_var - r_var) + torch.abs(g_mean - r_mean)) +
                (torch.abs(g_var - v_var) + torch.abs(g_mean - v_mean)) +
                (torch.abs(g_var - t_var) + torch.abs(g_mean - t_mean)) +
                (torch.abs(r_var - v_var) + torch.abs(r_mean - v_mean)) +
                (torch.abs(r_var - t_var) + torch.abs(r_mean - t_mean)) +
                (torch.abs(v_var - t_var) + torch.abs(v_mean - t_mean))).mean()
        return loss

    # 教师B 的细粒度对比学习损失
    # 修复后的 calc_infonce_loss 函数
    def calc_infonce_loss(self, embed1, embed2, nodes):
        z1 = F.normalize(embed1[nodes], p=2, dim=1)
        z2 = F.normalize(embed2[nodes], p=2, dim=1)

        pos_sim = torch.sum(z1 * z2, dim=-1) / self.temp
        sim_matrix = torch.matmul(z1, z2.T) / self.temp

        # 使用 logsumexp 避免数值溢出
        loss = (-pos_sim + torch.logsumexp(sim_matrix, dim=1)).mean()
        return loss

    def calc_fine_align(self, out, nodes):
        loss_v_id = self.calc_infonce_loss(out['result_embed_v'], out['result_embed_guide'], nodes)
        loss_t_id = self.calc_infonce_loss(out['result_embed_t'], out['result_embed_guide'], nodes)
        loss_v_t = self.calc_infonce_loss(out['result_embed_v'], out['result_embed_t'], nodes)
        return (loss_v_id + loss_t_id + loss_v_t) / 3.0

    def calculate_loss(self, interaction):
        user = interaction[0]
        pos_item = interaction[1] + self.n_users
        batch_nodes = torch.unique(torch.cat([user, pos_item]))

        # --- 1. 获取三分支前向传播结果 ---
        out_A = self.forward_branch(interaction, 'A')  # Teacher Coarse
        out_B = self.forward_branch(interaction, 'B')  # Teacher Fine
        out_S = self.forward_branch(interaction, 'S')  # Student

        # --- 2. 教师 A 损失 (BPR + 粗粒度对齐) ---
        loss_bpr_A = -torch.mean(torch.log2(torch.sigmoid(out_A['pos_scores'] - out_A['neg_scores']) + 1e-8))
        reg_A = self.reg_weight * ((out_A['v_pref'][user] ** 2).mean() + (out_A['t_pref'][user] ** 2).mean() + (
                    out_A['weight_u'] ** 2).mean())
        loss_A = loss_bpr_A + reg_A + self.align_weight * self.calc_coarse_align(out_A)

        # --- 3. 教师 B 损失 (BPR + 细粒度对齐) ---
        loss_bpr_B = -torch.mean(torch.log2(torch.sigmoid(out_B['pos_scores'] - out_B['neg_scores']) + 1e-8))
        reg_B = self.reg_weight * ((out_B['v_pref'][user] ** 2).mean() + (out_B['t_pref'][user] ** 2).mean() + (
                    out_B['weight_u'] ** 2).mean())
        loss_B = loss_bpr_B + reg_B + self.align_weight * self.calc_fine_align(out_B, batch_nodes)

        # --- 4. 学生网络基础损失 ---
        loss_bpr_S = -torch.mean(torch.log2(torch.sigmoid(out_S['pos_scores'] - out_S['neg_scores']) + 1e-8))
        reg_S = self.reg_weight * ((out_S['v_pref'][user] ** 2).mean() + (out_S['t_pref'][user] ** 2).mean() + (
                    out_S['weight_u'] ** 2).mean())

        # --- 5. 注意力引导的多教师知识蒸馏损失 ---
        # 截断教师梯度，防止学生反向影响教师训练
        emb_A_detach = out_A['result_embed'][batch_nodes].detach()
        emb_B_detach = out_B['result_embed'][batch_nodes].detach()
        emb_S = out_S['result_embed'][batch_nodes]

        # 利用注意力机制动态融合两个教师的 Embeddings
        fused_teacher_emb = self.attn_fusion(emb_S, [emb_A_detach, emb_B_detach])

        # 计算 MSE 蒸馏损失
        loss_kd = F.mse_loss(emb_S, fused_teacher_emb)

        # --- 6. 总损失 ---
        return loss_bpr_S + reg_S + self.kd_weight * loss_kd + loss_A + loss_B

    def full_sort_predict(self, interaction):
        # 推理阶段仅使用学生模型计算打分
        out_S = self.forward_branch(interaction, 'S')
        user_tensor = out_S['result_embed'][:self.n_users]
        item_tensor = out_S['result_embed'][self.n_users:]
        temp_user_tensor = user_tensor[interaction[0], :]
        return torch.matmul(temp_user_tensor, item_tensor.t())


# ==========================================
# 3. 基础组件及原有模型 (保持与原版一致)
# ==========================================
class GCN(torch.nn.Module):
    def __init__(self, num_user, num_item, aggr_mode, dim_latent=64, device=None, features=None):
        super(GCN, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.device = device
        self.dim_feat = features.size(1)
        self.dim_latent = dim_latent

        self.preference = nn.Parameter(nn.init.xavier_normal_(
            torch.tensor(np.random.randn(num_user, self.dim_latent), dtype=torch.float32, requires_grad=True)))

        self.MLP = nn.Linear(self.dim_feat, 4 * self.dim_latent)
        self.MLP_1 = nn.Linear(4 * self.dim_latent, self.dim_latent)
        self.conv_embed_1 = Base_gcn(self.dim_latent, self.dim_latent, aggr=aggr_mode)

    def forward(self, edge_index, features):
        temp_features = self.MLP_1(F.leaky_relu(self.MLP(features)))
        x = torch.cat((self.preference, temp_features), dim=0).to(self.device)
        x = F.normalize(x).to(self.device)

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

# 以下为你提供的原版 TMPA 和 TMPB (方便你需要时单独进行 ablation study)
# ... 为保持结构整洁，省略了原文件中的 TMPA 和 TMPB 详细定义，这部分可直接 copy 原文放在此段下 ...
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
        user_nodes, pos_item_nodes, neg_item_nodes = interaction[0], interaction[1], interaction[2]
        pos_item_nodes_offset = pos_item_nodes + self.n_users
        neg_item_nodes_offset = neg_item_nodes + self.n_users

        # 基础模态表示学习 (使用完整 edge_index)
        self.v_rep, self.v_preference = self.v_gcn(self.edge_index, self.v_feat)
        self.t_rep, self.t_preference = self.t_gcn(self.edge_index, self.t_feat)
        self.id_rep, _ = self.id_gcn(self.edge_index, self.id_feat)

        # 拼接表示
        representation = torch.cat((self.v_rep, self.t_rep), dim=1)
        guide_representation = torch.cat((self.id_rep, self.id_rep), dim=1)
        v_representation = torch.cat((self.v_rep, self.v_rep), dim=1)
        t_representation = torch.cat((self.t_rep, self.t_rep), dim=1)

        # 用户聚合逻辑 (权重融合)
        u_v, u_t = self.v_rep[:self.num_user], self.t_rep[:self.num_user]
        w = F.softmax(self.weight_u, dim=1).transpose(1, 2)
        user_rep = torch.cat((w[:, :, 0] * u_v, w[:, :, 1] * u_t), dim=1)

        # 其他引导用用户表示
        guide_user_rep = torch.cat((self.id_rep[:self.num_user], self.id_rep[:self.num_user]), dim=1)
        v_user_rep = torch.cat((u_v, u_v), dim=1)
        t_user_rep = torch.cat((u_t, u_t), dim=1)

        # 物品图传播 (Item-Item)
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

        # BPR 损失
        loss_value = -torch.mean(torch.log2(torch.sigmoid(pos_scores - neg_scores)))

        # 正则化损失
        reg_loss = self.reg_weight * ((self.v_preference[user] ** 2).mean() +
                                      (self.t_preference[user] ** 2).mean() +
                                      (self.weight_u ** 2).mean())

        # 分布对齐损失 (Alignment Loss)
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
        user_nodes, pos_item_nodes, neg_item_nodes = interaction[0], interaction[1], interaction[2]
        pos_item_nodes_offset = pos_item_nodes + self.n_users
        neg_item_nodes_offset = neg_item_nodes + self.n_users

        # 基础模态表示学习 (使用完整 edge_index)
        self.v_rep, self.v_preference = self.v_gcn(self.edge_index, self.v_feat)
        self.t_rep, self.t_preference = self.t_gcn(self.edge_index, self.t_feat)
        self.id_rep, _ = self.id_gcn(self.edge_index, self.id_feat)

        # 拼接表示
        representation = torch.cat((self.v_rep, self.t_rep), dim=1)
        guide_representation = torch.cat((self.id_rep, self.id_rep), dim=1)
        v_representation = torch.cat((self.v_rep, self.v_rep), dim=1)
        t_representation = torch.cat((self.t_rep, self.t_rep), dim=1)

        # 用户聚合逻辑 (权重融合)
        u_v, u_t = self.v_rep[:self.num_user], self.t_rep[:self.num_user]
        w = F.softmax(self.weight_u, dim=1).transpose(1, 2)
        user_rep = torch.cat((w[:, :, 0] * u_v, w[:, :, 1] * u_t), dim=1)

        # 其他引导用用户表示
        guide_user_rep = torch.cat((self.id_rep[:self.num_user], self.id_rep[:self.num_user]), dim=1)
        v_user_rep = torch.cat((u_v, u_v), dim=1)
        t_user_rep = torch.cat((u_t, u_t), dim=1)

        # 物品图传播 (Item-Item)
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

    # 细粒度对齐：基于 InfoNCE 的对比学习损失
    # 修复后的 calc_infonce_loss 函数
    def calc_infonce_loss(self, embed1, embed2, nodes):
        z1 = F.normalize(embed1[nodes], p=2, dim=1)
        z2 = F.normalize(embed2[nodes], p=2, dim=1)

        pos_sim = torch.sum(z1 * z2, dim=-1) / self.temp
        sim_matrix = torch.matmul(z1, z2.T) / self.temp

        # 使用 logsumexp 避免数值溢出
        loss = (-pos_sim + torch.logsumexp(sim_matrix, dim=1)).mean()
        return loss

    def calculate_loss(self, interaction):
        user = interaction[0]
        pos_item = interaction[1] + self.n_users

        pos_scores, neg_scores = self.forward(interaction)

        # BPR 损失
        loss_value = -torch.mean(torch.log2(torch.sigmoid(pos_scores - neg_scores) + 1e-8))

        # 正则化损失
        reg_loss = self.reg_weight * ((self.v_preference[user] ** 2).mean() +
                                      (self.t_preference[user] ** 2).mean() +
                                      (self.weight_u ** 2).mean())

        # 细粒度分布对齐损失 (Fine-grained Alignment Loss)
        # 提取出本 batch 中涉及到的唯一节点 (包含用户和交互的正物品)
        batch_nodes = torch.unique(torch.cat([user, pos_item]))

        # 在多模态之间进行细粒度对比对齐 (Visual - ID, Text - ID, Visual - Text)
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