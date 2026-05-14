# coding: utf-8
#基本Hinge Encrypt蒸馏
#增加了动态置信度
#专门实现ALLB的自蒸馏
import os
import sys
import types
import torch

# Mock torch_scatter if it doesn't exist
try:
    import torch_scatter
except ImportError:
    module = types.ModuleType('torch_scatter')
    def scatter_max(src, index, dim=-1, out=None, dim_size=None, fill_value=None):
        if dim < 0:
            dim = src.dim() + dim
        if dim_size is None:
            dim_size = int(index.max()) + 1
        if fill_value is None:
            fill_value = torch.finfo(src.dtype).min if src.is_floating_point() else torch.iinfo(src.dtype).min
        if out is None:
            out_shape = list(src.shape)
            out_shape[dim] = dim_size
            out = torch.full(out_shape, fill_value, dtype=src.dtype, device=src.device)
        view_shape = [1] * src.dim()
        view_shape[dim] = index.shape[0]
        index_expanded = index.view(view_shape).expand_as(src)
        out.scatter_reduce_(dim, index_expanded, src, reduce='amax', include_self=True)
        return out, None
    module.scatter_max = scatter_max
    sys.modules['torch_scatter'] = module

import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender
from utils_package.utils import get_model
import numpy as np
import scipy.sparse as sp
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, degree
from common.loss import BPRLoss, EmbLoss


class TeacherWeightAgent(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(TeacherWeightAgent, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        mu = self.net(state)
        std = torch.clamp(self.log_std.exp(), min=1e-3, max=1.0).expand_as(mu)
        dist = torch.distributions.Normal(mu, std)
        return dist

class KDJ(GeneralRecommender):
    def __init__(self, config, dataset):
        # 初始化学生模型 (继承 ALLB，即学生自身使用 ALLB 架构)
        super(KDJ, self).__init__(config, dataset)
        # ==========================================
        # 1. 蒸馏相关的超参数 (新增动态权重)
        # ==========================================
        self.temp = config.get('temp', 0.2)
        if isinstance(self.temp, list): self.temp = self.temp[0]
        self.reg_weight = config.get('reg_weight', 1e-4)
        if isinstance(self.reg_weight, list): self.reg_weight = self.reg_weight[0]
        self.alpha_hinge = config.get('alpha_hinge', 1.0)
        if isinstance(self.alpha_hinge, list): self.alpha_hinge = self.alpha_hinge[0]
        self.beta_ce = config.get('beta_ce', 1.0)
        if isinstance(self.beta_ce, list): self.beta_ce = self.beta_ce[0]

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
        self.device = config['device']
        self.align_weight = config.get('align_weight', 0.1)
        if isinstance(self.align_weight, list):
            self.align_weight = self.align_weight[0]
        self.adapter_weight = config.get('adapter_weight', 0.1)
        if isinstance(self.adapter_weight, list):
            self.adapter_weight = self.adapter_weight[0]
        self.use_global_align = config.get('use_global_align', True)
        self.use_local_align = config.get('use_local_align', True)

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

        # ==========================================
        # 2. 教师模型的加载与冻结 (强制均使用 ALLB 架构)
        # ==========================================
        # --- 教师 1 ---
        teacher1_weight = config['teacher1_weight']
        # 忽略 YAML 中的 teacher1_model，直接以 ALLB 架构实例化
        self.teacher1 = get_model('ALLB')(config, dataset)
        teacher1_path = os.path.abspath(os.path.join(os.getcwd(), 'saved', teacher1_weight))
        self.teacher1.load_state_dict(torch.load(teacher1_path, map_location=self.device), strict=False)
        self.teacher1.eval()
        for param in self.teacher1.parameters():
            param.requires_grad = False

        # --- 教师 2 ---
        teacher2_weight = config['teacher2_weight']
        self.teacher2 = get_model('ALLB')(config, dataset)
        teacher2_path = os.path.abspath(os.path.join(os.getcwd(), 'saved', teacher2_weight))
        self.teacher2.load_state_dict(torch.load(teacher2_path, map_location=self.device), strict=False)
        self.teacher2.eval()
        for param in self.teacher2.parameters():
            param.requires_grad = False

        # ==========================================
        # 3. 初始化强化学习智能体与 Adapter
        # ==========================================
        self.state_dim = (self.dim_latent * 2 + 4) * 2 # 2 teachers, each has user_rep, diff, loss, cos_sim, kl
        self.action_dim = 2
        self.rl_agent = TeacherWeightAgent(self.state_dim, self.action_dim).to(self.device)

        self.adapter_t1 = nn.Sequential(
            nn.Linear(self.dim_latent * 2, self.dim_latent * 2),
            nn.ReLU(),
            nn.Linear(self.dim_latent * 2, self.dim_latent * 2)
        ).to(self.device)
        self.adapter_t2 = nn.Sequential(
            nn.Linear(self.dim_latent * 2, self.dim_latent * 2),
            nn.ReLU(),
            nn.Linear(self.dim_latent * 2, self.dim_latent * 2)
        ).to(self.device)

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
        self.v_rep, self.v_pref = self.v_gcn(self.edge_index, self.v_feat)
        self.t_rep, self.t_pref = self.t_gcn(self.edge_index, self.t_feat)
        self.id_rep, _ = self.id_gcn(self.edge_index, self.id_feat)

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

    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.num_user
        return np.column_stack((rows, cols))

    def buildItemGraph(self, h):
        for i in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, h)
        return h

    def full_sort_predict(self, interaction):
        # 移除了重复的变量分配，使用与前向传播一致的逻辑确保评价时表示的更新
        self.v_rep, _ = self.v_gcn(self.edge_index, self.v_feat)
        self.t_rep, _ = self.t_gcn(self.edge_index, self.t_feat)
        self.id_rep, _ = self.id_gcn(self.edge_index, self.id_feat)

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
        # 1. 学生模型（ALLB架构）前向传播
        pos_scores, neg_scores = self.forward(interaction)

        # ==========================================
        # 2. 计算学生模型的基础 ALLB 损失 (BPR + 混合对齐 + 正则)
        # ==========================================
        loss_bpr = -torch.mean(F.logsigmoid(pos_scores - neg_scores))

        reg_loss = self.reg_weight * ((self.v_pref[user] ** 2).mean() +
                                      (self.t_pref[user] ** 2).mean() +
                                      (self.weight_u ** 2).mean())

        global_align = 0.0
        local_align = 0.0

        if self.use_global_align:
            r_var, r_mean = torch.var(self.result_embed), torch.mean(self.result_embed)
            id_var, id_mean = torch.var(self.id_rep), torch.mean(self.id_rep)
            global_align = torch.abs(r_var - id_var) + torch.abs(r_mean - id_mean)

        if self.use_local_align:
            from torch_scatter import scatter_max  # 确保使用了 scatter_max
            v_item = self.v_rep[self.num_user:]
            t_item = self.t_rep[self.num_user:]

            pool_v_all, _ = scatter_max(v_item[self.ui_cols_tensor], self.ui_rows_tensor, dim=0, dim_size=self.num_user)
            pool_t_all, _ = scatter_max(t_item[self.ui_cols_tensor], self.ui_rows_tensor, dim=0, dim_size=self.num_user)

            u_id_pref = self.id_gcn.preference[user]
            u_v_pool = pool_v_all[user]
            u_t_pool = pool_t_all[user]

            local_align = (torch.abs(torch.var(u_id_pref) - torch.var(u_v_pool)) +
                           torch.abs(torch.mean(u_id_pref) - torch.mean(u_v_pool)) +
                           torch.abs(torch.var(u_id_pref) - torch.var(u_t_pool)) +
                           torch.abs(torch.mean(u_id_pref) - torch.mean(u_t_pool)))

        student_base_loss = loss_bpr + reg_loss + self.align_weight * (global_align + local_align)

        # ==========================================
        # 3. 教师模型前向传播 (均采用 ALLB，使用克隆对象安全获取预测)
        # ==========================================
        with torch.no_grad():
            t1_pos, t1_neg = self.teacher1.forward(interaction.clone())
            t2_pos, t2_neg = self.teacher2.forward(interaction.clone())
            
            u_t1 = self.teacher1.result_embed[user]
            u_t2 = self.teacher2.result_embed[user]

        s_diff = pos_scores - neg_scores
        t1_diff = t1_pos - t1_neg
        t2_diff = t2_pos - t2_neg

        # ==========================================
        # 4. 提取特征构建 State 并采样 Action
        # ==========================================
        s_loss_sample = -F.logsigmoid(s_diff)
        t1_loss_sample = -F.logsigmoid(t1_diff)
        t2_loss_sample = -F.logsigmoid(t2_diff)

        s_user_emb = self.result_embed[user]
        s_user_aligned_t1 = self.adapter_t1(s_user_emb)
        s_user_aligned_t2 = self.adapter_t2(s_user_emb)

        with torch.no_grad():
            cos_sim_t1 = F.cosine_similarity(s_user_aligned_t1, u_t1, dim=-1)
            cos_sim_t2 = F.cosine_similarity(s_user_aligned_t2, u_t2, dim=-1)

            p_s = torch.sigmoid(s_diff / self.temp)
            p_t1 = torch.sigmoid(t1_diff / self.temp)
            p_t2 = torch.sigmoid(t2_diff / self.temp)

            eps = 1e-8
            kl_t1 = p_t1 * torch.log((p_t1 + eps) / (p_s + eps)) + (1 - p_t1) * torch.log((1 - p_t1 + eps) / (1 - p_s + eps))
            kl_t2 = p_t2 * torch.log((p_t2 + eps) / (p_s + eps)) + (1 - p_t2) * torch.log((1 - p_t2 + eps) / (1 - p_s + eps))

            state_t1 = torch.cat([
                u_t1,
                t1_diff.unsqueeze(1),
                t1_loss_sample.unsqueeze(1),
                cos_sim_t1.unsqueeze(1),
                kl_t1.unsqueeze(1)
            ], dim=1)

            state_t2 = torch.cat([
                u_t2,
                t2_diff.unsqueeze(1),
                t2_loss_sample.unsqueeze(1),
                cos_sim_t2.unsqueeze(1),
                kl_t2.unsqueeze(1)
            ], dim=1)

            state = torch.cat([state_t1, state_t2], dim=1)

        action_dist = self.rl_agent(state)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action).sum(dim=-1)

        # 动作转化为非负且和为1的权重向量
        weights = F.softmax(action, dim=-1)
        w1 = weights[:, 0]
        w2 = weights[:, 1]

        # ==========================================
        # 5. 加权的蒸馏损失
        # ==========================================
        hinge_loss_1_sample = torch.clamp(t1_diff - s_diff, min=0.0)
        hinge_loss_2_sample = torch.clamp(t2_diff - s_diff, min=0.0)
        hinge_loss = self.alpha_hinge * torch.mean(w1 * hinge_loss_1_sample + w2 * hinge_loss_2_sample)

        t1_bar = torch.sigmoid(t1_diff / self.temp)
        t2_bar = torch.sigmoid(t2_diff / self.temp)
        s_bar = torch.sigmoid(s_diff / self.temp)

        ce_loss_1_sample = -(t1_bar * torch.log(s_bar + eps) + (1 - t1_bar) * torch.log(1 - s_bar + eps))
        ce_loss_2_sample = -(t2_bar * torch.log(s_bar + eps) + (1 - t2_bar) * torch.log(1 - s_bar + eps))
        ce_loss = self.beta_ce * torch.mean(w1 * ce_loss_1_sample + w2 * ce_loss_2_sample)

        total_distill_loss = hinge_loss + ce_loss

        # ==========================================
        # 6. 计算 Adapter 损失与 RL 策略损失
        # ==========================================
        # 师生对比损失 (即 Adapter 的均方误差)
        adapter_loss = F.mse_loss(s_user_aligned_t1, u_t1.detach()) + \
                       F.mse_loss(s_user_aligned_t2, u_t2.detach())

        # 奖励设计：加权后的蒸馏损失和师生对比损失的总和取负值
        reward = - (w1 * hinge_loss_1_sample + w2 * hinge_loss_2_sample +
                    w1 * ce_loss_1_sample + w2 * ce_loss_2_sample).detach()

        # Baseline
        reward = (reward - reward.mean()) / (reward.std() + 1e-8)

        rl_loss = -torch.mean(log_prob * reward)

        # 最终损失
        return student_base_loss + total_distill_loss + rl_loss + self.adapter_weight * adapter_loss

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
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            return norm.view(-1, 1) * x_j
        return x_j

    def update(self, aggr_out):
        return aggr_out

