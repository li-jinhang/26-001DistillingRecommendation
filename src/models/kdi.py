# coding: utf-8
#基本Hinge Encrypt蒸馏
#改回了自适应权重
#专门实现ALLB的自蒸馏哦

#基于KDF修改的强化学习蒸馏
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
        if dim_size is None:
            dim_size = int(index.max()) + 1
        index_expanded = index.unsqueeze(-1).expand_as(src)
        out = torch.zeros((dim_size, src.size(-1)), dtype=src.dtype, device=src.device)
        out.scatter_reduce_(dim, index_expanded, src, reduce='amax', include_self=False)
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


class KDI(GeneralRecommender):
    def __init__(self, config, dataset):
        super(KDI, self).__init__(config, dataset)
        # ==========================================
        # 1. 蒸馏相关的超参数 (保持不变)
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
        # 2. 学生模型参数定义 (保持不变)
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
        self.use_global_align = config.get('use_global_align', True)
        self.use_local_align = config.get('use_local_align', True)

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        mm_adj_file = os.path.join(dataset_path, 'mm_adj_{}.pt'.format(self.knn_k))

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)

        if os.path.exists(mm_adj_file):
            self.mm_adj = torch.load(mm_adj_file, map_location=self.device)
            if not self.mm_adj.is_coalesced():
                self.mm_adj = self.mm_adj.coalesce()
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
            self.mm_adj = self.mm_adj.coalesce()
            torch.save(self.mm_adj, mm_adj_file)

        train_interactions = dataset.inter_matrix(form='coo').astype(np.float32)
        self.ui_rows_tensor = torch.LongTensor(train_interactions.row).to(self.device)
        self.ui_cols_tensor = torch.LongTensor(train_interactions.col).to(self.device)

        edge_index = self.pack_edge_index(train_interactions)
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)

        self.weight_u = nn.Parameter(nn.init.xavier_normal_(torch.empty(self.num_user, 2, 1)))
        self.id_feat = nn.Parameter(nn.init.xavier_normal_(torch.empty(self.num_item, self.dim_latent)))

        if self.v_feat is not None:
            self.v_gcn = GCN(self.num_user, self.num_item, self.aggr_mode, self.dim_latent, self.device, self.v_feat)
        if self.t_feat is not None:
            self.t_gcn = GCN(self.num_user, self.num_item, self.aggr_mode, self.dim_latent, self.device, self.t_feat)
        self.id_gcn = GCN(self.num_user, self.num_item, self.aggr_mode, self.dim_latent, self.device, self.id_feat)

        self.result_embed = None
        self.eval_result_embed = None  # 用于 full_sort_predict 缓存

        # 构建用于局部对齐的 User-Item 聚合稀疏矩阵 (Mean Pooling)
        from torch_geometric.utils import degree
        deg_u = degree(self.ui_rows_tensor, self.num_user, dtype=torch.float)
        deg_u_inv = deg_u.pow(-1.0)
        deg_u_inv[torch.isinf(deg_u_inv)] = 0.0
        val = deg_u_inv[self.ui_rows_tensor]
        indices = torch.stack([self.ui_rows_tensor, self.ui_cols_tensor], dim=0)
        self.ui_pool_mat = torch.sparse_coo_tensor(indices, val, (self.num_user, self.num_item)).to(self.device)

        # ==========================================
        # 3. 教师表征缓存
        # ==========================================
        # 直接从配置中的教师权重构建缓存，避免依赖不存在的离线随机特征表。
        t1_name = str(config.get('teacher1_model', 'ALLB'))
        t1_weight = config.get('teacher1_weight')
        t2_name = str(config.get('teacher2_model', 'ALLB'))
        t2_weight = config.get('teacher2_weight')
        if not t1_weight or not t2_weight:
            raise RuntimeError('KDI 缺少教师模型配置，请在 KDI.yaml 中提供 teacher1_weight 和 teacher2_weight。')

        t1_user_emb, t1_item_emb = self._build_teacher_embedding_cache(t1_name, t1_weight, config, dataset)
        t2_user_emb, t2_item_emb = self._build_teacher_embedding_cache(t2_name, t2_weight, config, dataset)

        self.t1_user_emb = nn.Embedding.from_pretrained(t1_user_emb, freeze=True)
        self.t1_item_emb = nn.Embedding.from_pretrained(t1_item_emb, freeze=True)
        self.t2_user_emb = nn.Embedding.from_pretrained(t2_user_emb, freeze=True)
        self.t2_item_emb = nn.Embedding.from_pretrained(t2_item_emb, freeze=True)

        # 【新增】多粒度对齐 Adapter (解决余弦相似度计算时的维度不匹配)
        student_emb_dim = self.dim_latent * 2  # 基于前向传播中的 cat(v_rep, t_rep)
        t1_emb_dim = self.t1_user_emb.weight.shape[1]
        t2_emb_dim = self.t2_user_emb.weight.shape[1]

        self.adapter_t1 = nn.Sequential(
            nn.Linear(student_emb_dim, t1_emb_dim),
            nn.ReLU(),
            nn.Linear(t1_emb_dim, t1_emb_dim)
        ).to(self.device)

        self.adapter_t2 = nn.Sequential(
            nn.Linear(student_emb_dim, t2_emb_dim),
            nn.ReLU(),
            nn.Linear(t2_emb_dim, t2_emb_dim)
        ).to(self.device)

        # ==========================================
        # 4. 强化学习(RL) Agent 配置与初始化
        # ==========================================
        self.state_dim = 7
        self.action_dim = 2
        self.rl_agent = TeacherWeightAgent(self.state_dim, self.action_dim).to(self.device)
        # 此处删除了单独的 self.rl_optimizer，让外部系统的通用优化器统一管理 model.parameters() 即可

    def _resolve_teacher_weight_path(self, weight_path):
        if os.path.isabs(weight_path):
            if os.path.exists(weight_path):
                return weight_path
            raise FileNotFoundError(f'KDI 教师权重不存在: {weight_path}')

        cwd = os.getcwd()
        candidates = [
            os.path.abspath(weight_path),
            os.path.abspath(os.path.join(cwd, weight_path)),
            os.path.abspath(os.path.join(cwd, 'saved', weight_path)),
            os.path.abspath(os.path.join(os.path.dirname(__file__), '..', weight_path)),
            os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'saved', weight_path)),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(f'KDI 教师权重不存在: {candidates[2]}')

    @staticmethod
    def _extract_state_dict(raw_state):
        state = raw_state
        if isinstance(state, dict) and 'state_dict' in state and isinstance(state['state_dict'], dict):
            state = state['state_dict']
        if not isinstance(state, dict):
            raise RuntimeError(f'KDI 教师权重格式非法: {type(raw_state)}')

        normalized = {}
        for key, value in state.items():
            if isinstance(key, str) and key.startswith('module.'):
                normalized[key[7:]] = value
            else:
                normalized[key] = value
        return normalized

    @staticmethod
    def _filter_compatible_state_dict(model_state, loaded_state):
        compatible = {}
        unexpected_keys = []
        shape_mismatch_keys = []
        for key, value in loaded_state.items():
            if key not in model_state:
                unexpected_keys.append(key)
                continue
            if tuple(model_state[key].shape) != tuple(value.shape):
                shape_mismatch_keys.append(key)
                continue
            compatible[key] = value
        return compatible, unexpected_keys, shape_mismatch_keys

    def _build_dummy_interaction(self):
        user = torch.zeros(1, dtype=torch.long, device=self.device)
        pos_item = torch.zeros(1, dtype=torch.long, device=self.device)
        neg_item = torch.zeros(1, dtype=torch.long, device=self.device)
        return (user, pos_item, neg_item)

    def _build_teacher_embedding_cache(self, teacher_name, teacher_weight, config, dataset):
        teacher_path = self._resolve_teacher_weight_path(str(teacher_weight))
        teacher_model = get_model(teacher_name)(config, dataset).to(self.device)

        raw_state = torch.load(teacher_path, map_location=self.device)
        loaded_state = self._extract_state_dict(raw_state)
        compatible_state, unexpected_keys, shape_mismatch_keys = self._filter_compatible_state_dict(
            teacher_model.state_dict(), loaded_state
        )
        if shape_mismatch_keys:
            raise RuntimeError(
                f'KDI 教师 `{teacher_name}` 权重维度不匹配: {shape_mismatch_keys[:5]}'
            )
        if unexpected_keys:
            print(f'WARNING: KDI 教师 `{teacher_name}` 存在未使用权重参数，已自动忽略: {unexpected_keys[:5]}')

        incompatible = teacher_model.load_state_dict(compatible_state, strict=False)
        missing_keys = list(getattr(incompatible, 'missing_keys', []))
        if missing_keys:
            print(f'WARNING: KDI 教师 `{teacher_name}` 存在缺失参数，使用兼容加载: {missing_keys[:5]}')

        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False

        with torch.no_grad():
            teacher_model(self._build_dummy_interaction())
            if not hasattr(teacher_model, 'result_embed') or teacher_model.result_embed is None:
                raise RuntimeError(f'KDI 教师 `{teacher_name}` 未生成 result_embed，无法执行蒸馏。')
            teacher_result = teacher_model.result_embed.detach().clone()

        user_emb = teacher_result[:self.num_user]
        item_emb = teacher_result[self.num_user:]
        teacher_model.result_embed = None
        del teacher_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return user_emb, item_emb

    def forward(self, interaction):
        user_nodes = interaction[0].clone()
        pos_item_nodes = interaction[1] + self.num_user
        neg_item_nodes = interaction[2] + self.num_user

        # 1. 基础多模态表示学习 (去除了 edge_weight)
        self.v_rep, self.v_pref = self.v_gcn(self.edge_index, self.v_feat)
        self.t_rep, self.t_pref = self.t_gcn(self.edge_index, self.t_feat)
        self.id_rep, _ = self.id_gcn(self.edge_index, self.id_feat)

        # 2. 拼接多模态表示
        representation = torch.cat((self.v_rep, self.t_rep), dim=1)

        # 3. 仿照 MENTOR 的干净拼接方式聚合用户表示
        w = F.softmax(self.weight_u, dim=1).transpose(1, 2)
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

    def train(self, mode=True):
        if mode:
            self.eval_result_embed = None
        return super(KDI, self).train(mode)

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
        for _ in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, h)
        return h

    def full_sort_predict(self, interaction):
        if not self.training and self.eval_result_embed is not None:
            user_rep = self.eval_result_embed[:self.num_user]
            item_rep = self.eval_result_embed[self.num_user:]
        else:
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

            self.eval_result_embed = torch.cat((user_rep, item_rep), dim=0)

        temp_user_tensor = user_rep[interaction[0], :]
        return torch.matmul(temp_user_tensor, item_rep.t())

    def calculate_loss(self, interaction):
        user = interaction[0]
        pos_item = interaction[1]
        neg_item = interaction[2]

        # 1. 学生模型前向传播
        pos_scores, neg_scores = self.forward(interaction)

        loss_bpr = -torch.mean(F.logsigmoid(pos_scores - neg_scores))

        reg_loss = self.reg_weight * ((self.v_pref[user] ** 2).mean() +
                                      (self.t_pref[user] ** 2).mean() +
                                      (self.weight_u ** 2).mean())

        global_align, local_align = 0.0, 0.0

        if self.use_global_align:
            batch_nodes = torch.cat([user, pos_item + self.num_user, neg_item + self.num_user])
            batch_embed = self.result_embed[batch_nodes]
            batch_id_rep = self.id_rep[batch_nodes]
            r_var, r_mean = torch.var(batch_embed), torch.mean(batch_embed)
            id_var, id_mean = torch.var(batch_id_rep), torch.mean(batch_id_rep)
            global_align = torch.abs(r_var - id_var) + torch.abs(r_mean - id_mean)

        if self.use_local_align:
            v_item = self.v_rep[self.num_user:]
            t_item = self.t_rep[self.num_user:]

            # 使用稀疏矩阵乘法替代 scatter_max 全量计算，大幅加速
            u_v_pool_all = torch.sparse.mm(self.ui_pool_mat, v_item)
            u_t_pool_all = torch.sparse.mm(self.ui_pool_mat, t_item)

            u_id_pref = self.id_gcn.preference[user]
            u_v_pool = u_v_pool_all[user]
            u_t_pool = u_t_pool_all[user]

            local_align = (torch.abs(torch.var(u_id_pref) - torch.var(u_v_pool)) +
                           torch.abs(torch.mean(u_id_pref) - torch.mean(u_v_pool)) +
                           torch.abs(torch.var(u_id_pref) - torch.var(u_t_pool)) +
                           torch.abs(torch.mean(u_id_pref) - torch.mean(u_t_pool)))

        student_base_loss = loss_bpr + reg_loss + self.align_weight * (global_align + local_align)

        # 2. 教师模型极速前向 (查表法内积，避免图卷积开销)
        with torch.no_grad():
            u_t1, p_t1, n_t1 = self.t1_user_emb(user), self.t1_item_emb(pos_item), self.t1_item_emb(neg_item)
            t1_pos = torch.sum(u_t1 * p_t1, dim=1)
            t1_neg = torch.sum(u_t1 * n_t1, dim=1)

            u_t2, p_t2, n_t2 = self.t2_user_emb(user), self.t2_item_emb(pos_item), self.t2_item_emb(neg_item)
            t2_pos = torch.sum(u_t2 * p_t2, dim=1)
            t2_neg = torch.sum(u_t2 * n_t2, dim=1)

        s_diff = pos_scores - neg_scores
        t1_diff = t1_pos - t1_neg
        t2_diff = t2_pos - t2_neg

        s_loss_sample = -F.logsigmoid(s_diff)
        t1_loss_sample = -F.logsigmoid(t1_diff)
        t2_loss_sample = -F.logsigmoid(t2_diff)

        # ==========================================
        # 3. 提取 Sample-wise 状态 (State) 并采样动作
        # ==========================================
        # 通过 Adapter 将学生特征投影到各教师维度空间 (此时保留梯度用于后续更新Adapter)
        s_user_emb = self.result_embed[user]
        s_user_aligned_t1 = self.adapter_t1(s_user_emb)
        s_user_aligned_t2 = self.adapter_t2(s_user_emb)

        with torch.no_grad():
            # dim=1 表示针对当前 batch 里的每条样本单独计算
            cos_sim_t1 = F.cosine_similarity(s_user_aligned_t1, u_t1, dim=1)
            cos_sim_t2 = F.cosine_similarity(s_user_aligned_t2, u_t2, dim=1)

            kl_t1 = F.mse_loss(s_diff, t1_diff, reduction='none')
            kl_t2 = F.mse_loss(s_diff, t2_diff, reduction='none')

            # State 构建: (Batch_Size, 7)，在 torch.no_grad() 内无需再写 .detach()
            state = torch.stack([
                s_loss_sample,
                t1_loss_sample,
                t2_loss_sample,
                kl_t1,
                kl_t2,
                cos_sim_t1,
                cos_sim_t2
            ], dim=1)

        action_dist = self.rl_agent(state)
        action = action_dist.sample()  # 动作采样，切断梯度流向蒸馏网络

        # 必须计算 log_prob 才能让梯度传回 Agent 网络
        log_prob = action_dist.log_prob(action).sum(dim=-1)

        # 动作转化为权重 (Batch_Size, 2)
        weights = F.softmax(action, dim=-1)
        w1, w2 = weights[:, 0], weights[:, 1]

        # ==========================================
        # 4. 基于样本级动作(权重)的加权蒸馏损失
        # ==========================================
        hinge_loss_1_sample = torch.clamp(t1_diff - s_diff, min=0.0)
        hinge_loss_2_sample = torch.clamp(t2_diff - s_diff, min=0.0)
        hinge_loss = self.alpha_hinge * torch.mean(w1 * hinge_loss_1_sample + w2 * hinge_loss_2_sample)

        t1_bar = torch.sigmoid(t1_diff / self.temp)
        t2_bar = torch.sigmoid(t2_diff / self.temp)
        s_bar = torch.sigmoid(s_diff / self.temp)
        eps = 1e-8

        ce_loss_1_sample = -(t1_bar * torch.log(s_bar + eps) + (1 - t1_bar) * torch.log(1 - s_bar + eps))
        ce_loss_2_sample = -(t2_bar * torch.log(s_bar + eps) + (1 - t2_bar) * torch.log(1 - s_bar + eps))
        ce_loss = self.beta_ce * torch.mean(w1 * ce_loss_1_sample + w2 * ce_loss_2_sample)

        total_distill_loss = hinge_loss + ce_loss

        # ==========================================
        # 5. 【修复2 & 修复3】计算 Adapter Loss 与 RL Policy Loss
        # ==========================================
        # 【修复2】Adapter需要独立的Loss来训练自己映射到教师空间，否则输出全是噪声
        adapter_loss = F.mse_loss(s_user_aligned_t1, u_t1.detach()) + \
                       F.mse_loss(s_user_aligned_t2, u_t2.detach())

        # 【修复3】Reward 逻辑修复：使用当前动作分配权重后，样本级别的负蒸馏损失作为 Reward。
        # Agent分配的权重如果能让学生有效降低与教师的差异（Loss小），则给予奖励。
        reward = - (w1 * hinge_loss_1_sample + w2 * hinge_loss_2_sample +
                    w1 * ce_loss_1_sample + w2 * ce_loss_2_sample).detach()

        # 基线化 (Baseline) 减小方差，帮助 RL 稳定收敛
        reward = (reward - reward.mean()) / (reward.std() + 1e-8)

        # Policy Gradient: 优化 Agent
        rl_loss = -torch.mean(log_prob * reward)

        # 最终损失融合了学生基础损失、蒸馏损失、Agent 代理损失以及 Adapter 对齐损失
        final_loss = student_base_loss + total_distill_loss + rl_loss + self.align_weight * adapter_loss

        return final_loss

class TeacherWeightAgent(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=32):
        super(TeacherWeightAgent, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        # 引入可学习的对数标准差，用于构建连续动作空间的 Normal 分布以进行探索
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        # 输出均值 mu
        mu = self.net(state)
        # 限制 std 避免过大导致梯度爆炸
        std = torch.clamp(self.log_std.exp(), min=1e-3, max=1.0).expand_as(mu)
        dist = torch.distributions.Normal(mu, std)
        return dist

class GCN(torch.nn.Module):
    def __init__(self, num_user, num_item, aggr_mode, dim_latent=64, device=None, features=None):
        super(GCN, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.device = device
        self.dim_feat = features.size(1)
        self.dim_latent = dim_latent

        # 【修复1】正确初始化可学习参数，移除 .to(device) 操作以保留其叶子节点属性
        # 模型在外部实例化后调用 model.to(device) 时，Parameter 会自动转移到对应的 GPU
        self.preference = nn.Parameter(
            nn.init.xavier_normal_(torch.empty(num_user, self.dim_latent, dtype=torch.float32))
        )

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

        # 多层演化，去除了 edge_weight
        h1 = self.conv_embed_1(x, edge_index)
        h2 = self.conv_embed_1(h1, edge_index)

        x_hat = x + h1 + h2
        return x_hat, self.preference


class Base_gcn(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr='add', **kwargs):
        super(Base_gcn, self).__init__(aggr=aggr, **kwargs)
        self.aggr = aggr
        self.cached_edge_index = None
        self.cached_norm = None

    def forward(self, x, edge_index):
        if self.cached_edge_index is None or self.cached_norm is None:
            edge_index, _ = remove_self_loops(edge_index)
            row, col = edge_index
            deg = degree(row, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            self.cached_norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            self.cached_edge_index = edge_index
        return self.propagate(self.cached_edge_index, x=x, size=(x.size(0), x.size(0)), norm=self.cached_norm)

    def message(self, x_j, norm):
        if self.aggr == 'add':
            return norm.view(-1, 1) * x_j
        return x_j

    def update(self, aggr_out):
        return aggr_out

