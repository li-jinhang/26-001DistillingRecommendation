import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss

class HAM(GeneralRecommender):
    def __init__(self, config, dataset):
        super(HAM, self).__init__(config, dataset)

        # 参数初始化    
        self.num_user = self.n_users
        self.num_item = self.n_items
        self.latent_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.knn_k = config['knn_k']
        self.lambda_cl = config['lambda_cl']  # InfoNCE 权重
        self.temp = config['temp']            # 对比学习温度系数
        self.device = config['device']
        self.v_item_emb = None
        self.t_item_emb = None
        self.v_trs = None
        self.t_trs = None
        self.has_v = self.v_feat is not None
        self.has_t = self.t_feat is not None

        # 1. 物品原始特征提取与变换 (假设预训练特征已由 dataset 加载)
        if self.has_v:
            self.v_item_emb = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.v_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
        if self.has_t:
            self.t_item_emb = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.t_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)

        # 2. 构建物品同构图 (KNN 基于余弦相似度)
        self.mm_adj = self._build_knn_graph()

        # 3. 超图关联矩阵 H (用户交互过的物品集合)
        interaction_matrix = dataset.inter_matrix(form='coo')
        self.H = self._build_hyper_matrix(interaction_matrix)

        # 4. 映射空间 (用于对比学习对齐)
        self.v_hyper_mlp = nn.Sequential(
            nn.Linear(self.feat_embed_dim, self.feat_embed_dim),
            nn.LeakyReLU(),
            nn.Linear(self.feat_embed_dim, self.latent_dim)
        )
        self.t_hyper_mlp = nn.Sequential(
            nn.Linear(self.feat_embed_dim, self.feat_embed_dim),
            nn.LeakyReLU(),
            nn.Linear(self.feat_embed_dim, self.latent_dim)
        )

        # 最终预测层
        self.user_id_embedding = nn.Embedding(self.num_user, self.latent_dim)
        self.item_id_embedding = nn.Embedding(self.num_item, self.latent_dim)
        self._skip_init_embedding_ids = {
            id(module) for module in (self.v_item_emb, self.t_item_emb) if module is not None
        }

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            if id(module) in self._skip_init_embedding_ids:
                return
            nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _build_knn_graph(self):
        """基于物品特征构建 KNN 物品同构图 (参考 MENTOR)"""
        combined_feat = []
        if self.v_feat is not None: combined_feat.append(F.normalize(self.v_feat))
        if self.t_feat is not None: combined_feat.append(F.normalize(self.t_feat))
        
        target_feat = torch.cat(combined_feat, dim=1)
        sim = torch.mm(target_feat, target_feat.t())
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        
        adj_size = sim.size()
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device).unsqueeze(1).expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        
        return self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        values = r_inv_sqrt[indices[0]] * r_inv_sqrt[indices[1]]
        return torch.sparse.FloatTensor(indices, values, adj_size).to(self.device)

    def _build_hyper_matrix(self, inter_mat):
        """构建归一化的超图关联矩阵 H_norm: D_v^{-1/2} * H * D_e^{-1/2}"""
        row = torch.tensor(inter_mat.col, dtype=torch.long) # Items
        col = torch.tensor(inter_mat.row, dtype=torch.long) # Users
        indices = torch.stack([row, col])
        V = torch.ones_like(row).float()
        
        # 构建未归一化的 H
        H = torch.sparse.FloatTensor(indices, V, (self.num_item, self.num_user))
        
        # 计算节点度 D_v (物品) 和超边度 D_e (用户)
        D_v = torch.sparse.sum(H, dim=1).to_dense()
        D_e = torch.sparse.sum(H, dim=0).to_dense()
        
        # 计算 D_v^{-1/2} 和 D_e^{-1/2}
        D_v_inv_sqrt = torch.pow(D_v + 1e-7, -0.5)
        D_e_inv_sqrt = torch.pow(D_e + 1e-7, -0.5)
        
        # 应用归一化: normalized_value = V * D_v^{-1/2}[item] * D_e^{-1/2}[user]
        normalized_values = V * D_v_inv_sqrt[row] * D_e_inv_sqrt[col]
        
        H_norm = torch.sparse.FloatTensor(indices, normalized_values, (self.num_item, self.num_user)).to(self.device)
        return H_norm

    def hyper_conv(self, item_embs):
        """
        超图卷积演化：
        由于 H 已经是 D_v^{-1/2} * H * D_e^{-1/2}，
        执行 H * H^T 即等效于标准的超图卷积 D_v^{-1/2} * H * D_e^{-1} * H^T * D_v^{-1/2}
        """
        # 1. 节点到超边: hyperedge_embs = H^T * item_embs
        hyperedge_embs = torch.sparse.mm(self.H.t(), item_embs)
        
        # 2. 超边到节点: updated_item_embs = H * hyperedge_embs
        updated_item_embs = torch.sparse.mm(self.H, hyperedge_embs)
        
        return hyperedge_embs, updated_item_embs

    def info_nce_loss(self, view1, view2):
        """对比学习损失实现"""
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        logits = torch.matmul(view1, view2.transpose(0, 1)) / self.temp
        pos_logits = (view1 * view2).sum(dim=-1) / self.temp
        return (-pos_logits + torch.logsumexp(logits, dim=1)).mean()

    def _compute_representations(self):
        zeros_user = self.user_id_embedding.weight.new_zeros(self.num_user, self.latent_dim)
        zeros_item = self.item_id_embedding.weight.new_zeros(self.num_item, self.latent_dim)

        if self.has_v:
            v_feat = self.v_trs(self.v_item_emb.weight)
            # 超图关联聚合
            v_hyper_user, v_updated_item = self.hyper_conv(v_feat)
            # KNN物品同构图聚合
            v_knn_item = torch.sparse.mm(self.mm_adj, v_feat)
            
            v_item_final = v_updated_item + v_knn_item
            v_user_final = self.v_hyper_mlp(v_hyper_user)
        else:
            v_item_final = zeros_item
            v_user_final = zeros_user

        if self.has_t:
            t_feat = self.t_trs(self.t_item_emb.weight)
            # 超图关联聚合
            t_hyper_user, t_updated_item = self.hyper_conv(t_feat)
            # KNN物品同构图聚合
            t_knn_item = torch.sparse.mm(self.mm_adj, t_feat)
            
            t_item_final = t_updated_item + t_knn_item
            t_user_final = self.t_hyper_mlp(t_hyper_user)
        else:
            t_item_final = zeros_item
            t_user_final = zeros_user

        self.v_user_final = v_user_final
        self.t_user_final = t_user_final
        
        # L2归一化 ID 嵌入，防止尺度差异导致融合失效
        norm_user_id = F.normalize(self.user_id_embedding.weight, p=2, dim=1)
        norm_item_id = F.normalize(self.item_id_embedding.weight, p=2, dim=1)
        
        # 整合 ID 嵌入和多模态更新后的物品特征 (相加前对多模态特征也进行归一化)
        final_item_rep = norm_item_id.clone()
        final_user_rep = norm_user_id.clone()
        
        if self.has_v:
            final_item_rep = final_item_rep + F.normalize(v_item_final, p=2, dim=1)
            final_user_rep = final_user_rep + F.normalize(v_user_final, p=2, dim=1)
            
        if self.has_t:
            final_item_rep = final_item_rep + F.normalize(t_item_final, p=2, dim=1)
            final_user_rep = final_user_rep + F.normalize(t_user_final, p=2, dim=1)

        return final_user_rep, final_item_rep

    def forward(self, interaction):
        user_indices, pos_item_indices, neg_item_indices = interaction[0], interaction[1], interaction[2]
        final_user_rep, final_item_rep = self._compute_representations()

        # 5. 计算推荐分数
        u_emb = final_user_rep[user_indices]
        pos_i_emb = final_item_rep[pos_item_indices]
        neg_i_emb = final_item_rep[neg_item_indices]

        pos_scores = torch.mul(u_emb, pos_i_emb).sum(dim=1)
        neg_scores = torch.mul(u_emb, neg_i_emb).sum(dim=1)

        return pos_scores, neg_scores

    def calculate_loss(self, interaction):
        # 1. 推荐主任务损失 (BPR Loss)
        pos_scores, neg_scores = self.forward(interaction)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))

        # 2. 多模态超边对比对齐损失 (InfoNCE)
        cl_loss = self.info_nce_loss(self.v_user_final, self.t_user_final) if self.has_v and self.has_t else 0.0

        # 3. 联合优化
        total_loss = bpr_loss + self.lambda_cl * cl_loss
        
        # 加上 L2 正则化
        reg_loss = (self.user_id_embedding.weight**2).mean() + (self.item_id_embedding.weight**2).mean()
        
        return total_loss + 1e-4 * reg_loss

    def full_sort_predict(self, interaction):
        user_indices = interaction[0]
        final_user_rep, final_item_rep = self._compute_representations()
        u_emb = final_user_rep[user_indices]
        return torch.matmul(u_emb, final_item_rep.t())