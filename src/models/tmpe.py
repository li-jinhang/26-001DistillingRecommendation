# coding: utf-8
# 包含 TMPA (粗粒度对齐教师), TMPB (细粒度对齐教师)
# 以及 TMPE (基于状态机的离线多阶段多教师知识蒸馏学生模型)

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, degree
import logging

from common.abstract_recommender import GeneralRecommender


# =========================================
# 1. 基础模型 (BaseModel)
# =========================================
class BaseModel(GeneralRecommender):
    """
    一个包含通用初始化逻辑和辅助函数的基类，
    用于减少 TMPE, TMPA, 和 TMPB 中的重复代码。
    """
    def __init__(self, config, dataset):
        super(BaseModel, self).__init__(config, dataset)
        
        self.logger = logging.getLogger(self.__class__.__name__)

        self.logger.info(f"[{self.__class__.__name__}] Initialize Model...")

        # --- 通用参数 ---
        self.num_user = self.n_users
        self.num_item = self.n_items
        self.dim_x = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_mm_layers']
        self.knn_k = config['knn_k']
        self.mm_image_weight = config['mm_image_weight']
        self.aggr_mode = 'add'
        self.dropout = config['dropout']
        self.reg_weight = config['reg_weight']
        self.temp = config['temp']
        self.dim_latent = 64
        self.result_embed = None

        # --- 加载图和多模态特征 ---
        self._load_graph_and_features(config, dataset)

        # --- 初始化 GCN 和其他层 ---
        self._init_layers()

    def _load_graph_and_features(self, config, dataset):
        """加载交互图、多模态特征和 KNN 邻接矩阵"""
        dataset_path = os.path.abspath(os.path.join(config['data_path'], config['dataset']))
        
        # 加载用户图字典
        user_graph_dict_path = os.path.join(dataset_path, config['user_graph_dict_file'])
        if os.path.exists(user_graph_dict_path):
            self.user_graph_dict = np.load(user_graph_dict_path, allow_pickle=True).item()

        # 加载多模态特征
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)

        # 加载或构建多模态邻接矩阵
        mm_adj_file = os.path.join(dataset_path, f'mm_adj_{self.knn_k}.pt')
        if os.path.exists(mm_adj_file):
            self.mm_adj = torch.load(mm_adj_file, map_location=self.device)
        else:
            self.mm_adj = self._build_mm_adj()
            torch.save(self.mm_adj, mm_adj_file)

        # 构建交互图
        train_interactions = dataset.inter_matrix(form='coo').astype(np.float32)
        edge_index = self._pack_edge_index(train_interactions)
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)

    def _build_mm_adj(self):
        """构建多模态邻接矩阵"""
        image_adj, text_adj = None, None
        if self.v_feat is not None:
            _, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
        if self.t_feat is not None:
            _, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())

        if image_adj is not None and text_adj is not None:
            return self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
        return image_adj if image_adj is not None else text_adj

    def _init_layers(self):
        """初始化 GCN 和其他模型层"""
        if self.v_feat is not None:
            self.v_gcn = GCN(self.num_user, self.num_item, self.aggr_mode, self.dim_latent, self.device, self.v_feat)
        if self.t_feat is not None:
            self.t_gcn = GCN(self.num_user, self.num_item, self.aggr_mode, self.dim_latent, self.device, self.t_feat)

        self.id_feat = nn.Parameter(
            nn.init.xavier_normal_(torch.tensor(np.random.randn(self.n_items, self.dim_latent), dtype=torch.float32, requires_grad=True), gain=1).to(self.device)
        )
        self.id_gcn = GCN(self.num_user, self.num_item, self.aggr_mode, self.dim_latent, self.device, self.id_feat)

        self.weight_u = nn.Parameter(
            nn.init.xavier_normal_(torch.tensor(np.random.randn(self.num_user, 2, 1), dtype=torch.float32, requires_grad=True))
        )

    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = F.normalize(mm_embeddings, p=2, dim=-1)
        sim = torch.mm(context_norm, context_norm.t())
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        
        adj_size = sim.size()
        indices0 = torch.arange(knn_ind.shape[0], device=self.device).unsqueeze(1).expand(-1, self.knn_k)
        indices = torch.stack((indices0.flatten(), knn_ind.flatten()), 0)
        
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        values = r_inv_sqrt[indices[0]] * r_inv_sqrt[indices[1]]
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def _pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        return np.column_stack((rows, cols))

    def buildItemGraph(self, h):
        for _ in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, h)
        return h

    def full_sort_predict(self, interaction):
        user_tensor = self.result_embed[:self.n_users]
        item_tensor = self.result_embed[self.n_users:]
        temp_user_tensor = user_tensor[interaction[0], :]
        return torch.matmul(temp_user_tensor, item_tensor.t())


# =========================================
# 2. 注意力融合模块
# =========================================
class AttentionFusion(nn.Module):
    def __init__(self, dim):
        super(AttentionFusion, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim * 2, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1)
        )

    def forward(self, stu_emb, teacher_embs):
        weights = [self.attn(torch.cat([stu_emb, t_emb], dim=1)) for t_emb in teacher_embs]
        weights = F.softmax(torch.cat(weights, dim=1), dim=1)
        
        fused_emb = sum(weights[:, i:i+1] * t_emb for i, t_emb in enumerate(teacher_embs))
        return fused_emb


# =========================================
# 3. 离线多教师知识蒸馏模型 (TMPE)
# =========================================
class TMPE(BaseModel):
    def __init__(self, config, dataset):
        super(TMPE, self).__init__(config, dataset)
        self.kd_weight = config['kd_weight']

        # --- 状态机定义 ---
        self.STAGE_1, self.STAGE_2 = 1, 2
        self.cur_stage = self.STAGE_1
        total_epochs = int(config['epochs']) if config.get('epochs') is not None else 100
        default_stage1 = max(1, min(30, max(1, total_epochs // 3)))
        self.stage_epochs = config.get('stage_epochs', [default_stage1, total_epochs])
        if len(self.stage_epochs) < 2:
            self.stage_epochs = [default_stage1, total_epochs]
        self.stage_epochs = [int(self.stage_epochs[0]), int(self.stage_epochs[1])]
        if self.stage_epochs[1] < self.stage_epochs[0]:
            self.stage_epochs[1] = self.stage_epochs[0]
        self.stage_epoch = 0
        self.stage_start_time = time.time()
        self.stage_losses = []
        
        # 记录训练过程用于可视化
        self.history_losses_stage1 = []
        self.history_losses_stage2 = []

        # --- 实例化教师和注意力融合模块 ---
        self.teacher_1 = TMPA(config, dataset)
        self.teacher_2 = TMPB(config, dataset)
        self.attn_fusion = AttentionFusion(self.dim_latent * 2).to(self.device)

        self.logger.info(f"[{self.__class__.__name__}] Teacher-1 (TMPA) and Teacher-2 (TMPB) have been initialized inside TMPE.")

        # --- 保存路径配置 ---
        self.t1_path = config.get('teacher_1_save_path', './saved/teacher_1.pth')
        self.t2_path = config.get('teacher_2_save_path', './saved/teacher_2.pth')
        self.stu_path = config.get('student_save_path', './saved/student_tmpe.pth')
        self.report_path = config.get('report_save_path', './saved/training_report.png')

        os.makedirs(os.path.dirname(self.t1_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.stu_path), exist_ok=True)

        # --- 进入第一阶段 ---
        self._enter_stage(self.STAGE_1)

    def _enter_stage(self, stage):
        self.cur_stage = stage
        self.stage_start_time = time.time()
        self.stage_losses = []
        self.stage_epoch = 0

        # 根据阶段设置模型参数的 requires_grad
        if stage == self.STAGE_1:
            self.logger.info("\n[Stage 1/2] Independently Training Teacher-1 (TMPA) and Teacher-2 (TMPB)")
            for param in self.teacher_1.parameters(): param.requires_grad = True
            for param in self.teacher_2.parameters(): param.requires_grad = True
            for name, param in self.named_parameters():
                if 'teacher' not in name: param.requires_grad = False

        elif stage == self.STAGE_2:
            self.logger.info("\n[Stage 2/2] Training Student (TMPE) with Knowledge Distillation")
            
            # 加载预训练教师权重
            if os.path.exists(self.t1_path):
                self.teacher_1.load_state_dict(torch.load(self.t1_path, map_location=self.device))
                self.logger.info(f"  -> Loaded pre-trained Teacher-1 from {self.t1_path}")
            else:
                self.logger.warning(f"  -> Warning: pre-trained Teacher-1 not found at {self.t1_path}")

            if os.path.exists(self.t2_path):
                self.teacher_2.load_state_dict(torch.load(self.t2_path, map_location=self.device))
                self.logger.info(f"  -> Loaded pre-trained Teacher-2 from {self.t2_path}")
            else:
                self.logger.warning(f"  -> Warning: pre-trained Teacher-2 not found at {self.t2_path}")

            self.logger.info(f"  - Distillation Temp: {self.temp}")
            self.logger.info(f"  - Soft Label (KD) Weight: {self.kd_weight}")
            self.logger.info(f"  - Hard Label (BPR) Weight: 1.0")
            
            for param in self.teacher_1.parameters(): param.requires_grad = False
            for param in self.teacher_2.parameters(): param.requires_grad = False
            for name, param in self.named_parameters():
                if 'teacher' not in name: param.requires_grad = True

    def _exit_stage(self):
        duration = time.time() - self.stage_start_time
        avg_loss = np.mean(self.stage_losses) if self.stage_losses else 0.0
        self.logger.info(f"Stage {self.cur_stage} Finished. Duration: {duration:.2f}s, Avg Loss: {avg_loss:.4f}")
        
        if self.cur_stage == self.STAGE_1:
            # 第一阶段结束，保存教师权重
            try:
                torch.save(self.teacher_1.state_dict(), self.t1_path)
                torch.save(self.teacher_2.state_dict(), self.t2_path)
                self.logger.info(f"  -> Teacher-1 (TMPA) weights saved to {self.t1_path}")
                self.logger.info(f"  -> Teacher-2 (TMPB) weights saved to {self.t2_path}")
            except Exception as e:
                self.logger.error(f"  -> Failed to save teacher weights: {e}")
            
        elif self.cur_stage == self.STAGE_2:
            # 第二阶段结束，保存学生权重并生成报告
            try:
                torch.save(self.state_dict(), self.stu_path)
                self.logger.info(f"  -> Final Student (TMPE) weights saved to {self.stu_path}")
                self._generate_report()
            except Exception as e:
                self.logger.error(f"  -> Failed to save student weights: {e}")

    def _generate_report(self):
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            plt.plot(self.history_losses_stage1, label='Stage 1 (Teachers: TMPA & TMPB)', color='blue')
            x_stage2 = range(len(self.history_losses_stage1), len(self.history_losses_stage1) + len(self.history_losses_stage2))
            plt.plot(x_stage2, self.history_losses_stage2, label='Stage 2 (Student: TMPE KD)', color='orange')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('TMPE Multi-Stage Training Process')
            plt.legend()
            plt.grid(True)
            plt.savefig(self.report_path)
            plt.close()
            self.logger.info(f"  -> Training visualization report saved to {self.report_path}")
        except Exception as e:
            self.logger.warning(f"  -> Warning: Could not generate report: {e}")

    def pre_epoch_processing(self):
        self.stage_epoch += 1
        model_name = "TMPA & TMPB" if self.cur_stage == self.STAGE_1 else "TMPE"
        self.logger.info(f"--- Starting Epoch {self.stage_epoch} for Stage {self.cur_stage} (Model: {model_name}) ---")
        if self.cur_stage == self.STAGE_1 and self.stage_epoch > self.stage_epochs[0]:
            self._exit_stage()
            self._enter_stage(self.STAGE_2)
            
    def post_epoch_processing(self):
        # 记录日志，供 Trainer 打印和保存
        avg_loss = np.mean(self.stage_losses) if self.stage_losses else 0.0
        model_name = "TMPA & TMPB" if self.cur_stage == self.STAGE_1 else "TMPE"
        
        if self.cur_stage == self.STAGE_1:
            self.history_losses_stage1.append(avg_loss)
        else:
            self.history_losses_stage2.append(avg_loss)
            if len(self.stage_epochs) > 1 and self.stage_epoch >= self.stage_epochs[1]:
                self._exit_stage()
        
        self.stage_losses = [] # 清空当前 epoch 损失列表
        log_msg = f"[Stage {self.cur_stage} - {model_name}] Epoch {self.stage_epoch} completed. Stage Avg Loss: {avg_loss:.4f}"
        self.logger.info(log_msg)
        return log_msg

    def should_skip_early_stopping(self):
        return self.cur_stage == self.STAGE_1

    def get_diagnostic_param_names(self):
        if self.cur_stage == self.STAGE_1:
            return 'teacher_1.id_feat', 'teacher_2.id_gcn.MLP_1.bias'
        return 'v_gcn.preference', 'attn_fusion.attn.2.bias'

    def forward_student(self, interaction):
        v_rep, v_pref = self.v_gcn(self.edge_index, self.v_feat)
        t_rep, t_pref = self.t_gcn(self.edge_index, self.t_feat)
        
        u_v, u_t = v_rep[:self.num_user], t_rep[:self.num_user]
        w = F.softmax(self.weight_u, dim=1).transpose(1, 2)
        user_rep = torch.cat((w[:, :, 0] * u_v, w[:, :, 1] * u_t), dim=1)

        representation = torch.cat((v_rep, t_rep), dim=1)
        item_rep = representation[self.num_user:] + self.buildItemGraph(representation[self.num_user:])
        
        self.result_embed = torch.cat((user_rep, item_rep), dim=0)

        out_dict = {'result_embed': self.result_embed, 'v_pref': v_pref, 't_pref': t_pref, 'weight_u': self.weight_u}
        
        if len(interaction) > 2:
            user, pos, neg = interaction[0], interaction[1] + self.n_users, interaction[2] + self.n_users
            out_dict['pos_scores'] = torch.sum(self.result_embed[user] * self.result_embed[pos], dim=1)
            out_dict['neg_scores'] = torch.sum(self.result_embed[user] * self.result_embed[neg], dim=1)
            
        return out_dict

    def calculate_loss(self, interaction):
        # 阶段 1: 独立训练两名教师
        if self.cur_stage == self.STAGE_1:
            loss_1 = self.teacher_1.calculate_loss(interaction)
            loss_2 = self.teacher_2.calculate_loss(interaction)
            total_loss = loss_1 + loss_2
            self.stage_losses.append(total_loss.item())
            
            # 为了框架验证集评估 (evaluate 依赖 self.result_embed) 能够正常运行
            with torch.no_grad():
                self.forward_student(interaction)
            return total_loss

        # --- STAGE 2: 学生蒸馏损失 ---
        user, pos_item = interaction[0], interaction[1] + self.n_users
        batch_nodes = torch.unique(torch.cat([user, pos_item]))

        # 1. 获取教师表示和预测得分 (不计算梯度)
        self.teacher_1.eval()
        self.teacher_2.eval()
        with torch.no_grad():
            t1_pos, t1_neg = self.teacher_1(interaction)
            t2_pos, t2_neg = self.teacher_2(interaction)
            
            # 软标签 (带有温度的 Softmax)
            logits_T1 = torch.stack([t1_pos, t1_neg], dim=1) / self.temp
            logits_T2 = torch.stack([t2_pos, t2_neg], dim=1) / self.temp
            prob_T = (F.softmax(logits_T1, dim=1) + F.softmax(logits_T2, dim=1)) / 2.0

            emb_T1 = self.teacher_1.result_embed[batch_nodes].detach()
            emb_T2 = self.teacher_2.result_embed[batch_nodes].detach()

        # 2. 学生前向传播
        out_S = self.forward_student(interaction)
        emb_S = out_S['result_embed'][batch_nodes]

        # 3. 知识蒸馏损失 (软标签损失 + 特征级损失)
        logits_S = torch.stack([out_S['pos_scores'], out_S['neg_scores']], dim=1) / self.temp
        log_prob_S = F.log_softmax(logits_S, dim=1)
        loss_kd_soft = F.kl_div(log_prob_S, prob_T, reduction='batchmean') * (self.temp ** 2)

        fused_teacher_emb = self.attn_fusion(emb_S, [emb_T1, emb_T2])
        loss_kd_emb = F.mse_loss(emb_S, fused_teacher_emb)

        # 4. 硬标签 BPR 损失和正则化
        loss_bpr = -torch.mean(torch.log2(torch.sigmoid(out_S['pos_scores'] - out_S['neg_scores']) + 1e-8))
        reg_loss = self.reg_weight * ((out_S['v_pref'][user]**2).mean() + (out_S['t_pref'][user]**2).mean() + (out_S['weight_u']**2).mean())
        
        # 加权组合
        total_loss = loss_bpr + reg_loss + self.kd_weight * (loss_kd_soft + loss_kd_emb)
        self.stage_losses.append(total_loss.item())
        
        return total_loss


# =========================================
# 4. 基础 GCN 组件
# =========================================
class GCN(nn.Module):
    def __init__(self, num_user, num_item, aggr_mode, dim_latent, device, features):
        super(GCN, self).__init__()
        self.num_user, self.num_item, self.device = num_user, num_item, device
        self.dim_latent, self.dim_feat = dim_latent, features.size(1)

        self.preference = nn.Parameter(nn.init.xavier_normal_(torch.randn(num_user, self.dim_latent, requires_grad=True)))
        self.MLP = nn.Linear(self.dim_feat, 4 * self.dim_latent)
        self.MLP_1 = nn.Linear(4 * self.dim_latent, self.dim_latent)
        self.conv_embed_1 = BaseGCN(self.dim_latent, self.dim_latent, aggr=aggr_mode)

    def forward(self, edge_index, features):
        temp_features = self.MLP_1(F.leaky_relu(self.MLP(features)))
        x = F.normalize(torch.cat((self.preference, temp_features), dim=0))
        
        h1 = self.conv_embed_1(x, edge_index)
        h2 = self.conv_embed_1(h1, edge_index)
        
        return x + h1 + h2, self.preference


class BaseGCN(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr='add', **kwargs):
        super(BaseGCN, self).__init__(aggr=aggr, **kwargs)

    def forward(self, x, edge_index):
        edge_index, _ = remove_self_loops(edge_index)
        return self.propagate(edge_index, x=x, size=(x.size(0), x.size(0)))

    def message(self, x_j, edge_index, size):
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return norm.view(-1, 1) * x_j


# =========================================
# 5. 教师模型 (TMPA, TMPB)
# =========================================
class TMPA(BaseModel):
    def __init__(self, config, dataset):
        super(TMPA, self).__init__(config, dataset)
        self.align_weight = config['align_weight']
        self.logger.info(f"[{self.__class__.__name__}] Teacher-1 Initialization Completed. Mode: Coarse-grained Alignment.")

    def forward(self, interaction):
        v_rep, self.v_preference = self.v_gcn(self.edge_index, self.v_feat)
        t_rep, self.t_preference = self.t_gcn(self.edge_index, self.t_feat)
        id_rep, _ = self.id_gcn(self.edge_index, self.id_feat)

        # --- 构建用户和物品表示 ---
        u_v, u_t = v_rep[:self.num_user], t_rep[:self.num_user]
        w = F.softmax(self.weight_u, dim=1).transpose(1, 2)
        user_rep = torch.cat((w[:, :, 0] * u_v, w[:, :, 1] * u_t), dim=1)
        
        representation = torch.cat((v_rep, t_rep), dim=1)
        item_rep = representation[self.num_user:] + self.buildItemGraph(representation[self.num_user:])
        self.result_embed = torch.cat((user_rep, item_rep), dim=0)

        # --- 构建用于对齐的多种表示 ---
        self.result_embed_guide = self._build_aligned_embed(id_rep, id_rep)
        self.result_embed_v = self._build_aligned_embed(v_rep, v_rep)
        self.result_embed_t = self._build_aligned_embed(t_rep, t_rep)

        # --- 计算得分 ---
        user, pos, neg = interaction[0], interaction[1] + self.n_users, interaction[2] + self.n_users
        pos_scores = torch.sum(self.result_embed[user] * self.result_embed[pos], dim=1)
        neg_scores = torch.sum(self.result_embed[user] * self.result_embed[neg], dim=1)
        return pos_scores, neg_scores

    def _build_aligned_embed(self, rep1, rep2):
        user_rep = torch.cat((rep1[:self.num_user], rep2[:self.num_user]), dim=1)
        item_rep = torch.cat((rep1[self.num_user:], rep2[self.num_user:]), dim=1)
        item_rep += self.buildItemGraph(item_rep)
        return torch.cat((user_rep, item_rep), dim=0)

    def calculate_loss(self, interaction):
        user = interaction[0]
        pos_scores, neg_scores = self.forward(interaction)

        loss_bpr = -torch.mean(torch.log2(torch.sigmoid(pos_scores - neg_scores) + 1e-8))
        reg_loss = self.reg_weight * ((self.v_preference[user]**2).mean() + (self.t_preference[user]**2).mean() + (self.weight_u**2).mean())

        # --- 高斯分布对齐损失 ---
        embeds = [self.result_embed, self.result_embed_guide, self.result_embed_v, self.result_embed_t]
        means = [torch.mean(e) for e in embeds]
        variances = [torch.var(e) for e in embeds]
        
        align_loss = 0
        for i in range(len(embeds)):
            for j in range(i + 1, len(embeds)):
                align_loss += torch.abs(means[i] - means[j]) + torch.abs(variances[i] - variances[j])

        return loss_bpr + reg_loss + self.align_weight * align_loss.mean()


class TMPB(BaseModel):
    def __init__(self, config, dataset):
        super(TMPB, self).__init__(config, dataset)
        self.align_weight = config['align_weight']
        self.logger.info(f"[{self.__class__.__name__}] Teacher-2 Initialization Completed. Mode: Fine-grained InfoNCE Alignment.")

    def forward(self, interaction):
        v_rep, self.v_preference = self.v_gcn(self.edge_index, self.v_feat)
        t_rep, self.t_preference = self.t_gcn(self.edge_index, self.t_feat)
        id_rep, _ = self.id_gcn(self.edge_index, self.id_feat)

        # --- 构建用户和物品表示 ---
        u_v, u_t = v_rep[:self.num_user], t_rep[:self.num_user]
        w = F.softmax(self.weight_u, dim=1).transpose(1, 2)
        user_rep = torch.cat((w[:, :, 0] * u_v, w[:, :, 1] * u_t), dim=1)
        
        representation = torch.cat((v_rep, t_rep), dim=1)
        item_rep = representation[self.num_user:] + self.buildItemGraph(representation[self.num_user:])
        self.result_embed = torch.cat((user_rep, item_rep), dim=0)

        # --- 构建用于对齐的多种表示 ---
        self.result_embed_guide = self._build_aligned_embed(id_rep, id_rep)
        self.result_embed_v = self._build_aligned_embed(v_rep, v_rep)
        self.result_embed_t = self._build_aligned_embed(t_rep, t_rep)

        # --- 计算得分 ---
        user, pos, neg = interaction[0], interaction[1] + self.n_users, interaction[2] + self.n_users
        pos_scores = torch.sum(self.result_embed[user] * self.result_embed[pos], dim=1)
        neg_scores = torch.sum(self.result_embed[user] * self.result_embed[neg], dim=1)
        return pos_scores, neg_scores

    def _build_aligned_embed(self, rep1, rep2):
        user_rep = torch.cat((rep1[:self.num_user], rep2[:self.num_user]), dim=1)
        item_rep = torch.cat((rep1[self.num_user:], rep2[self.num_user:]), dim=1)
        item_rep += self.buildItemGraph(item_rep)
        return torch.cat((user_rep, item_rep), dim=0)

    def calc_infonce_loss(self, embed1, embed2, nodes):
        z1 = F.normalize(embed1[nodes], p=2, dim=1)
        z2 = F.normalize(embed2[nodes], p=2, dim=1)
        
        pos_sim = torch.sum(z1 * z2, dim=-1)
        sim_matrix = torch.matmul(z1, z2.t())
        
        loss = -torch.log(torch.exp(pos_sim / self.temp) / torch.sum(torch.exp(sim_matrix / self.temp), dim=1))
        return loss.mean()

    def calculate_loss(self, interaction):
        user, pos_item = interaction[0], interaction[1] + self.n_users
        pos_scores, neg_scores = self.forward(interaction)

        loss_bpr = -torch.mean(torch.log2(torch.sigmoid(pos_scores - neg_scores) + 1e-8))
        reg_loss = self.reg_weight * ((self.v_preference[user]**2).mean() + (self.t_preference[user]**2).mean() + (self.weight_u**2).mean())

        # --- InfoNCE 对比损失 ---
        batch_nodes = torch.unique(torch.cat([user, pos_item]))
        align_loss_v_id = self.calc_infonce_loss(self.result_embed_v, self.result_embed_guide, batch_nodes)
        align_loss_t_id = self.calc_infonce_loss(self.result_embed_t, self.result_embed_guide, batch_nodes)
        align_loss_v_t = self.calc_infonce_loss(self.result_embed_v, self.result_embed_t, batch_nodes)
        
        align_loss = (align_loss_v_id + align_loss_t_id + align_loss_v_t) / 3.0
        
        return loss_bpr + reg_loss + self.align_weight * align_loss
