#强化学习蒸馏初步实现

# coding: utf-8
import logging
import os
import random
import sys
import types
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# ALLB 依赖 torch_scatter；当前环境缺包时退化到 scatter_reduce 实现，
# 以保证预训练教师模型至少可以被导入、加载并参与推理。
try:
    import torch_scatter  # noqa: F401
except ImportError:
    module = types.ModuleType("torch_scatter")

    def scatter_max(src, index, dim=-1, out=None, dim_size=None, fill_value=None):
        if dim_size is None:
            dim_size = int(index.max()) + 1
        index_expanded = index.unsqueeze(-1).expand_as(src)
        out = torch.full(
            (dim_size, src.size(-1)),
            torch.finfo(src.dtype).min,
            dtype=src.dtype,
            device=src.device,
        )
        out.scatter_reduce_(dim, index_expanded, src, reduce="amax", include_self=True)
        return out, None

    module.scatter_max = scatter_max
    sys.modules["torch_scatter"] = module

from models.tmpe import BaseModel
from utils_package.utils import get_model


class TeacherPolicyNetwork(nn.Module):
    """为每个教师生成样本级蒸馏权重。"""

    def __init__(self, state_dim, hidden_dim, action_std):
        super().__init__()
        self.action_std = float(action_std)
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, state):
        logits = self.net(state)
        probs = F.softmax(logits, dim=-1)
        return probs[:, 1]

    def sample_action(self, state):
        mean_weight = self.forward(state)
        std = torch.full_like(mean_weight, self.action_std)
        dist = Normal(mean_weight, std)
        raw_action = dist.rsample()
        action = raw_action.clamp(1e-4, 1.0 - 1e-4)
        log_prob = dist.log_prob(raw_action)
        return action, log_prob

    def evaluate_action(self, state, action):
        mean_weight = self.forward(state)
        std = torch.full_like(mean_weight, self.action_std)
        dist = Normal(mean_weight, std)
        return dist.log_prob(action)


class TeacherEnsemble(nn.Module):
    """加载并冻结预训练教师模型。"""

    def __init__(self, config, dataset, device, logger):
        super().__init__()
        self.config = config
        self.dataset = dataset
        self.device = device
        self.logger = logger
        self.strict = bool(config.get("teacher_strict_load", False))
        self.teacher_specs = self._parse_teacher_specs()
        self.teachers = nn.ModuleList()
        self.teacher_names = []
        self.cached_result_embs = []
        self.teacher_bank = None
        self._init_teachers()
        self._cache_teacher_embeddings()

    @staticmethod
    def _extract_scalar(value, default=None):
        if value is None:
            return default
        if isinstance(value, list):
            return value[0] if value else default
        return value

    def _parse_teacher_specs(self):
        specs = self.config.get("teacher_models", None)
        if isinstance(specs, list) and len(specs) > 0:
            normalized = []
            for item in specs:
                if isinstance(item, dict) and "name" in item and "weight_path" in item:
                    normalized.append(
                        {"name": str(item["name"]), "weight_path": str(item["weight_path"])}
                    )
            if normalized:
                return normalized

        legacy_specs = []
        for idx in (1, 2):
            model_name = self.config.get(f"teacher{idx}_model", None)
            weight_name = self.config.get(f"teacher{idx}_weight", None)
            if model_name and weight_name:
                legacy_specs.append(
                    {"name": str(model_name), "weight_path": str(weight_name)}
                )
        if legacy_specs:
            return legacy_specs

        raise RuntimeError("KDG 缺少教师模型配置，请在 KDG.yaml 中提供 `teacher_models` 或旧版教师配置。")

    def _resolve_weight_path(self, weight_path):
        if os.path.isabs(weight_path):
            return weight_path

        cwd = os.getcwd()
        candidates = [
            os.path.abspath(weight_path),
            os.path.abspath(os.path.join(cwd, weight_path)),
            os.path.abspath(os.path.join(cwd, "saved", weight_path)),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", weight_path)),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "saved", weight_path)),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return candidates[2]

    @staticmethod
    def _extract_state_dict(raw_state):
        state = raw_state
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise RuntimeError(f"教师权重格式非法: {type(raw_state)}")
        normalized = {}
        for key, value in state.items():
            normalized_key = key[7:] if isinstance(key, str) and key.startswith("module.") else key
            normalized[normalized_key] = value
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

    def _init_teachers(self):
        for spec in self.teacher_specs:
            teacher_name = spec["name"]
            weight_path = self._resolve_weight_path(spec["weight_path"])
            if not os.path.exists(weight_path):
                raise FileNotFoundError(f"KDG 教师权重不存在: {weight_path}")

            teacher_cls = get_model(teacher_name)
            teacher_model = teacher_cls(self.config, self.dataset).to(self.device)
            raw_state = torch.load(weight_path, map_location=self.device)
            loaded_state = self._extract_state_dict(raw_state)
            compatible_state, unexpected_keys, shape_mismatch_keys = self._filter_compatible_state_dict(
                teacher_model.state_dict(), loaded_state
            )

            if shape_mismatch_keys:
                raise RuntimeError(
                    f"KDG 教师 `{teacher_name}` 权重维度不匹配: {shape_mismatch_keys[:5]}"
                )
            if unexpected_keys:
                self.logger.warning(
                    "[KDG] %s 存在未使用权重参数，已自动忽略: %s",
                    teacher_name,
                    unexpected_keys[:5],
                )

            missing_before_load = [
                key for key in teacher_model.state_dict().keys() if key not in compatible_state
            ]
            use_strict = self.strict and len(missing_before_load) == 0
            incompatible = teacher_model.load_state_dict(compatible_state, strict=use_strict)
            missing_keys = list(getattr(incompatible, "missing_keys", []))
            if self.strict and missing_before_load:
                raise RuntimeError(
                    f"KDG 教师 `{teacher_name}` 缺少关键参数: {missing_before_load[:5]}"
                )
            if missing_keys:
                self.logger.warning(
                    "[KDG] %s 存在缺失参数，使用兼容加载: %s",
                    teacher_name,
                    missing_keys[:5],
                )

            teacher_model.eval()
            for param in teacher_model.parameters():
                param.requires_grad = False
            self.teachers.append(teacher_model)
            self.teacher_names.append(teacher_name)
            self.logger.info("[KDG] 教师模型已加载并冻结: %s <- %s", teacher_name, weight_path)

    def _build_dummy_interaction(self):
        # 教师在蒸馏阶段被冻结，先用一个最小合法 batch 触发整图编码缓存。
        user = torch.zeros(1, dtype=torch.long, device=self.device)
        pos_item = torch.zeros(1, dtype=torch.long, device=self.device)
        neg_item = torch.zeros(1, dtype=torch.long, device=self.device)
        return (user, pos_item, neg_item)

    @torch.no_grad()
    def _cache_teacher_embeddings(self):
        dummy_interaction = self._build_dummy_interaction()
        self.cached_result_embs = []

        for teacher_name, teacher in zip(self.teacher_names, self.teachers):
            teacher.eval()
            teacher(dummy_interaction)
            if not hasattr(teacher, "result_embed") or teacher.result_embed is None:
                raise RuntimeError("教师模型未生成 `result_embed`，无法进行特征蒸馏。")
            self.cached_result_embs.append(teacher.result_embed.detach().clone())
            teacher.result_embed = None
            self.logger.info("[KDG] 教师缓存已构建: %s", teacher_name)
        self.teacher_bank = torch.stack(self.cached_result_embs, dim=0)

    @torch.no_grad()
    def infer(self, interaction, temperature, user, pos_item):
        item_offset = pos_item - interaction[1]
        neg_item = interaction[2] + item_offset
        teacher_bank = self.teacher_bank
        user_emb = teacher_bank[:, user, :]
        pos_emb = teacher_bank[:, pos_item, :]
        neg_emb = teacher_bank[:, neg_item, :]
        pos_scores = torch.sum(user_emb * pos_emb, dim=-1)
        neg_scores = torch.sum(user_emb * neg_emb, dim=-1)
        logits = torch.stack([pos_scores, neg_scores], dim=-1) / temperature
        teacher_probs = F.softmax(logits, dim=-1).permute(1, 0, 2).detach()
        teacher_embs = (0.5 * (user_emb + pos_emb)).permute(1, 0, 2).detach()
        teacher_margins = (pos_scores - neg_scores).transpose(0, 1).detach()
        return teacher_probs, teacher_embs, teacher_margins


class KDG(BaseModel):
    """面向训练框架的 KDG 多教师策略蒸馏模型。"""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.embedding_dim = self.dim_latent * 2

        self.temperature = float(self._scalar(config.get("temp", 0.7)))
        self.reg_weight = float(self._scalar(config.get("reg_weight", 1e-4)))
        self.alpha = float(self._scalar(config.get("alpha", 0.8)))
        self.beta = float(self._scalar(config.get("beta", 0.5)))
        self.reward_alpha = float(self._scalar(config.get("reward_alpha", self.alpha)))
        self.reward_beta = float(self._scalar(config.get("reward_beta", self.beta)))
        self.action_std = float(self._scalar(config.get("action_std", 0.08)))
        self.agent_hidden_dim = int(self._scalar(config.get("agent_hidden_dim", 64)))
        self.buffer_size = int(self._scalar(config.get("buffer_size", 64)))
        self.agent_history_batches = int(self._scalar(config.get("agent_history_batches", 4)))
        self.agent_loss_weight = float(self._scalar(config.get("agent_loss_weight", 0.2)))
        self.agent_replay_update_interval = max(
            1, int(self._scalar(config.get("agent_replay_update_interval", 2)))
        )

        self.teacher_ensemble = TeacherEnsemble(config, dataset, self.device, self.logger)
        self.num_teachers = len(self.teacher_ensemble.teachers)

        self.adapter = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        ).to(self.device)

        self.state_dim = self.embedding_dim + 4
        self.agents = nn.ModuleList(
            [
                TeacherPolicyNetwork(self.state_dim, self.agent_hidden_dim, self.action_std)
                for _ in range(self.num_teachers)
            ]
        ).to(self.device)

        self.history = [deque(maxlen=self.buffer_size) for _ in range(self.num_teachers)]
        self.epoch_idx = 0
        self.global_step = 0
        self.batch_metrics_cache = []
        self.logger.info("[KDG] 已接入教师集合: %s", self.teacher_ensemble.teacher_names)

    @staticmethod
    def _scalar(value, default=None):
        if value is None:
            return default
        if isinstance(value, list):
            return value[0] if value else default
        return value

    def train(self, mode=True):
        super().train(mode)
        self.teacher_ensemble.eval()
        return self

    def get_diagnostic_param_names(self):
        return "v_gcn.preference", "agents.0.net.4.bias"

    @staticmethod
    def _minmax_normalize_per_sample(reward_matrix):
        min_reward = reward_matrix.min(dim=1, keepdim=True).values
        max_reward = reward_matrix.max(dim=1, keepdim=True).values
        scale = max_reward - min_reward
        normalized = (reward_matrix - min_reward) / (scale + 1e-8)
        equal_mask = scale < 1e-8
        if equal_mask.any():
            normalized = torch.where(equal_mask, torch.full_like(normalized, 0.5), normalized)
        return normalized

    def pre_epoch_processing(self):
        self.epoch_idx += 1
        self.batch_metrics_cache = []

    def post_epoch_processing(self):
        if not self.batch_metrics_cache:
            return None
        keys = [
            "total_loss",
            "student_loss",
            "agent_loss",
            "bpr_loss",
            "soft_kd_loss",
            "emb_kd_loss",
            "mean_weight",
        ]
        avg_metrics = {
            key: float(np.mean([batch[key] for batch in self.batch_metrics_cache]))
            for key in keys
        }
        msg = (
            f"[KDG] Epoch {self.epoch_idx} "
            f"total={avg_metrics['total_loss']:.6f}, "
            f"student={avg_metrics['student_loss']:.6f}, "
            f"agent={avg_metrics['agent_loss']:.6f}, "
            f"bpr={avg_metrics['bpr_loss']:.6f}, "
            f"soft={avg_metrics['soft_kd_loss']:.6f}, "
            f"emb={avg_metrics['emb_kd_loss']:.6f}, "
            f"mean_w={avg_metrics['mean_weight']:.6f}"
        )
        self.logger.info(msg)
        return msg

    def _student_forward(self, interaction=None):
        v_rep, v_pref = self.v_gcn(self.edge_index, self.v_feat)
        t_rep, t_pref = self.t_gcn(self.edge_index, self.t_feat)

        u_v = v_rep[: self.num_user]
        u_t = t_rep[: self.num_user]
        w = F.softmax(self.weight_u, dim=1).transpose(1, 2)
        user_rep = torch.cat((w[:, :, 0] * u_v, w[:, :, 1] * u_t), dim=1)

        representation = torch.cat((v_rep, t_rep), dim=1)
        item_rep = representation[self.num_user :] + self.buildItemGraph(representation[self.num_user :])
        self.result_embed = torch.cat((user_rep, item_rep), dim=0)

        out = {
            "result_embed": self.result_embed,
            "v_pref": v_pref,
            "t_pref": t_pref,
            "weight_u": self.weight_u,
            "user_rep": user_rep,
            "item_rep": item_rep,
        }
        if interaction is not None and len(interaction) > 2:
            user = interaction[0]
            pos = interaction[1] + self.n_users
            neg = interaction[2] + self.n_users
            out["user"] = user
            out["pos_scores"] = torch.sum(self.result_embed[user] * self.result_embed[pos], dim=1)
            out["neg_scores"] = torch.sum(self.result_embed[user] * self.result_embed[neg], dim=1)
        return out

    def full_sort_predict(self, interaction):
        out = self._student_forward()
        temp_user_tensor = out["user_rep"][interaction[0], :]
        return torch.matmul(temp_user_tensor, out["item_rep"].t())

    def _build_states(self, student_out, teacher_probs, teacher_embs, teacher_margins, user, pos_item):
        student_pair_emb = 0.5 * (
            student_out["result_embed"][user] + student_out["result_embed"][pos_item]
        )
        adapted_student_emb = self.adapter(student_pair_emb)
        student_logits = torch.stack(
            [student_out["pos_scores"], student_out["neg_scores"]], dim=1
        ) / self.temperature
        student_log_prob = F.log_softmax(student_logits, dim=1)

        teacher_rank_loss = -F.logsigmoid(teacher_margins)
        expanded_student_emb = adapted_student_emb.unsqueeze(1)
        cosine = F.cosine_similarity(expanded_student_emb, teacher_embs, dim=-1)
        student_log_prob = student_log_prob.unsqueeze(1)
        soft_kd_matrix = F.kl_div(student_log_prob, teacher_probs, reduction="none").sum(dim=-1)
        emb_kd_matrix = F.mse_loss(
            expanded_student_emb, teacher_embs, reduction="none"
        ).mean(dim=-1)
        states = torch.cat(
            [
                teacher_embs,
                teacher_margins.unsqueeze(-1),
                teacher_rank_loss.unsqueeze(-1),
                cosine.unsqueeze(-1),
                soft_kd_matrix.unsqueeze(-1),
            ],
            dim=-1,
        )
        return states, soft_kd_matrix, emb_kd_matrix

    def _sample_weight_matrix(self, states):
        actions = []
        log_probs = []
        detached_states = states.detach()
        for teacher_idx, agent in enumerate(self.agents):
            action, log_prob = agent.sample_action(detached_states[:, teacher_idx, :])
            actions.append(action)
            log_probs.append(log_prob)
        weight_matrix = torch.stack(actions, dim=1)
        weight_matrix = weight_matrix / weight_matrix.sum(dim=1, keepdim=True).clamp_min(1e-8)
        log_prob_matrix = torch.stack(log_probs, dim=1)
        return weight_matrix, log_prob_matrix

    def _compute_student_loss(self, student_out, soft_kd_matrix, emb_kd_matrix, weight_matrix):
        user = student_out["user"]
        bpr_per_sample = -F.logsigmoid(student_out["pos_scores"] - student_out["neg_scores"])
        reg_loss = self.reg_weight * (
            (student_out["v_pref"][user] ** 2).mean()
            + (student_out["t_pref"][user] ** 2).mean()
            + (student_out["weight_u"] ** 2).mean()
        )

        soft_kd = (weight_matrix.detach() * soft_kd_matrix).sum(dim=1)
        emb_kd = (weight_matrix.detach() * emb_kd_matrix).sum(dim=1)
        student_loss = (
            bpr_per_sample.mean()
            + reg_loss
            + self.alpha * soft_kd.mean()
            + self.beta * emb_kd.mean()
        )
        return student_loss, bpr_per_sample, soft_kd, emb_kd

    def _compute_rewards(self, bpr_per_sample, soft_kd_matrix, emb_kd_matrix):
        reward_matrix = (
            -bpr_per_sample.unsqueeze(1)
            - self.reward_alpha * soft_kd_matrix.detach()
            - self.reward_beta * emb_kd_matrix.detach()
        )
        return self._minmax_normalize_per_sample(reward_matrix)

    def _store_history(self, states, actions, rewards):
        for teacher_idx in range(self.num_teachers):
            self.history[teacher_idx].append(
                {
                    # 回放样本保留在当前设备，避免每步都发生 CPU/GPU 往返搬运。
                    "state": states[:, teacher_idx, :].detach(),
                    "action": actions[:, teacher_idx].detach(),
                    "reward": rewards[:, teacher_idx].detach(),
                }
            )

    def _sample_history(self, teacher_idx):
        history = self.history[teacher_idx]
        if len(history) == 0:
            return None
        sampled = random.sample(history, k=min(self.agent_history_batches, len(history)))
        state = torch.cat([entry["state"] for entry in sampled], dim=0)
        action = torch.cat([entry["action"] for entry in sampled], dim=0)
        reward = torch.cat([entry["reward"] for entry in sampled], dim=0)
        return state, action, reward

    def _compute_agent_loss(self, log_prob_matrix, rewards):
        current_advantage = rewards - rewards.mean(dim=1, keepdim=True)
        current_loss = -(log_prob_matrix * current_advantage.detach()).mean()
        if self.global_step % self.agent_replay_update_interval != 0:
            return current_loss

        replay_losses = []
        for teacher_idx, agent in enumerate(self.agents):
            sampled = self._sample_history(teacher_idx)
            if sampled is None:
                continue
            replay_states, replay_actions, replay_rewards = sampled
            replay_advantage = replay_rewards - replay_rewards.mean()
            replay_log_prob = agent.evaluate_action(replay_states, replay_actions)
            replay_losses.append(-(replay_log_prob * replay_advantage.detach()).mean())

        if replay_losses:
            replay_loss = torch.stack(replay_losses).mean()
            return 0.5 * (current_loss + replay_loss)
        return current_loss

    def calculate_loss(self, interaction):
        self.global_step += 1
        student_out = self._student_forward(interaction)
        user = interaction[0]
        pos_item = interaction[1] + self.n_users

        teacher_probs, teacher_embs, teacher_margins = self.teacher_ensemble.infer(
            interaction, self.temperature, user, pos_item
        )
        states, soft_kd_matrix, emb_kd_matrix = self._build_states(
            student_out, teacher_probs, teacher_embs, teacher_margins, user, pos_item
        )
        weight_matrix, log_prob_matrix = self._sample_weight_matrix(states)

        student_loss, bpr_per_sample, soft_kd, emb_kd = self._compute_student_loss(
            student_out, soft_kd_matrix, emb_kd_matrix, weight_matrix
        )
        rewards = self._compute_rewards(bpr_per_sample, soft_kd_matrix, emb_kd_matrix)
        self._store_history(states, weight_matrix, rewards)
        agent_loss = self._compute_agent_loss(log_prob_matrix, rewards)
        total_loss = student_loss + self.agent_loss_weight * agent_loss

        self.batch_metrics_cache.append(
            {
                "total_loss": float(total_loss.detach().item()),
                "student_loss": float(student_loss.detach().item()),
                "agent_loss": float(agent_loss.detach().item()),
                "bpr_loss": float(bpr_per_sample.mean().detach().item()),
                "soft_kd_loss": float(soft_kd.mean().detach().item()),
                "emb_kd_loss": float(emb_kd.mean().detach().item()),
                "mean_weight": float(weight_matrix.mean().detach().item()),
            }
        )
        return total_loss
