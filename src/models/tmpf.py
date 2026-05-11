import os
import json
import time
import logging
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.tmpe import BaseModel
from utils_package.utils import get_model


class TeacherFusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )

    def forward(self, student_emb: torch.Tensor, teacher_embs: List[torch.Tensor]) -> torch.Tensor:
        if len(teacher_embs) == 1:
            return teacher_embs[0]
        logits = [self.attn(torch.cat([student_emb, t_emb], dim=1)) for t_emb in teacher_embs]
        weights = F.softmax(torch.cat(logits, dim=1), dim=1)
        fused = torch.zeros_like(teacher_embs[0])
        for idx, t_emb in enumerate(teacher_embs):
            fused = fused + weights[:, idx:idx + 1] * t_emb
        return fused


class DistillationCriterion(nn.Module):
    def __init__(self, temp: float, hard_weight: float, soft_weight: float, emb_weight: float):
        super().__init__()
        self.temp = float(temp)
        self.hard_weight = float(hard_weight)
        self.soft_weight = float(soft_weight)
        self.emb_weight = float(emb_weight)

    def fuse_teacher_prob(self, teacher_probs: List[torch.Tensor]) -> torch.Tensor:
        if len(teacher_probs) == 1:
            return teacher_probs[0]
        stacked = torch.stack(teacher_probs, dim=1)
        entropy = -torch.sum(stacked * torch.log(stacked.clamp_min(1e-12)), dim=2)
        confidence = 1.0 / entropy.clamp_min(1e-8)
        confidence = confidence / confidence.sum(dim=1, keepdim=True).clamp_min(1e-8)
        fused = torch.sum(confidence.unsqueeze(-1) * stacked, dim=1)
        return fused

    def forward(
        self,
        student_pos: torch.Tensor,
        student_neg: torch.Tensor,
        teacher_probs: List[torch.Tensor],
        student_batch_emb: torch.Tensor,
        teacher_batch_embs: List[torch.Tensor],
        fused_teacher_emb: torch.Tensor,
        reg_loss: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        hard_loss = -torch.mean(torch.log2(torch.sigmoid(student_pos - student_neg) + 1e-8)) + reg_loss
        fused_prob = self.fuse_teacher_prob(teacher_probs)
        student_logits = torch.stack([student_pos, student_neg], dim=1) / self.temp
        kd_soft_loss = F.kl_div(F.log_softmax(student_logits, dim=1), fused_prob, reduction='batchmean') * (self.temp ** 2)
        kd_emb_loss = F.mse_loss(student_batch_emb, fused_teacher_emb)
        total = self.hard_weight * hard_loss + self.soft_weight * kd_soft_loss + self.emb_weight * kd_emb_loss
        metric_dict = {
            'total_loss': float(total.detach().item()),
            'hard_loss': float(hard_loss.detach().item()),
            'soft_loss': float(kd_soft_loss.detach().item()),
            'emb_loss': float(kd_emb_loss.detach().item())
        }
        return total, metric_dict


class TeacherEnsemble(nn.Module):
    def __init__(self, config, dataset, device: torch.device, logger: logging.Logger):
        super().__init__()
        self.config = config
        self.dataset = dataset
        self.device = device
        self.logger = logger
        self.strict = bool(config.get('teacher_strict_load', True))
        self.teacher_specs = self._parse_teacher_specs(config)
        self.teachers = nn.ModuleList()
        self.teacher_names: List[str] = []
        self._init_teachers()

    def _parse_teacher_specs(self, config) -> List[Dict[str, str]]:
        specs = config.get('teacher_models', None)
        if specs is not None and isinstance(specs, list) and len(specs) > 0:
            normalized = []
            for item in specs:
                if isinstance(item, dict) and 'name' in item and 'weight_path' in item:
                    normalized.append({'name': str(item['name']), 'weight_path': str(item['weight_path'])})
            if len(normalized) > 0:
                return normalized
        names = config.get('teacher_model_names', None)
        paths = config.get('teacher_weight_paths', None)
        if isinstance(names, list) and isinstance(paths, list) and len(names) == len(paths) and len(names) > 0:
            return [{'name': str(n), 'weight_path': str(p)} for n, p in zip(names, paths)]
        default_specs = [
            {'name': 'TMPA', 'weight_path': str(config.get('teacher_1_save_path', './saved/teacher_1.pth'))},
            {'name': 'TMPB', 'weight_path': str(config.get('teacher_2_save_path', './saved/teacher_2.pth'))}
        ]
        return default_specs

    def _load_state_dict_parallel(self, specs: List[Dict[str, str]]) -> Dict[str, Dict]:
        workers = int(self.config.get('teacher_load_workers', max(1, len(specs))))
        loaded = {}
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            future_map = {}
            for spec in specs:
                abs_path = os.path.abspath(spec['weight_path'])
                future = executor.submit(torch.load, abs_path, map_location='cpu')
                future_map[future] = (spec['name'], abs_path)
            for future in as_completed(future_map):
                name, abs_path = future_map[future]
                state = future.result()
                loaded[abs_path] = {'name': name, 'state_dict': state}
                self.logger.info(f'[TMPF] 教师权重并行加载完成: {name} <- {abs_path}')
        return loaded

    @staticmethod
    def _extract_state_dict(raw_state: Dict) -> Dict[str, torch.Tensor]:
        state = raw_state
        if isinstance(state, dict) and 'state_dict' in state and isinstance(state['state_dict'], dict):
            state = state['state_dict']
        if not isinstance(state, dict):
            raise RuntimeError(f'教师权重格式非法，期望 dict 或包含 state_dict 的 dict，实际为 {type(raw_state)}')
        normalized = {}
        for key, value in state.items():
            normalized_key = key[7:] if isinstance(key, str) and key.startswith('module.') else key
            normalized[normalized_key] = value
        return normalized

    @staticmethod
    def _filter_compatible_state_dict(
        model_state: Dict[str, torch.Tensor],
        loaded_state: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], List[str], List[str]]:
        compatible = {}
        unexpected_keys: List[str] = []
        shape_mismatch_keys: List[str] = []
        for key, value in loaded_state.items():
            if key not in model_state:
                unexpected_keys.append(key)
                continue
            model_tensor = model_state[key]
            if hasattr(value, 'shape') and hasattr(model_tensor, 'shape'):
                if tuple(value.shape) != tuple(model_tensor.shape):
                    shape_mismatch_keys.append(key)
                    continue
            compatible[key] = value
        return compatible, unexpected_keys, shape_mismatch_keys

    def _init_teachers(self):
        loaded_states = self._load_state_dict_parallel(self.teacher_specs)
        for spec in self.teacher_specs:
            t_name = spec['name']
            t_path = os.path.abspath(spec['weight_path'])
            if not os.path.exists(t_path):
                raise FileNotFoundError(f'教师权重文件不存在: {t_path}')
            teacher_cls = get_model(t_name)
            teacher_model = teacher_cls(self.config, self.dataset).to(self.device)
            loaded_state = self._extract_state_dict(loaded_states[t_path]['state_dict'])
            compatible_state, unexpected_keys, shape_mismatch_keys = self._filter_compatible_state_dict(
                teacher_model.state_dict(),
                loaded_state
            )
            if len(unexpected_keys) > 0:
                preview = unexpected_keys[:5]
                self.logger.warning(f'[TMPF] {t_name} 权重包含未使用参数（已自动忽略）: {preview}')
            if len(shape_mismatch_keys) > 0:
                raise RuntimeError(f'[TMPF] {t_name} 权重维度不匹配: {shape_mismatch_keys[:5]}')
            missing_before_load = [k for k in teacher_model.state_dict().keys() if k not in compatible_state]
            optional_missing = [k for k in missing_before_load if k.startswith('result_embed')]
            required_missing = [k for k in missing_before_load if not k.startswith('result_embed')]
            use_strict = self.strict and len(required_missing) == 0 and len(optional_missing) == 0
            if self.strict and len(optional_missing) > 0 and len(required_missing) == 0:
                self.logger.warning(f'[TMPF] {t_name} 缺失运行时参数（已允许兼容加载）: {optional_missing[:5]}')
            incompatible = teacher_model.load_state_dict(compatible_state, strict=use_strict)
            missing_keys = list(getattr(incompatible, 'missing_keys', []))
            if len(required_missing) > 0:
                raise RuntimeError(f'[TMPF] {t_name} 缺失关键参数: {required_missing[:5]}')
            if len(missing_keys) > 0:
                self.logger.warning(f'[TMPF] {t_name} 存在缺失参数（strict=False）: {missing_keys[:5]}')
            teacher_model.eval()
            for param in teacher_model.parameters():
                param.requires_grad = False
            self.teachers.append(teacher_model)
            self.teacher_names.append(t_name)
            self.logger.info(f'[TMPF] 教师模型已冻结: {t_name}')

    def _extract_scores(self, teacher_model: nn.Module, interaction) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = teacher_model(interaction)
        if isinstance(outputs, tuple) and len(outputs) >= 2:
            return outputs[0], outputs[1]
        if isinstance(outputs, dict) and 'pos_scores' in outputs and 'neg_scores' in outputs:
            return outputs['pos_scores'], outputs['neg_scores']
        raise RuntimeError('教师模型前向输出不包含可用的 pos/neg 分数')

    @torch.no_grad()
    def infer(self, interaction, temp: float, batch_nodes: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        workers = int(self.config.get('teacher_extract_workers', max(1, len(self.teachers))))

        def _infer_single(teacher_model: nn.Module):
            pos_scores, neg_scores = self._extract_scores(teacher_model, interaction)
            logits = torch.stack([pos_scores, neg_scores], dim=1) / temp
            probs = F.softmax(logits, dim=1)
            if not hasattr(teacher_model, 'result_embed') or teacher_model.result_embed is None:
                raise RuntimeError('教师模型未生成 result_embed，无法进行特征蒸馏')
            emb = teacher_model.result_embed[batch_nodes].detach()
            return probs, emb

        results = []
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = [executor.submit(_infer_single, teacher_model) for teacher_model in self.teachers]
            for future in as_completed(futures):
                results.append(future.result())
        teacher_probs = [item[0] for item in results]
        teacher_embs = [item[1] for item in results]
        return teacher_probs, teacher_embs


class TMPF(BaseModel):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.temp = float(config.get('distill_temp', config.get('temp', 0.7)))
        self.hard_label_weight = float(config.get('hard_label_weight', 1.0))
        self.soft_label_weight = float(config.get('soft_label_weight', config.get('kd_weight', 0.2)))
        self.embedding_kd_weight = float(config.get('embedding_kd_weight', 1.0))
        self.teacher_ensemble = TeacherEnsemble(config, dataset, self.device, self.logger)
        self.teacher_fusion = TeacherFusion(self.dim_latent * 2).to(self.device)
        self.distill_criterion = DistillationCriterion(
            temp=self.temp,
            hard_weight=self.hard_label_weight,
            soft_weight=self.soft_label_weight,
            emb_weight=self.embedding_kd_weight
        ).to(self.device)
        self.report_path = os.path.abspath(config.get('report_save_path', './saved/tmpf_training_report.json'))
        self.plot_path = os.path.abspath(config.get('plot_save_path', './saved/tmpf_training_curve.png'))
        self.student_save_path = os.path.abspath(config.get('student_save_path', './saved/tmpf_student_final.pth'))
        self.latest_ckpt_path = os.path.abspath(config.get('latest_checkpoint_path', './saved/tmpf_latest.pth'))
        os.makedirs(os.path.dirname(self.report_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.plot_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.student_save_path), exist_ok=True)
        self.epoch_idx = 0
        self.epoch_metrics = {'total_loss': [], 'hard_loss': [], 'soft_loss': [], 'emb_loss': []}
        self.batch_metrics_cache = []
        self.training_start_time = time.time()
        self.logger.info(f'[TMPF] 教师集合: {self.teacher_ensemble.teacher_names}')
        self.logger.info(f'[TMPF] 温度参数: {self.temp}, 硬标签权重: {self.hard_label_weight}, 软标签权重: {self.soft_label_weight}, 特征蒸馏权重: {self.embedding_kd_weight}')

    def get_diagnostic_param_names(self):
        return 'v_gcn.preference', 'teacher_fusion.attn.2.bias'

    def pre_epoch_processing(self):
        self.epoch_idx += 1
        self.batch_metrics_cache = []
        self.logger.info(f'[TMPF] 开始训练 Epoch {self.epoch_idx}')

    def _student_forward(self, interaction):
        v_rep, v_pref = self.v_gcn(self.edge_index, self.v_feat)
        t_rep, t_pref = self.t_gcn(self.edge_index, self.t_feat)
        u_v = v_rep[:self.num_user]
        u_t = t_rep[:self.num_user]
        w = F.softmax(self.weight_u, dim=1).transpose(1, 2)
        user_rep = torch.cat((w[:, :, 0] * u_v, w[:, :, 1] * u_t), dim=1)
        representation = torch.cat((v_rep, t_rep), dim=1)
        item_rep = representation[self.num_user:] + self.buildItemGraph(representation[self.num_user:])
        self.result_embed = torch.cat((user_rep, item_rep), dim=0)
        user, pos, neg = interaction[0], interaction[1] + self.n_users, interaction[2] + self.n_users
        pos_scores = torch.sum(self.result_embed[user] * self.result_embed[pos], dim=1)
        neg_scores = torch.sum(self.result_embed[user] * self.result_embed[neg], dim=1)
        return {
            'result_embed': self.result_embed,
            'v_pref': v_pref,
            't_pref': t_pref,
            'weight_u': self.weight_u,
            'user': user,
            'pos_scores': pos_scores,
            'neg_scores': neg_scores
        }

    def calculate_loss(self, interaction):
        out_s = self._student_forward(interaction)
        user = out_s['user']
        pos_item = interaction[1] + self.n_users
        batch_nodes = torch.unique(torch.cat([user, pos_item]))
        teacher_probs, teacher_embs = self.teacher_ensemble.infer(interaction, self.temp, batch_nodes)
        student_batch_emb = out_s['result_embed'][batch_nodes]
        fused_teacher_emb = self.teacher_fusion(student_batch_emb, teacher_embs)
        reg_loss = self.reg_weight * (
            (out_s['v_pref'][user] ** 2).mean() +
            (out_s['t_pref'][user] ** 2).mean() +
            (out_s['weight_u'] ** 2).mean()
        )
        total_loss, metric_dict = self.distill_criterion(
            student_pos=out_s['pos_scores'],
            student_neg=out_s['neg_scores'],
            teacher_probs=teacher_probs,
            student_batch_emb=student_batch_emb,
            teacher_batch_embs=teacher_embs,
            fused_teacher_emb=fused_teacher_emb,
            reg_loss=reg_loss
        )
        self.batch_metrics_cache.append(metric_dict)
        return total_loss

    def _save_epoch_artifacts(self):
        payload = {
            'model': 'TMPF',
            'epoch': self.epoch_idx,
            'training_seconds': float(time.time() - self.training_start_time),
            'teacher_models': self.teacher_ensemble.teacher_names,
            'temperature': self.temp,
            'hard_label_weight': self.hard_label_weight,
            'soft_label_weight': self.soft_label_weight,
            'embedding_kd_weight': self.embedding_kd_weight,
            'epoch_metrics': self.epoch_metrics
        }
        with open(self.report_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        torch.save(self.state_dict(), self.latest_ckpt_path)
        total_epochs = int(self.config.get('epochs', 0))
        if total_epochs > 0 and self.epoch_idx >= total_epochs:
            torch.save(self.state_dict(), self.student_save_path)
        try:
            import matplotlib.pyplot as plt
            x = np.arange(1, len(self.epoch_metrics['total_loss']) + 1)
            plt.figure(figsize=(10, 6))
            plt.plot(x, self.epoch_metrics['total_loss'], label='total_loss')
            plt.plot(x, self.epoch_metrics['hard_loss'], label='hard_loss')
            plt.plot(x, self.epoch_metrics['soft_loss'], label='soft_loss')
            plt.plot(x, self.epoch_metrics['emb_loss'], label='emb_loss')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(self.plot_path)
            plt.close()
        except Exception as exc:
            self.logger.warning(f'[TMPF] 可视化生成失败: {exc}')

    def post_epoch_processing(self):
        if len(self.batch_metrics_cache) == 0:
            return None
        avg_metrics = {}
        for key in self.epoch_metrics.keys():
            avg_metrics[key] = float(np.mean([m[key] for m in self.batch_metrics_cache]))
            self.epoch_metrics[key].append(avg_metrics[key])
        self._save_epoch_artifacts()
        msg = (
            f"[TMPF] Epoch {self.epoch_idx} "
            f"total={avg_metrics['total_loss']:.6f}, "
            f"hard={avg_metrics['hard_loss']:.6f}, "
            f"soft={avg_metrics['soft_loss']:.6f}, "
            f"emb={avg_metrics['emb_loss']:.6f}, "
            f"latest_ckpt={self.latest_ckpt_path}"
        )
        self.logger.info(msg)
        return msg
