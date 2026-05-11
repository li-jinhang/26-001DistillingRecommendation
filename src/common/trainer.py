import os
import itertools
import json
import hashlib
from copy import deepcopy
import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
import matplotlib.pyplot as plt

from time import time
from logging import getLogger

from utils_package.utils import get_local_time, early_stopping, dict2str
from utils_package.topk_evaluator import TopKEvaluator


class AbstractTrainer(object):
    r"""Trainer Class is used to manage the training and evaluation processes of recommender system models.
    AbstractTrainer is an abstract class in which the fit() and evaluate() method should be implemented according
    to different training and evaluation strategies.
    """

    def __init__(self, config, model):
        self.config = config
        self.model = model

    def fit(self, train_data):
        r"""Train the model based on the train data.

        """
        raise NotImplementedError('Method [next] should be implemented.')

    def evaluate(self, eval_data):
        r"""Evaluate the model based on the eval data.

        """

        raise NotImplementedError('Method [next] should be implemented.')


class Trainer(AbstractTrainer):
    r"""The basic Trainer for basic training and evaluation strategies in recommender systems. This class defines common
    functions for training and evaluation processes of most recommender system models, including fit(), evaluate(),
   and some other features helpful for model training and evaluation.

    Generally speaking, this class can serve most recommender system models, If the training process of the model is to
    simply optimize a single loss without involving any complex training strategies, such as adversarial learning,
    pre-training and so on.

    Initializing the Trainer needs two parameters: `config` and `model`. `config` records the parameters information
    for controlling training and evaluation, such as `learning_rate`, `epochs`, `eval_step` and so on.
    More information can be found in [placeholder]. `model` is the instantiated object of a Model Class.

    """

    def __init__(self, config, model):
        super(Trainer, self).__init__(config, model)

        self.logger = getLogger()
        self.learner = config['learner']
        self.learning_rate = config['learning_rate']
        self.epochs = config['epochs']
        self.eval_step = min(config['eval_step'], self.epochs)
        self.stopping_step = config['stopping_step']
        self.clip_grad_norm = config['clip_grad_norm']
        self.valid_metric = config['valid_metric'].lower()
        self.valid_metric_bigger = config['valid_metric_bigger']
        self.test_batch_size = config['eval_batch_size']
        self.device = config['device']
        self.weight_decay = 0.0
        if config['weight_decay'] is not None:
            wd = config['weight_decay']
            self.weight_decay = eval(wd) if isinstance(wd, str) else wd
        self.disable_weight_decay = bool(config.get('disable_weight_decay', False))
        if self.disable_weight_decay:
            self.weight_decay = 0.0

        self.req_training = config['req_training']
        self.diagnose_training = bool(config.get('diagnose_training', False))
        self.diagnose_batches = max(1, int(config.get('diagnose_batches', 1)))
        self.disable_lr_scheduler = bool(config.get('disable_lr_scheduler', False))
        self.disable_grad_clip = bool(config.get('disable_grad_clip', False))
        self.grad_accum_steps = max(1, int(config.get('grad_accum_steps', 1)))
        self.use_amp = bool(config.get('use_amp', False)) and self.device.type == 'cuda'

        self.start_epoch = 0
        self.cur_step = 0
        self.prev_valid_score = None
        self.first_stagnant_epoch = None
        self.first_stagnant_value = None
        self.latest_probe_delta_norm = None
        self.last_diag_epoch = {}
        self._warned_skip_early_stop = False

        tmp_dd = {}
        for j, k in list(itertools.product(config['metrics'], config['topk'])):
            tmp_dd[f'{j.lower()}@{k}'] = 0.0
        self.best_valid_score = -1
        self.best_valid_result = tmp_dd
        self.best_test_upon_valid = tmp_dd
        self.train_loss_dict = dict()
        self.optimizer = self._build_optimizer()
        if self.disable_weight_decay:
            for group in self.optimizer.param_groups:
                group['weight_decay'] = 0.0
        checkpoint_dir = self.config['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.saved_model_file = os.path.join(
            checkpoint_dir, '{}-{}-{}.pth'.format(self.config['model'], self.config['dataset'], get_local_time())
        )

        self.lr_scheduler = None
        if not self.disable_lr_scheduler:
            lr_scheduler = config['learning_rate_scheduler']
            scheduler_base = float(lr_scheduler[0]) if lr_scheduler else 1.0
            scheduler_span = float(lr_scheduler[1]) if lr_scheduler and len(lr_scheduler) > 1 else 1.0
            if scheduler_base <= 0 or scheduler_span <= 0:
                self.logger.warning('Invalid learning_rate_scheduler=%s, fallback to [1.0, 1.0].', lr_scheduler)
                scheduler_base = 1.0
                scheduler_span = 1.0
            fac = lambda epoch: scheduler_base ** (epoch / scheduler_span)
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)

        self.eval_type = config['eval_type']
        self.evaluator = TopKEvaluator(config)
        self.amp_scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.first_param_name, self.first_param_ref, self.last_param_name, self.last_param_ref = self._select_probe_parameters()
        self.diag_records = []
        self.diag_file_path = None
        self._freeze_snapshot()

        self.item_tensor = None
        self.tot_item_num = None

    def _select_probe_parameters(self):
        named_map = {name: param for name, param in self.model.named_parameters()}
        if hasattr(self.model, 'get_diagnostic_param_names'):
            first_name, last_name = self.model.get_diagnostic_param_names()
            if first_name in named_map and last_name in named_map:
                return first_name, named_map[first_name], last_name, named_map[last_name]
        named_params = [(name, param) for name, param in self.model.named_parameters() if param.requires_grad]
        if len(named_params) == 0:
            named_params = list(self.model.named_parameters())
        if len(named_params) == 0:
            return None, None, None, None
        first_name, first_param = named_params[0]
        last_name, last_param = named_params[-1]
        return first_name, first_param, last_name, last_param

    def _tensor_norm(self, tensor):
        if tensor is None:
            return 0.0
        return float(torch.norm(tensor.detach()).item())

    def _param_grad_norm(self, param):
        if param is None:
            return 0.0
        return self._tensor_norm(param.grad)

    def _freeze_snapshot(self):
        if not self.diagnose_training:
            return
        diag_dir = os.path.join(self.config['checkpoint_dir'], 'diagnostics')
        os.makedirs(diag_dir, exist_ok=True)
        meta_seed = int(self.config.get('seed', 0))
        run_id_src = f"{self.config['model']}_{self.config['dataset']}_{meta_seed}_{self.learning_rate}_{self.epochs}"
        run_id = hashlib.md5(run_id_src.encode('utf-8')).hexdigest()[:8]
        self.diag_file_path = os.path.join(diag_dir, f'diag_{self.config["model"]}_{self.config["dataset"]}_{run_id}.json')
        config_snapshot = deepcopy(self.config.final_config_dict if hasattr(self.config, 'final_config_dict') else dict(self.config))
        for key, value in list(config_snapshot.items()):
            if isinstance(value, torch.device):
                config_snapshot[key] = str(value)
        snapshot = {
            'config': config_snapshot,
            'initial_optimizer_lrs': [float(group['lr']) for group in self.optimizer.param_groups],
            'initial_optimizer_weight_decays': [float(group.get('weight_decay', 0.0)) for group in self.optimizer.param_groups],
            'probe_first_param': self.first_param_name,
            'probe_last_param': self.last_param_name,
            'records': []
        }
        with open(self.diag_file_path, 'w', encoding='utf-8') as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False)
        self.logger.info('[DIAG] Snapshot saved: %s', self.diag_file_path)

    def _append_diag_record(self, record):
        if not self.diagnose_training:
            return
        self.diag_records.append(record)
        if self.diag_file_path is None:
            return
        with open(self.diag_file_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        payload['records'] = self.diag_records
        with open(self.diag_file_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def _batch_tensor_stats(self, interaction):
        stats = []
        if interaction is None:
            return stats
        for idx, tensor in enumerate(interaction):
            if not torch.is_tensor(tensor):
                continue
            det = tensor.detach()
            unique_num = int(torch.unique(det).numel()) if det.numel() > 0 else 0
            all_zero = bool(torch.all(det == 0).item()) if det.numel() > 0 else True
            stats.append({
                'index': idx,
                'shape': list(det.shape),
                'unique_num': unique_num,
                'all_zero': all_zero
            })
        return stats

    def _should_skip_early_stopping(self):
        if hasattr(self.model, 'should_skip_early_stopping'):
            return bool(self.model.should_skip_early_stopping())
        return False

    def _update_stagnation(self, epoch_idx, valid_score):
        if self.prev_valid_score is not None and np.isclose(valid_score, self.prev_valid_score, rtol=0.0, atol=1e-12):
            if self.first_stagnant_epoch is None:
                self.first_stagnant_epoch = int(epoch_idx)
                self.first_stagnant_value = float(valid_score)
        self.prev_valid_score = float(valid_score)

    def _build_optimizer(self):
        r"""Init the Optimizer

        Returns:
            torch.optim: the optimizer
        """
        if self.learner.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def _train_epoch(self, train_data, epoch_idx, loss_func=None):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, It will return a
            tuple which includes the sum of loss in each part.
        """
        if not self.req_training:
            return 0.0, []
        self.model.train()
        self.first_param_name, self.first_param_ref, self.last_param_name, self.last_param_ref = self._select_probe_parameters()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        loss_batches = []
        batch_loss_values = []
        first_batch_users = []
        probe_delta_norm_epoch = None
        self.optimizer.zero_grad(set_to_none=True)
        for batch_idx, interaction in enumerate(train_data):
            if batch_idx == 0 and interaction is not None and len(interaction) > 0 and torch.is_tensor(interaction[0]):
                first_batch_users = interaction[0].detach().cpu().tolist()[:16]
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                losses = loss_func(interaction)
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
                current_loss_value = float(loss.item())
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
                current_loss_value = float(losses.item())
            if self._check_nan(loss):
                self.logger.info('Loss is nan at epoch: {}, batch index: {}. Exiting.'.format(epoch_idx, batch_idx))
                return loss, torch.tensor(0.0)
            grad_first_before = self._param_grad_norm(self.first_param_ref)
            grad_last_before = self._param_grad_norm(self.last_param_ref)
            loss_for_backward = loss / self.grad_accum_steps
            scaler_before = float(self.amp_scaler.get_scale()) if self.use_amp else None
            if self.use_amp:
                self.amp_scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()
            grad_first_after = self._param_grad_norm(self.first_param_ref)
            grad_last_after = self._param_grad_norm(self.last_param_ref)
            do_step = ((batch_idx + 1) % self.grad_accum_steps == 0) or (batch_idx == len(train_data) - 1)
            probe_before = self.first_param_ref.detach().clone() if self.first_param_ref is not None else None
            probe_grad_norm = self._param_grad_norm(self.first_param_ref)
            if do_step:
                if self.use_amp:
                    self.amp_scaler.unscale_(self.optimizer)
                if self.clip_grad_norm and not self.disable_grad_clip:
                    clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
                if self.use_amp:
                    self.amp_scaler.step(self.optimizer)
                    self.amp_scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
            scaler_after = float(self.amp_scaler.get_scale()) if self.use_amp else None
            probe_after = self.first_param_ref.detach().clone() if self.first_param_ref is not None else None
            probe_delta_norm = 0.0
            probe_lr = float(self.optimizer.param_groups[0]['lr']) if len(self.optimizer.param_groups) > 0 else 0.0
            if do_step and probe_before is not None and probe_after is not None:
                probe_delta_norm = float(torch.norm(probe_after - probe_before).item())
                probe_delta_norm_epoch = probe_delta_norm
            loss_batches.append(loss.detach())
            batch_loss_values.append(current_loss_value)
            if self.diagnose_training and batch_idx < self.diagnose_batches:
                record = {
                    'epoch': int(epoch_idx),
                    'batch': int(batch_idx),
                    'loss': current_loss_value,
                    'lr_groups': [float(group['lr']) for group in self.optimizer.param_groups],
                    'grad_norm_first_before': grad_first_before,
                    'grad_norm_first_after': grad_first_after,
                    'grad_norm_last_before': grad_last_before,
                    'grad_norm_last_after': grad_last_after,
                    'probe_param_delta_norm': probe_delta_norm,
                    'probe_param_grad_norm': probe_grad_norm,
                    'probe_lr': probe_lr,
                    'probe_param_name': self.first_param_name,
                    'probe_last_param_name': self.last_param_name,
                    'batch_tensor_stats': self._batch_tensor_stats(interaction)
                }
                if self.use_amp:
                    record['amp_scaler_before'] = scaler_before
                    record['amp_scaler_after'] = scaler_after
                    record['amp_scale_dropped'] = bool(scaler_after is not None and scaler_before is not None and scaler_after < scaler_before)
                self._append_diag_record(record)
        loss_min = float(min(batch_loss_values)) if len(batch_loss_values) > 0 else 0.0
        loss_max = float(max(batch_loss_values)) if len(batch_loss_values) > 0 else 0.0
        loss_span = float(loss_max - loss_min)
        self.last_diag_epoch = {
            'epoch': int(epoch_idx),
            'first_batch_users': first_batch_users,
            'loss_min': loss_min,
            'loss_max': loss_max,
            'loss_span': loss_span,
            'latest_probe_param_delta_norm': probe_delta_norm_epoch
        }
        if self.diagnose_training:
            self._append_diag_record({
                'epoch': int(epoch_idx),
                'loss_min': loss_min,
                'loss_max': loss_max,
                'loss_span': loss_span
            })
        return total_loss, loss_batches

    def _valid_epoch(self, valid_data):
        r"""Valid the model with valid data

        Args:
            valid_data (DataLoader): the valid data

        Returns:
            float: valid score
            dict: valid result
        """
        valid_result = self.evaluate(valid_data)
        valid_score = valid_result[self.valid_metric] if self.valid_metric else valid_result['NDCG@20']
        return valid_score, valid_result

    def _check_nan(self, loss):
        if torch.isnan(loss):
            #raise ValueError('Training loss is nan')
            return True

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        train_loss_output = 'epoch %d training [time: %.2fs, ' % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            train_loss_output = ', '.join('train_loss%d: %.4f' % (idx + 1, loss) for idx, loss in enumerate(losses))
        else:
            train_loss_output += 'train loss: %.4f' % losses
        return train_loss_output + ']'

    def fit(self, train_data, valid_data=None, test_data=None, saved=True, verbose=True):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            test_data (DataLoader, optional): None
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True

        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        for epoch_idx in range(self.start_epoch, self.epochs):
            training_start_time = time()
            self.model.pre_epoch_processing()
            train_loss, _ = self._train_epoch(train_data, epoch_idx)
            if torch.is_tensor(train_loss):
                break
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if self.diagnose_training:
                self._append_diag_record({
                    'epoch': int(epoch_idx),
                    'lr_after_scheduler': [float(group['lr']) for group in self.optimizer.param_groups],
                    'latest_probe_param_delta_norm': self.last_diag_epoch.get('latest_probe_param_delta_norm', None),
                    'first_batch_users': self.last_diag_epoch.get('first_batch_users', [])
                })

            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            post_info = self.model.post_epoch_processing()
            if verbose:
                self.logger.info(train_loss_output)
                if post_info is not None:
                    self.logger.info(post_info)

            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data)
                self._update_stagnation(epoch_idx, valid_score)
                skip_early_stopping = self._should_skip_early_stopping()
                if skip_early_stopping:
                    if not self._warned_skip_early_stop:
                        self.logger.info('[DIAG] Early stopping skipped for current model stage.')
                        self._warned_skip_early_stop = True
                    stop_flag = False
                    update_flag = valid_score > self.best_valid_score if self.valid_metric_bigger else valid_score < self.best_valid_score
                    if update_flag:
                        self.best_valid_score = valid_score
                        self.cur_step = 0
                else:
                    self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                        valid_score, self.best_valid_score, self.cur_step,
                        max_step=self.stopping_step, bigger=self.valid_metric_bigger)
                valid_end_time = time()
                valid_score_output = "epoch %d evaluating [time: %.2fs, valid_score: %f]" % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = 'valid result: \n' + dict2str(valid_result)
                _, test_result = self._valid_epoch(test_data)
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                    self.logger.info('test result: \n' + dict2str(test_result))
                if self.diagnose_training:
                    self._append_diag_record({
                        'epoch': int(epoch_idx),
                        'valid_score': float(valid_score),
                        'valid_result': valid_result,
                        'test_result': test_result,
                        'first_stagnant_epoch': self.first_stagnant_epoch,
                        'first_stagnant_value': self.first_stagnant_value
                    })
                if update_flag:
                    update_output = '██ ' + self.config['model'] + '--Best validation results updated!!!'
                    if verbose:
                        self.logger.info(update_output)
                    self.best_valid_result = valid_result
                    self.best_test_upon_valid = test_result
                    if saved:
                        torch.save(self.model.state_dict(), self.saved_model_file)
                        if verbose:
                            self.logger.info('Saving current best: %s', self.saved_model_file)

                if stop_flag:
                    stop_output = '+++++Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break
        if self.diagnose_training:
            self._append_diag_record({
                'summary': {
                    'first_stagnant_epoch': self.first_stagnant_epoch,
                    'first_stagnant_value': self.first_stagnant_value,
                    'best_valid_score': float(self.best_valid_score)
                }
            })
        return self.best_valid_score, self.best_valid_result, self.best_test_upon_valid


    @torch.no_grad()
    def evaluate(self, eval_data, is_test=False, idx=0):
        r"""Evaluate the model based on the eval data.
        Returns:
            dict: eval result, key is the eval metric and value in the corresponding metric value
        """
        self.model.eval()

        # batch full users
        batch_matrix_list = []
        skip_mask = int(self.config.get('debug_overfit_samples', 0) or 0) > 0
        for batch_idx, batched_data in enumerate(eval_data):
            scores = self.model.full_sort_predict(batched_data)
            if not skip_mask:
                masked_items = batched_data[1]
                scores[masked_items[0], masked_items[1]] = -1e10
            _, topk_index = torch.topk(scores, max(self.config['topk']), dim=-1)
            batch_matrix_list.append(topk_index)
        return self.evaluator.evaluate(batch_matrix_list, eval_data, is_test=is_test, idx=idx)

    def plot_train_loss(self, show=True, save_path=None):
        r"""Plot the train loss in each epoch

        Args:
            show (bool, optional): whether to show this figure, default: True
            save_path (str, optional): the data path to save the figure, default: None.
                                       If it's None, it will not be saved.
        """
        epochs = list(self.train_loss_dict.keys())
        epochs.sort()
        values = [float(self.train_loss_dict[epoch]) for epoch in epochs]
        plt.plot(epochs, values)
        plt.xticks(epochs)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if show:
            plt.show()
        if save_path:
            plt.savefig(save_path)
