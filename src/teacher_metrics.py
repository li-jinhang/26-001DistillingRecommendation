import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score

def calculate_disagreement_rate(preds_a, preds_b):
    """
    计算不一致率 (Disagreement Rate)
    :param preds_a: 模型A的预测标签，形状为 (N,) 的 numpy array 或 list
    :param preds_b: 模型B的预测标签，形状为 (N,) 的 numpy array 或 list
    :return: 不一致率 (0~1之间)
    """
    preds_a = np.array(preds_a)
    preds_b = np.array(preds_b)
    
    if preds_a.shape != preds_b.shape:
        raise ValueError("两个模型的预测标签形状必须一致")
    
    disagreement_count = np.sum(preds_a != preds_b)
    disagreement_rate = disagreement_count / len(preds_a)
    return disagreement_rate

def calculate_cohens_kappa(preds_a, preds_b):
    """
    计算 Cohen's Kappa 系数
    用于衡量两个分类器对同一批样本分类结果的一致性。
    1 表示完全一致，0 表示如同随机的一致，负数表示比随机还差。
    
    :param preds_a: 模型A的预测标签，形状为 (N,)
    :param preds_b: 模型B的预测标签，形状为 (N,)
    :return: Kappa 系数
    """
    preds_a = np.array(preds_a)
    preds_b = np.array(preds_b)
    
    if preds_a.shape != preds_b.shape:
        raise ValueError("两个模型的预测标签形状必须一致")
    
    kappa = cohen_kappa_score(preds_a, preds_b)
    return kappa

def calculate_kl_divergence(probs_a, probs_b, reduction='batchmean'):
    """
    计算 KL 散度 (Kullback-Leibler Divergence)
    衡量分布 P(模型A) 和分布 Q(模型B) 之间的差异，计算公式为 KL(P || Q) = sum(P * log(P/Q))
    
    :param probs_a: 模型A的预测概率分布 (P)，形状为 (N, C)，需满足按行求和为1
    :param probs_b: 模型B的预测概率分布 (Q)，形状为 (N, C)，需满足按行求和为1
    :param reduction: 缩减方式，默认为 'batchmean' (即除以 batch size)
    :return: KL 散度值
    """
    # 转换为 tensor
    tensor_a = torch.tensor(probs_a, dtype=torch.float32)
    tensor_b = torch.tensor(probs_b, dtype=torch.float32)
    
    # 避免出现 log(0) 的情况，增加极小值 epsilon
    epsilon = 1e-9
    tensor_a = torch.clamp(tensor_a, min=epsilon, max=1.0)
    tensor_b = torch.clamp(tensor_b, min=epsilon, max=1.0)
    
    # 在 PyTorch 中，F.kl_div(input, target) 相当于计算 KL(target || input)
    # 输入 (input) 应该是对数概率，目标 (target) 应该是标准概率分布
    # 要计算 KL(A || B)，则 A 为 target，B 为 input (需要取 log)
    log_probs_b = torch.log(tensor_b)
    kl_div_a_b = F.kl_div(log_probs_b, tensor_a, reduction=reduction)
    
    return kl_div_a_b.item()

def calculate_symmetric_kl_divergence(probs_a, probs_b, reduction='batchmean'):
    """
    计算对称 KL 散度 (Symmetric KL Divergence / Jeffrey's Divergence)
    Sym_KL(A, B) = (KL(A || B) + KL(B || A)) / 2
    """
    kl_a_b = calculate_kl_divergence(probs_a, probs_b, reduction=reduction)
    kl_b_a = calculate_kl_divergence(probs_b, probs_a, reduction=reduction)
    return (kl_a_b + kl_b_a) / 2.0

if __name__ == "__main__":
    import os
    import sys
    
    # 将当前目录切换到 src，以便 configurator 能够正确找到 configs/ 目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    sys.path.append(script_dir)
    
    from utils_package.configurator import Config
    from utils_package.dataset import RecDataset
    from utils_package.dataloader import EvalDataLoader, TrainDataLoader
    from utils_package.utils import get_model
    
    print("===== 多教师模型 (ALLB架构) 真实模型差异性对比 =====")
    
    # 1. 初始化配置
    config_dict = {
        'gpu_id': 0,
        'epochs': 1, # 不训练，仅为了初始化
    }
    config = Config('ALLB', 'baby', config_dict)
    
    # 2. 加载数据集
    dataset = RecDataset(config)
    train_dataset, valid_dataset, test_dataset = dataset.split()
    
    # 必须调用 str 初始化 dataset.inter_num
    str(train_dataset)
    str(test_dataset)
    
    train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    test_data = EvalDataLoader(config, test_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size'])
    
    # 3. 初始化模型
    device = config['device']
    train_data.pretrain_setup()
    
    model_a = get_model('ALLB')(config, train_data).to(device)
    model_b = get_model('ALLB')(config, train_data).to(device)
    
    # 4. 加载权重
    weight_path_a = os.path.join('saved', 'ALLB-baby-global.pth')
    weight_path_b = os.path.join('saved', 'ALLB-baby-local.pth')
    
    model_a.load_state_dict(torch.load(weight_path_a, map_location=device), strict=False)
    model_b.load_state_dict(torch.load(weight_path_b, map_location=device), strict=False)
    
    model_a.eval()
    model_b.eval()
    
    all_preds_a = []
    all_preds_b = []
    all_probs_a = []
    all_probs_b = []
    
    print("开始计算测试集上的模型输出...")
    with torch.no_grad():
        # 为了初始化 ALLB 的 result_embed，需要先做一次前向传播
        dummy_interaction = (
            torch.zeros(1, dtype=torch.long, device=device),
            torch.zeros(1, dtype=torch.long, device=device),
            torch.zeros(1, dtype=torch.long, device=device)
        )
        model_a(dummy_interaction)
        model_b(dummy_interaction)
        
        for batch_idx, batched_data in enumerate(test_data):
            # 获取打分矩阵 (N_users_in_batch, N_items)
            scores_a = model_a.full_sort_predict(batched_data)
            scores_b = model_b.full_sort_predict(batched_data)
            
            # 转换为概率分布 (Softmax)
            probs_a = F.softmax(scores_a, dim=-1)
            probs_b = F.softmax(scores_b, dim=-1)
            
            # 获取预测的离散标签 (Argmax，即预测得分最高的 item)
            preds_a = torch.argmax(probs_a, dim=-1)
            preds_b = torch.argmax(probs_b, dim=-1)
            
            all_preds_a.extend(preds_a.cpu().numpy())
            all_preds_b.extend(preds_b.cpu().numpy())
            all_probs_a.append(probs_a.cpu().numpy())
            all_probs_b.append(probs_b.cpu().numpy())
            
    # 拼接所有的 numpy array
    probs_a_np = np.concatenate(all_probs_a, axis=0)
    probs_b_np = np.concatenate(all_probs_b, axis=0)
    
    # 5. 计算指标
    dis_rate = calculate_disagreement_rate(all_preds_a, all_preds_b)
    kappa = calculate_cohens_kappa(all_preds_a, all_preds_b)
    kl_div = calculate_kl_divergence(probs_a_np, probs_b_np)
    sym_kl = calculate_symmetric_kl_divergence(probs_a_np, probs_b_np)
    
    print(f"\n===== 对比结果 =====")
    print(f"模型A: {os.path.basename(weight_path_a)}")
    print(f"模型B: {os.path.basename(weight_path_b)}")
    print(f"1. 不一致率 (Disagreement Rate)  : {dis_rate:.4f} ({dis_rate*100:.1f}%)")
    print(f"2. Cohen's Kappa 系数            : {kappa:.4f}")
    print(f"3. KL 散度 (KL(A || B))          : {kl_div:.4f}")
    print(f"   对称 KL 散度 (Jeffrey's)      : {sym_kl:.4f}")
