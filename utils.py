import json
import math
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def ensure_dir(path):#确保目录存在，如果目录不存在则创建
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_pickle(path):
    path = Path(path)
    with path.open("rb") as file_obj:
        try:
            return pickle.load(file_obj)
        except UnicodeDecodeError:
            file_obj.seek(0)
            return pickle.load(file_obj, encoding="latin1")


def save_json(path, payload):
    path = Path(path)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def calc_auc(raw_arr):#计算 AUC输入 raw_arr 是一个包含预测概率和真实标签的列表，输出是 AUC 的值
    arr = sorted(raw_arr, key=lambda d: d[0], reverse=True)#将输入的 raw_arr 按照预测概率从大到小排序，排序后的 arr 中每个元素仍然是一个包含预测概率和真实标签的列表，但现在它们按照预测概率的大小顺序排列
    pos, neg = 0.0, 0.0#pos 表示正例的数量，neg 表示负例的数量
    for record in arr:
        if record[1] == 1.0:#
            pos += 1#正样本
        else:
            neg += 1

    if pos == 0 or neg == 0:
        return 0.0

    fp, tp = 0.0, 0.0#fp 表示假正例的数量，tp 表示真正例的数量
    xy_arr = []
    for record in arr:
        if record[1] == 1.0:#
            tp += 1
        else:
            fp += 1
         # 记录当前 (FPR, TPR) 坐标
        xy_arr.append([fp / neg, tp / pos])

    auc = 0.0
    prev_x = 0.0
    prev_y = 0.0
    for x, y in xy_arr:
        if x != prev_x:
            # 梯形面积 = 底 × (上底 + 下底) / 2
            auc += ((x - prev_x) * (y + prev_y) / 2.0)
            prev_x = x
            prev_y = y
    return auc


def tf_style_ctr_loss(logits, class_targets):
    probs = torch.softmax(logits, dim=1) + 1e-8
    targets = F.one_hot(class_targets, num_classes=2).float()
    return -torch.mean(torch.log(probs) * targets)
#计算交叉熵损失，首先对 logits 应用 softmax 函数得到预测概率 probs，然后将 class_targets 转换为 one-hot 编码的 targets，最后得到平均的交叉熵损失值


def tf_style_accuracy(logits, class_targets):
    probs = torch.softmax(logits, dim=1)
    targets = F.one_hot(class_targets, num_classes=2).float()
    return (torch.round(probs) == targets).float().mean().item()
#计算准确率，首先对 logits 应用 softmax 函数得到预测概率 probs，然后将 class_targets 转换为 one-hot 编码的 targets，最后比较预测概率的四舍五入结果与真实标签是否相等，并计算平均值作为准确率


def calc_rela_impr(measured_auc, baseline_auc):#
    if baseline_auc is None or baseline_auc <= 0.5:
        return None
    return (((measured_auc - 0.5) / (baseline_auc - 0.5)) - 1.0) * 100.0


def count_lines(path):#计算数据文件的行数，输入 path 是数据文件的路径，输出是数据文件的行数
    count = 0
    with Path(path).open("r", encoding="utf-8") as file_obj:
        for count, _ in enumerate(file_obj, start=1):
            pass
    return count


def estimate_steps(num_lines, batch_size):#根据数据文件的行数和批大小估计训练步骤的数量，输入 num_lines 是数据文件的总行数，batch_size 是每个批次的样本数量，输出是训练步骤的数量
    return int(math.ceil(float(num_lines) / float(batch_size)))
