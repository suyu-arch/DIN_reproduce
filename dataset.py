'''把原始文本数据 → 转成模型可用的 batch 张量（含 padding + mask）'''
from pathlib import Path

import numpy as np

from utils import load_pickle


SEP = "\x02"


class DataIteratorTorch(object):

    def __init__(
        self,
        source,
        uid_voc,
        mid_voc,
        cat_voc,
        batch_size=128,
        maxlen=100,
        skip_empty=False,
        sort_by_length=True,
        max_batch_size=20,
        minlen=None,
    ):
        self.source_path = str(source)#将输入的 source 参数转换为字符串类型
        self.uid_voc = load_pickle(uid_voc)
        self.mid_voc = load_pickle(mid_voc)
        self.cat_voc = load_pickle(cat_voc)

        self.batch_size = batch_size
        self.maxlen = maxlen#最大序列长度，用于截断或者补齐历史行为序列
        self.minlen = minlen#最小序列长度，用于过滤掉历史行为序列长度小于 minlen 的样本
        self.skip_empty = skip_empty#如果 skip_empty 为 True，则会过滤掉历史行为序列长度为 0 的样本
        self.sort_by_length = sort_by_length#如果 sort_by_length 为 True，则会根据历史行为序列的长度对样本进行排序，以便在训练过程中更高效地处理不同长度的序列
        self.k = batch_size * max_batch_size#每次从数据源中读取 k 条样本进行排序和批处理

        self.n_uid = len(self.uid_voc)
        self.n_mid = len(self.mid_voc)
        self.n_cat = len(self.cat_voc)

        self.end_of_data = False
        self.source = None
        self.source_buffer = []
        self.reset()

    def get_n(self):
        return self.n_uid, self.n_mid, self.n_cat

    def reset(self):#重置数据迭代器的状态，关闭之前打开的数据源文件（如果存在），重新打开数据源文件，并清空缓冲区，以便从头开始迭代数据
        if self.source is not None:
            self.source.close()
        self.source = Path(self.source_path).open("r", encoding="utf-8")
        self.source_buffer = []

    def __iter__(self):
        return self

    def __next__(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        targets = []
        '''当缓冲区为空时，从数据源文件中读取 k 条样本，并根据历史行为序列的长度进行排序（如果 sort_by_length 为 True）'''
        if len(self.source_buffer) == 0:
            for _ in range(self.k):
                line = self.source.readline()
                if line == "":
                    break
                self.source_buffer.append(line.rstrip("\n").split("\t"))

            '''让相似长度的样本在一个batch → 减少padding浪费'''
            if self.sort_by_length and self.source_buffer:
                his_length = np.array([len(s[4].split(SEP)) if s[4] else 0 for s in self.source_buffer])
                tidx = his_length.argsort()
                self.source_buffer = [self.source_buffer[i] for i in tidx]
            else:
                self.source_buffer.reverse()

        if len(self.source_buffer) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        '''解析单条样本，转换成模型输入格式，并进行过滤和批处理'''
        try:
            while True:
                try:
                    ss = self.source_buffer.pop()#从buffer尾部取（短的优先）
                except IndexError:
                    break

                #原始ID → embedding索引
                uid = self.uid_voc[ss[1]] if ss[1] in self.uid_voc else 0
                mid = self.mid_voc[ss[2]] if ss[2] in self.mid_voc else 0
                cat = self.cat_voc[ss[3]] if ss[3] in self.cat_voc else 0

                mid_list = []
                for fea in ss[4].split(SEP):
                    if not fea:
                        continue
                    mid_list.append(self.mid_voc[fea] if fea in self.mid_voc else 0)

                cat_list = []
                for fea in ss[5].split(SEP):
                    if not fea:
                        continue
                    cat_list.append(self.cat_voc[fea] if fea in self.cat_voc else 0)

                if self.minlen is not None and len(mid_list) <= self.minlen:
                    continue
                if self.skip_empty and (not mid_list):
                    continue

                source.append([uid, mid, cat, mid_list, cat_list])
                targets.append(int(ss[0]))#标签

                #batch满了就停止
                if len(source) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True#如果在读取数据源文件时发生 IOError 异常，则将 end_of_data 标志设置为 True，以便在下一次迭代时重置数据迭代器并停止迭代

        if len(source) == 0:#
            return self.__next__()

        return prepare_batch(source, targets, self.maxlen)


def prepare_batch(source, targets, maxlen):
    lengths_x = [len(s[4]) for s in source]#每个样本的历史行为序列长度列表，s[4] 是历史类别序列，len(s[4]) 就是历史行为序列的长度
    seqs_mid = [inp[3] for inp in source]#每个样本的历史行为序列列表，inp[3] 是历史行为序列，seqs_mid 就是一个包含所有样本历史行为序列的列表
    seqs_cat = [inp[4] for inp in source]#历史类别序列

    '''截断maxlen'''
    if maxlen is not None:
        new_seqs_mid = []
        new_seqs_cat = []
        new_lengths_x = []
        for l_x, inp in zip(lengths_x, source):
            if l_x > maxlen:
                new_seqs_mid.append(inp[3][l_x - maxlen:])#保留最近行为
                new_seqs_cat.append(inp[4][l_x - maxlen:])
                new_lengths_x.append(maxlen)
            else:
                new_seqs_mid.append(inp[3])
                new_seqs_cat.append(inp[4])
                new_lengths_x.append(l_x)
        lengths_x = new_lengths_x
        seqs_mid = new_seqs_mid
        seqs_cat = new_seqs_cat

    n_samples = len(seqs_mid)
    maxlen_x = int(np.max(lengths_x))

    '''初始化padding后的历史行为序列张量和对应的mask张量'''
    mid_his = np.zeros((n_samples, maxlen_x), dtype=np.int64)
    cat_his = np.zeros((n_samples, maxlen_x), dtype=np.int64)
    mid_mask = np.zeros((n_samples, maxlen_x), dtype=np.float32)

    for idx, (seq_mid, seq_cat) in enumerate(zip(seqs_mid, seqs_cat)):
        seq_len = lengths_x[idx]#idx 是样本索引，seq_mid 是该样本的历史行为序列，seq_cat 是该样本的历史类别序列，seq_len 是该样本的历史行为序列长度
        mid_mask[idx, :seq_len] = 1.0#将 mid_mask 中对应历史行为序列长度的部分设置为 1.0，表示这些位置是有效的历史行为，后续模型可以根据这个 mask 来区分有效和无效的历史行为
        mid_his[idx, :seq_len] = seq_mid
        cat_his[idx, :seq_len] = seq_cat

    click_targets = np.array(targets, dtype=np.float32)
    # Match the TensorFlow project convention:
    # click -> class 0, non-click -> class 1
    class_targets = np.array([0 if target == 1 else 1 for target in targets], dtype=np.int64)

    return {
        "uids": np.array([inp[0] for inp in source], dtype=np.int64),
        "mids": np.array([inp[1] for inp in source], dtype=np.int64),
        "cats": np.array([inp[2] for inp in source], dtype=np.int64),
        "mid_his": mid_his,
        "cat_his": cat_his,
        "mask": mid_mask,
        "seq_lens": np.array(lengths_x, dtype=np.int64),
        "class_targets": class_targets,
        "click_targets": click_targets,#1是点击，0是未点击
    }
