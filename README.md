# dien_pytorch

## 1. 目标

这个目录是基于 https://github.com/mouna99/dien 新建的一套 **PyTorch 版 CTR 复现工程**。

- 复现 `DIN`
- 实现 `Wide&Deep` 作为基线模型
- 引入 DIN 论文里 `RelaImpr` 计算方式


## 2. 已创建文件说明

### `dien_pytorch/__init__.py`

大致内容：

- 包初始化文件

作用：

- 让 `dien_pytorch` 成为一个标准 Python 包目录

### `dien_pytorch/requirements.txt`

大致内容：

- `torch`
- `numpy`

作用：

- 给出最基础的运行依赖
- `torch` 需要你按自己机器的 CUDA 版本安装对应 GPU 版

### `dien_pytorch/utils.py`

大致内容：

- 目录创建
- 随机种子固定
- pickle 加载
- JSON 保存
- AUC 计算
- CTR loss
- accuracy
- `RelaImpr` 计算
- 行数统计与 step 估算

作用：

- 放置训练、评估、对比共用的工具函数
- 其中：
  - `calc_auc()` 对齐原 TensorFlow 项目
  - `calc_rela_impr()` 用于计算 `DIN` 相对 `Wide&Deep` 的提升

### `dien_pytorch/dataset.py`

大致内容：

- `DataIteratorTorch`
- `prepare_batch()`

作用：

- 直接读取根目录下的：
  - `local_train_splitByUser`
  - `local_test_splitByUser`
  - `uid_voc.pkl`
  - `mid_voc.pkl`
  - `cat_voc.pkl`
- 保持和原项目一致的样本结构：
  - 当前 user
  - 当前 item
  - 当前 category
  - 历史 item 序列
  - 历史 category 序列
- 负责：
  - 序列裁剪
  - padding
  - mask 构造
  - target 构造

### `dien_pytorch/model.py`

大致内容：

- `Dice`
- `DinAttention`
- `BaseCTRModel`
- `DIN`
- `WideDeep`
- `build_model()`

作用：

- 复现原 TensorFlow 项目里的两条模型路径：
  - `Model_DIN`
  - `Model_WideDeep`

#### `DIN` 大致结构

- uid embedding
- item embedding = mid embedding + cat embedding
- history embedding
- `DinAttention`
- 拼接：
  - `uid_emb`
  - `item_eb`
  - `item_his_eb_sum`
  - `item_eb * item_his_eb_sum`
  - `att_fea`
- `BatchNorm -> FC(200) -> Dice -> FC(80) -> Dice -> FC(2)`

#### `WideDeep` 大致结构

- Deep 分支：
  - `uid_emb`
  - `item_eb`
  - `item_his_eb_sum`
  - `BatchNorm -> FC(200) -> PReLU -> FC(80) -> PReLU -> FC(2)`
- Wide 分支：
  - `item_eb`
  - `item_his_eb_sum`
  - `item_eb * item_his_eb_sum`
  - `Linear(2)`
- 最后两路 logits 相加

### `dien_pytorch/train_din.py`

大致内容：

- 参数解析
- 数据路径解析
- train loop
- eval loop
- checkpoint 保存
- best model 保存

作用：

- 统一训练和测试入口
- 现在支持：
  - `--model-name DIN`
  - `--model-name WIDE_DEEP`
- 如果存在 `Wide&Deep` 的 best checkpoint，则在 `DIN` 训练和测试时自动计算 `RelaImpr`

### `dien_pytorch/compare_models.py`

大致内容：

- 加载 `DIN` 和 `Wide&Deep` 的 best checkpoint
- 在同一测试集上评估
- 计算 `RelaImpr`
- 输出 JSON 对比结果

作用：

- 统一完成“DIN vs Wide&Deep”的最终对比
- 将 `Wide&Deep` 作为 `DIN` 的基准模型

---


## 3. RelaImpr 说明

这里使用的 `RelaImpr` 是 CTR 论文里常见的 AUC 相对提升写法。结合这次的对比设定：

- measured model：`DIN`
- baseline model：`Wide&Deep`

计算方式为：

```text
RelaImpr = ((AUC(DIN) - 0.5) / (AUC(Wide&Deep) - 0.5) - 1) * 100%
```

这是把 DIN 论文里使用的 `RelaImpr` 思路迁移到当前项目里，并明确把 `Wide&Deep` 作为基准模型。

---

