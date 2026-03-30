# dien_pytorch

## 1. 目标

这个目录是基于当前 TensorFlow 版项目，新建的一套 **PyTorch 版 CTR 复现工程**。这次的重点不再只是 `DIN`，而是：

- 复现 `DIN`
- 实现 `Wide&Deep` 作为基线模型
- 引入 DIN 论文里常见的 `RelaImpr` 计算方式
- 支持 `Python 3.10`
- 支持 GPU 训练

当前没有实现的部分：

- `DIEN`
- `DIN-V2-*`
- attention-GRU 路径

---

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
- TensorFlow 风格 CTR loss
- TensorFlow 风格 accuracy
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

## 3. 和原项目的映射关系

PyTorch 版当前复现的是原项目里的两条路径：

- `script/train.py` 中 `model_type == 'DIN'`
- `script/model.py` 中 `Model_DIN`
- `script/utils.py` 中 `din_attention()`
- `script/model.py` 中 `Model_WideDeep`

没有复现的部分：

- `DIEN`
- `DIN-V2-*`
- 自定义 attention-GRU
- `rnn.py`

所以这套 `dien_pytorch/` 是 **DIN + Wide&Deep 对比复现目录**。

---

## 4. RelaImpr 说明

这里使用的 `RelaImpr` 是 CTR 论文里常见的 AUC 相对提升写法。结合这次的对比设定：

- measured model：`DIN`
- baseline model：`Wide&Deep`

计算方式为：

```text
RelaImpr = ((AUC(DIN) - 0.5) / (AUC(Wide&Deep) - 0.5) - 1) * 100%
```

这是把 DIN 论文里使用的 `RelaImpr` 思路迁移到当前项目里，并明确把 `Wide&Deep` 作为基准模型。

---

## 5. 建议的目录使用方式

根目录仍然保留原项目的数据文件，例如：

- `local_train_splitByUser`
- `local_test_splitByUser`
- `uid_voc.pkl`
- `mid_voc.pkl`
- `cat_voc.pkl`

PyTorch 版默认直接从项目根目录读取这些文件，不需要你再复制一份。

---

## 6. Python 3.10 + GPU 复现流程

下面按 Windows + Conda + GPU 的方式来写。

### 步骤 1：进入项目根目录

```powershell
cd 'D:\Y7000P\OneDrive\桌面\搜广推\dien'
```

### 步骤 2：创建 Python 3.10 环境

```powershell
conda create -n dien-pytorch python=3.10 -y
conda activate dien-pytorch
```

### 步骤 3：安装 PyTorch GPU 版本

这一步要根据你本机 CUDA 版本安装。最稳妥的方式是去 PyTorch 官方安装页选择命令：

- <https://pytorch.org/get-started/locally/>

例如常见形式会类似：

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 步骤 4：补安装基础依赖

```powershell
pip install numpy
```

### 步骤 5：确认 GPU 可见

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

### 步骤 6：确认根目录数据已经准备好

需要至少存在：

- `local_train_splitByUser`
- `local_test_splitByUser`
- `uid_voc.pkl`
- `mid_voc.pkl`
- `cat_voc.pkl`

### 步骤 7：先训练 Wide&Deep 基线

```powershell
python .\dien_pytorch\train_din.py train --model-name WIDE_DEEP --device cuda --seed 3
```

### 步骤 8：再训练 DIN

```powershell
python .\dien_pytorch\train_din.py train --model-name DIN --device cuda --seed 3
```

如果 `Wide&Deep` 的 best checkpoint 已经存在，那么 `DIN` 的训练日志里会自动额外输出：

- `RelaImpr(WIDE_DEEP)`

### 步骤 9：分别测试两个模型

```powershell
python .\dien_pytorch\train_din.py test --model-name WIDE_DEEP --device cuda --seed 3
python .\dien_pytorch\train_din.py test --model-name DIN --device cuda --seed 3
```

### 步骤 10：输出最终对比结果

```powershell
python .\dien_pytorch\compare_models.py --device cuda --seed 3
```

它会输出：

- `Wide&Deep` 测试集指标
- `DIN` 测试集指标
- `RelaImpr(DIN vs Wide&Deep)`
- JSON 对比结果文件

---

## 7. 输出文件位置

训练过程中会在：

- `dien_pytorch/outputs/`

下面生成：

- 各模型运行目录
- `config.json`
- `train.log`
- `metrics.tsv`
- `checkpoints/`
- `comparisons/`

其中最重要的是：

- `dien_pytorch/outputs/checkpoints/wide_deep_best_seed3.pt`
- `dien_pytorch/outputs/checkpoints/din_best_seed3.pt`

`compare_models.py` 生成的对比结果会放在：

- `dien_pytorch/outputs/comparisons/`

---

## 8. 运行命令汇总

### 训练 Wide&Deep

```powershell
python .\dien_pytorch\train_din.py train --model-name WIDE_DEEP --device cuda --seed 3
```

### 训练 DIN

```powershell
python .\dien_pytorch\train_din.py train --model-name DIN --device cuda --seed 3
```

### 测试 Wide&Deep

```powershell
python .\dien_pytorch\train_din.py test --model-name WIDE_DEEP --device cuda --seed 3
```

### 测试 DIN

```powershell
python .\dien_pytorch\train_din.py test --model-name DIN --device cuda --seed 3
```

### 输出最终对比

```powershell
python .\dien_pytorch\compare_models.py --device cuda --seed 3
```

### 强制 CPU

```powershell
python .\dien_pytorch\train_din.py train --model-name DIN --device cpu
```

---

## 9. 当前实现边界

这套 PyTorch 工程的目标是：

- 尽量按原项目 `DIN` 和 `Wide&Deep` 的输入与结构复现

但它不是“整个 TensorFlow 仓库的全量迁移”。当前版本主要是：

- 包含 `DIN`
- 包含 `Wide&Deep`
- 引入 `RelaImpr`
- 不实现 `DIEN`
- 不复刻 TensorFlow 版所有历史细节

如果你下一步要继续扩展，我建议顺序是：

1. 先对齐 PyTorch 版 `Wide&Deep` 和 `DIN` 的训练曲线
2. 再验证 `RelaImpr` 的稳定性
3. 最后再考虑是否继续迁移 `DIEN`
