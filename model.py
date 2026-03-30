import torch
from torch import nn


class Dice(nn.Module):
    def __init__(self, feature_dim, epsilon=1e-9):#epsilon=1e-9 是为了避免除以零的情况，feature_dim 是输入特征的维度
        super(Dice, self).__init__()
        self.alpha = nn.Parameter(torch.zeros(feature_dim))
        self.epsilon = epsilon

    def forward(self, x):
        mean = x.mean(dim=0, keepdim=True)#计算输入 x 在特征维度上的均值，keepdim=True 表示保持维度不变，mean 的形状与输入 x 的特征维度相同
        std = torch.sqrt(((x - mean) ** 2 + self.epsilon).mean(dim=0, keepdim=True))#计算输入 x 在特征维度上的标准差，首先计算 (x - mean) ** 2 来得到每个元素与均值的差的平方，然后加上 epsilon 来避免除以零的情况，最后对结果取平均并开平方得到标准差 std
        x_normed = (x - mean) / (std + self.epsilon)
        x_p = torch.sigmoid(x_normed)#对标准化后的输入 x_normed 应用 sigmoid 函数得到 x_p，x_p 的值在 0 和 1 之间，表示每个特征的激活程度
        return self.alpha * (1.0 - x_p) * x + x_p * x

'''根据当前候选 item，从用户历史中挑出“相关兴趣”的 item，计算它们与候选 item 的注意力权重，并对相关兴趣的 item 进行加权求和，得到一个表示用户当前兴趣的向量'''
class DinAttention(nn.Module):
    def __init__(self, input_dim):
        super(DinAttention, self).__init__()
        self.fc1 = nn.Linear(input_dim * 4, 80)#输入维度是 input_dim * 4，因为在 forward 方法中会将查询向量、历史行为向量、它们的差和它们的积拼接在一起作为输入，输出维度是 80，可以根据需要调整这个值
        self.fc2 = nn.Linear(80, 40)#第二层全连接层，输入维度是 80，输出维度是 40，可以根据需要调整这个值
        self.fc3 = nn.Linear(40, 1)#第三层全连接层，输入维度是 40，输出维度是 1，因为我们需要为每个历史行为计算一个注意力权重，最后通过 softmax 将这些权重归一化
        self.activation = nn.Sigmoid()#在每层全连接层之后使用 sigmoid 激活函数，可以根据需要选择其他激活函数，如 ReLU 或 PReLU

    def forward(self, query, facts, mask):
        #query:当前候选embedding，facts 是用户历史行为embedding，mask：1表示有效历史行为，0表示无效历史行为
        batch_size, seq_len, hidden_dim = facts.size()#facts 的形状是 (batch_size, seq_len, hidden_dim)，其中 batch_size 是批次大小，seq_len 是历史行为序列的长度，hidden_dim 是每个行为的嵌入维度
        #扩展query 
        queries = query.unsqueeze(1).expand(batch_size, seq_len, hidden_dim)#将查询向量 query 的形状从 (batch_size, hidden_dim) 扩展为 (batch_size, seq_len, hidden_dim)，这样就可以与历史行为向量 facts 进行逐元素操作
        #构造交互特征，包括查询向量、历史行为向量、它们的差和它们的积（DIN的核心）
        din_all = torch.cat([queries, facts, queries - facts, queries * facts], dim=-1)

        #MLP计算注意力权重，得到的 scores 的形状是 (batch_size, seq_len, 1)，表示每个历史行为的注意力权重
        scores = self.activation(self.fc1(din_all))
        scores = self.activation(self.fc2(scores))
        scores = self.fc3(scores).transpose(1, 2)#将 scores 的形状从 (batch_size, seq_len, 1) 转置为 (batch_size, 1, seq_len)，这样就可以与 mask 进行逐元素操作，并且在后续的矩阵乘法中正确地应用注意力权重

        '''用户历史行为序列长度不同，需要 padding 到相同长度'''
        mask = mask.unsqueeze(1).bool()#将 mask 的形状从 (batch_size, seq_len) 扩展为 (batch_size, 1, seq_len)，并转换为布尔类型，这样在后续的操作中可以使用这个 mask 来区分有效和无效的历史行为
        paddings = torch.full_like(scores, -2 ** 32 + 1)#创建一个与 scores 形状相同的张量 paddings，填充了一个非常小的值 -2 ** 32 + 1，这个值在后续的 softmax 操作中会被当作无效历史行为的位置，使得这些位置的注意力权重接近于零
        scores = torch.where(mask, scores, paddings)#使用 torch.where 函数将 scores 中对应无效历史行为的位置替换为 paddings 中的值，这样在后续的 softmax 操作中这些位置的注意力权重会被压制到接近于零
        scores = torch.softmax(scores, dim=-1)#对 scores 在最后一个维度上应用 softmax 函数，将注意力权重归一化，使得每个历史行为的注意力权重在 0 和 1 之间，并且所有历史行为的注意力权重之和为 1

        return torch.matmul(scores, facts)#使用 torch.matmul 函数将归一化后的注意力权重 scores 与历史行为向量 facts 进行矩阵乘法，得到加权求和后的用户兴趣表示，输出的形状是 (batch_size, 1, hidden_dim)


class BaseCTRModel(nn.Module):
    def __init__(self, n_uid, n_mid, n_cat, embedding_dim=18):
        super(BaseCTRModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.item_dim = embedding_dim * 2
        #因为在 encode_inputs 方法中会将 item_emb 和 item_his_emb_sum 拼接在一起作为输入，所以 item_dim 是 embedding_dim 的两倍

        self.uid_embedding = nn.Embedding(n_uid, embedding_dim)#embedding 层将用户id映射到embedding向量
        self.mid_embedding = nn.Embedding(n_mid, embedding_dim)
        self.cat_embedding = nn.Embedding(n_cat, embedding_dim)

    def encode_inputs(self, batch):
        uid_emb = self.uid_embedding(batch["uids"])

        mid_emb = self.mid_embedding(batch["mids"])
        cat_emb = self.cat_embedding(batch["cats"])
        item_eb = torch.cat([mid_emb, cat_emb], dim=1)#拼接，当前 item embedding， (batch_size, item_dim)，其中 item_dim 是 embedding_dim 的两倍

        mid_his_emb = self.mid_embedding(batch["mid_his"])#历史行为 item embedding， (batch_size, seq_len, embedding_dim)
        cat_his_emb = self.cat_embedding(batch["cat_his"])#历史行为类别 embedding， (batch_size, seq_len, embedding_dim)
        item_his_eb = torch.cat([mid_his_emb, cat_his_emb], dim=2)#拼接，历史行为 embedding， (batch_size, seq_len, item_dim)，其中 item_dim 是 embedding_dim 的两倍
        item_his_eb_sum = torch.sum(item_his_eb, dim=1)#对历史行为 embedding 在序列长度维度上求和，粗粒度兴趣，形状是 (batch_size, item_dim)，其中 item_dim 是 embedding_dim 的两倍

        return {
            "uid_emb": uid_emb,
            "item_eb": item_eb,
            "item_his_eb": item_his_eb,
            "item_his_eb_sum": item_his_eb_sum,
            "mask": batch["mask"],
        }


class DIN(BaseCTRModel):
    def __init__(self, n_uid, n_mid, n_cat, embedding_dim=18):
        super(DIN, self).__init__(n_uid, n_mid, n_cat, embedding_dim=embedding_dim)
        self.attention = DinAttention(self.item_dim)

        feature_dim = embedding_dim + self.item_dim * 4
        #因为在 forward 方法中会将 uid_emb、item_eb、item_his_eb_sum、item_eb * item_his_eb_sum 和 att_fea 拼接在一起作为输入，其中 uid_emb 的维度是 embedding_dim，item_eb 的维度是 item_dim，item_his_eb_sum 的维度是 item_dim，item_eb * item_his_eb_sum 的维度也是 item_dim，att_fea 的维度也是 item_dim，所以总的输入维度是 embedding_dim + item_dim * 4
        self.bn1 = nn.BatchNorm1d(feature_dim)#批归一化层，对输入特征进行归一化，feature_dim 是输入特征的维度，可以根据需要调整这个值
        self.fc1 = nn.Linear(feature_dim, 200)
        self.dice1 = Dice(200)
        self.fc2 = nn.Linear(200, 80)
        self.dice2 = Dice(80)
        self.fc3 = nn.Linear(80, 2)#输出层，输入维度是 80，输出维度是 2，因为我们需要预测点击和未点击两个类别的概率，可以根据需要调整这个值

    def forward(self, batch):
        encoded = self.encode_inputs(batch)
        attention_output = self.attention(encoded["item_eb"], encoded["item_his_eb"], encoded["mask"])#调用 DinAttention 模块，传入当前候选 item 的 embedding、用户历史行为的 embedding 和历史行为的 mask，得到加权求和后的用户兴趣表示 attention_output，形状是 (batch_size, 1, item_dim)，其中 item_dim 是 embedding_dim 的两倍
        att_fea = torch.sum(attention_output, dim=1)#对 attention_output 在序列长度维度上求和，得到一个表示用户当前兴趣的向量，形状是 (batch_size, item_dim)，其中 item_dim 是 embedding_dim 的两倍

        features = torch.cat(#构建特征
            [
                encoded["uid_emb"],
                encoded["item_eb"],
                encoded["item_his_eb_sum"],
                encoded["item_eb"] * encoded["item_his_eb_sum"],#交互
                att_fea,#注意力加权求和后的用户兴趣表示
            ],
            dim=1,
        )

        x = self.bn1(features)
        x = self.fc1(x)
        x = self.dice1(x)
        x = self.fc2(x)
        x = self.dice2(x)
        logits = self.fc3(x)
        return logits


class WideDeep(BaseCTRModel):
    def __init__(self, n_uid, n_mid, n_cat, embedding_dim=18):
        super(WideDeep, self).__init__(n_uid, n_mid, n_cat, embedding_dim=embedding_dim)

        deep_input_dim = embedding_dim + self.item_dim * 2
        #因为在 forward 方法中会将 uid_emb、item_eb 和 item_his_eb_sum 拼接在一起作为深度部分的输入，其中 uid_emb 的维度是 embedding_dim，item_eb 的维度是 item_dim，item_his_eb_sum 的维度也是 item_dim，所以 deep_input_dim 是 embedding_dim + item_dim * 2
        wide_input_dim = self.item_dim * 3
        #因为在 forward 方法中会将 item_eb、item_his_eb_sum 和 item_eb * item_his_eb_sum 拼接在一起作为宽度部分的输入，其中 item_eb 的维度是 item_dim，item_his_eb_sum 的维度也是 item_dim，item_eb * item_his_eb_sum 的维度也是 item_dim，所以 wide_input_dim 是 item_dim * 3

        self.bn1 = nn.BatchNorm1d(deep_input_dim)
        self.deep_fc1 = nn.Linear(deep_input_dim, 200)
        self.prelu1 = nn.PReLU(num_parameters=200)
        self.deep_fc2 = nn.Linear(200, 80)
        self.prelu2 = nn.PReLU(num_parameters=80)
        self.deep_fc3 = nn.Linear(80, 2)
        self.wide_fc = nn.Linear(wide_input_dim, 2)

    def forward(self, batch):
        encoded = self.encode_inputs(batch)

        '''深度部分，捕捉非线性、高阶交叉'''
        deep_features = torch.cat(
            [
                encoded["uid_emb"],
                encoded["item_eb"],
                encoded["item_his_eb_sum"],
            ],
            dim=1,
        )
        deep_x = self.bn1(deep_features)
        deep_x = self.deep_fc1(deep_x)
        deep_x = self.prelu1(deep_x)
        deep_x = self.deep_fc2(deep_x)
        deep_x = self.prelu2(deep_x)
        deep_logits = self.deep_fc3(deep_x)

        '''宽度部分,捕捉记忆性、低阶交叉'''
        wide_features = torch.cat(
            [
                encoded["item_eb"],
                encoded["item_his_eb_sum"],
                encoded["item_eb"] * encoded["item_his_eb_sum"],
            ],
            dim=1,
        )
        wide_logits = self.wide_fc(wide_features)
        return deep_logits + wide_logits


def build_model(model_name, n_uid, n_mid, n_cat, embedding_dim=18):
    model_name = model_name.upper()
    if model_name == "DIN":
        return DIN(n_uid, n_mid, n_cat, embedding_dim=embedding_dim)
    if model_name in ["WIDE_DEEP", "WIDE&DEEP", "WIDEDEEP"]:
        return WideDeep(n_uid, n_mid, n_cat, embedding_dim=embedding_dim)
    raise ValueError("Unsupported model_name: %s" % model_name)
