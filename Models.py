from utils import *
from Dataset import *


# ====================== 5. 定义Wide&Deep模型 ======================
class WideDeep(nn.Module):
    def __init__(self, wide_dim, deep_numeric_dim, embed_config, hidden_dims=[128, 64]):
        super().__init__()
        # Wide分支：捕捉显式规则
        self.wide_linear = nn.Linear(wide_dim, 1)
        
        # Deep分支：捕捉隐式特征（Embedding + 全连接）
        self.embedding_layers = nn.ModuleList()
        self.embed_total_dim = 0
        for vocab_size, embed_dim in embed_config:
            self.embedding_layers.append(nn.Embedding(vocab_size, embed_dim))
            self.embed_total_dim += embed_dim
        
        # 全连接层
        deep_input_dim = deep_numeric_dim + self.embed_total_dim
        self.deep_fc = nn.Sequential()
        prev_dim = deep_input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            self.deep_fc.add_module(f"fc_{i}", nn.Linear(prev_dim, hidden_dim))
            self.deep_fc.add_module(f"relu_{i}", nn.ReLU())
            self.deep_fc.add_module(f"dropout_{i}", nn.Dropout(0.1))  # 防过拟合
            prev_dim = hidden_dim
        self.deep_out = nn.Linear(prev_dim, 1)
        
        # 输出激活层
        self.sigmoid = nn.Sigmoid()

    def forward(self, wide_x, deep_x):
        # Wide分支前向
        wide_out = self.wide_linear(wide_x)
        
        # Deep分支前向
        deep_numeric_x = deep_x[:, :8]  # 前8维是数值特征
        deep_discrete_x = deep_x[:, 8:].long()  # 后8维是离散特征索引
        
        # 离散特征嵌入
        embed_outs = []
        for i, embed_layer in enumerate(self.embedding_layers):
            embed_outs.append(embed_layer(deep_discrete_x[:, i]))
        embed_outs = torch.cat(embed_outs, dim=1)
        
        # 拼接数值特征和嵌入特征
        deep_combined = torch.cat([deep_numeric_x, embed_outs], dim=1)
        deep_hidden = self.deep_fc(deep_combined)
        deep_out = self.deep_out(deep_hidden)
        
        # 融合两个分支
        total_out = wide_out + deep_out
        # pred = self.sigmoid(total_out)
        # return pred.squeeze()
        # 移除sigmoid，直接返回logits
        return total_out.squeeze()  # 不再应用sigmoid