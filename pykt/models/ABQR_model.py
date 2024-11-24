import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
# import glo
from examples.glo import set_value, get_value, _init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GraphAttentionLayer(nn.Module):  # 图注意力机制
    def __init__(self, in_feature, out_feature, alpha, dropout):  # 初始化参数
        super(GraphAttentionLayer, self).__init__()

        self.out_feature = out_feature

        self.W = nn.Parameter(torch.empty((in_feature, out_feature)).to(device))  # 参数 W
        self.a = nn.Parameter(torch.empty((2 * out_feature, 1)).to(device))  # a 用于注意力分数
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(p=dropout).to(device)
        self.leakRelu = nn.LeakyReLU(alpha).to(device)  # 非线性激活

    def prepare(self, h):  # prepare 函数计算节点对之间的初始注意力分数
        h = h.to(device)
        h_i = torch.matmul(h, self.a[:self.out_feature, :].to(device))  # [n, out_feature]
        h_j = torch.matmul(h, self.a[self.out_feature:, :].to(device))  # [n, out_feature]
        e = h_i + h_j.T  # [n, n]
        e = self.leakRelu(e)
        return e

    def forward(self, x, adj):
        # x: n x in_feature, 问题特征矩阵
        # adj: n x n, 问题之间是否存在相同的知识点

        adj = get_value('regist_pos_matrix').to_dense().to(device)

        batch_size = 10000
        device = x.device
        num_nodes = x.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1  # 处理大量数据图的方法，直接分批次
        indices = torch.arange(0, num_nodes).to(device)

        h = torch.matmul(x.to(device), self.W.to(device))  # n x out_feature
        dd = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]

            h_i = torch.matmul(h[mask], self.a[:self.out_feature, :].to(device))  # mask x 1
            h_j = torch.matmul(h, self.a[self.out_feature:, :].to(device))  # n x 1
            e = h_i + h_j.T  # mask x n
            e = self.leakRelu(e)

            now_mask = (adj[mask] <= 0).to(device)  # mask x n
            attn = torch.masked_fill(e, now_mask, -1e9).to(device)  # mask n

            attn = torch.softmax(attn, dim=-1).to(device)  # mask n
            res = torch.matmul(attn, h.to(device))
            res = F.elu(res).to(device)  # mask x n

            res = self.dropout(res)

            dd.append(res)
        dd = torch.vstack(dd).to(device)

        return dd


class GCNConv(nn.Module):  # 提取特征
    def __init__(self, in_dim, out_dim, p):
        super(GCNConv, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.w = nn.Parameter(torch.rand((in_dim, out_dim)).to(device))
        nn.init.xavier_uniform_(self.w)

        self.b = nn.Parameter(torch.rand((out_dim)).to(device))
        nn.init.zeros_(self.b)

        self.dropout = nn.Dropout(p=p).to(device)

    def forward(self, x, adj):
        x = self.dropout(x.to(device))
        x = torch.matmul(x, self.w.to(device))
        x = torch.sparse.mm(adj.float().to(device), x)
        x = x + self.b.to(device)
        return x.to(device)


class MLP_Predictor(nn.Module):  # MLP用来预测最后得结果
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP_Predictor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=True).to(device),
            nn.BatchNorm1d(hidden_size).to(device),
            nn.PReLU().to(device),
            nn.Linear(hidden_size, output_size, bias=True).to(device)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def forward(self, x):
        return self.net(x.to(device))

def drop_feature(x, drop_prob):  # 随机将x一些特征置为0，对应生成另一个F
    drop_mask = torch.empty(
        (x.size(1),),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone().to(device)
    x[:, drop_mask] = 0
    return x

def normalize_graph(A):  # 对图的邻接矩阵 A 进行归一化
    eps = 1e-8
    A = A.to_dense().to(device)
    deg_inv_sqrt = (A.sum(dim=-1).clamp(min=0.) + eps).pow(-0.5).to(device)
    if A.size()[0] != A.size()[1]:
        A = deg_inv_sqrt.unsqueeze(-1) * (deg_inv_sqrt.unsqueeze(-1) * A)
    else:
        A = deg_inv_sqrt.unsqueeze(-1) * A * deg_inv_sqrt.unsqueeze(-2)
    return A.to_sparse().to(device)

def drop_adj(edge_index, drop_prob):  # 对邻接矩阵中的边进行随机丢弃
    begin_size = edge_index.size()
    use_edge = edge_index._indices().to(device)
    drop_mask = torch.empty(
        (use_edge.size(1),),
        dtype=torch.float32,
        device=use_edge.device).uniform_(0, 1) >= drop_prob
    y = use_edge.clone().to(device)
    res = y[:, drop_mask]
    values = torch.ones(res.shape[1]).to(device)
    size = begin_size
    graph = torch.sparse.FloatTensor(res, values, size).to(device)
    graph = normalize_graph(graph).to(device)
    return graph

def augment_graph(x, feat_drop, edge, edge_drop):  # 结合节点特征增强和边增强，对图进行整体的增强操作
    drop_x = drop_feature(x, feat_drop)
    drop_edge = drop_adj(edge, edge_drop)
    return drop_x.to(device), drop_edge.to(device)

def drop_adj_gat(edge_index, drop_prob):  # 对邻接矩阵中的边进行随机丢弃。
    begin_size = edge_index.size()
    use_edge = edge_index._indices().to(device)
    drop_mask = torch.empty(
        (use_edge.size(1),),
        dtype=torch.float32,
        device=use_edge.device).uniform_(0, 1) >= drop_prob
    y = use_edge.clone().to(device)
    res = y[:, drop_mask]
    values = torch.ones(res.shape[1]).to(device)
    size = begin_size
    graph = torch.sparse.FloatTensor(res, values, size).to(device)
    return graph

def augment_graph_gat(x, feat_drop, edge, edge_drop):  # 对 GAT 模型进行图增强，结合特征和边的随机丢弃。
    drop_x = drop_feature(x, feat_drop)
    drop_edge = drop_adj_gat(edge, edge_drop)
    return drop_x.to(device), drop_edge.to(device)

class CosineDecayScheduler:  # 余弦衰减的学习率调度器
    def __init__(self, max_val, warmup_steps, total_steps):
        self.max_val = max_val
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def get(self, step):
        if step < self.warmup_steps:
            return self.max_val * step / self.warmup_steps
        elif self.warmup_steps <= step <= self.total_steps:
            return self.max_val * (1 + np.cos((step - self.warmup_steps) * np.pi /
                                              (self.total_steps - self.warmup_steps))) / 2
        else:
            raise ValueError('Step ({}) > total number of steps ({}).'.format(step, self.total_steps))

def loss_fn(x, y):  # 计算两个嵌入（x 和 y）之间的损失
    x = F.normalize(x.to(device), dim=-1, p=2)
    y = F.normalize(y.to(device), dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

def sce_loss(x, y, alpha=3):  # 计算嵌入之间的对比损失
    x = F.normalize(x.to(device), p=2, dim=-1)
    y = F.normalize(y.to(device), p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    loss = loss.mean().to(device)
    return loss


class BGRL(nn.Module):  # 实现对抗学习的损失计算和在线编码器的编码结果
    def __init__(self, d, p, drop_feat1, drop_feat2, drop_edge1, drop_edge2):
        super(BGRL, self).__init__()

        self.drop_feat1, self.drop_feat2, self.drop_edge1, self.drop_edge2 = drop_feat1, drop_feat2, drop_edge1, drop_edge2

        self.online_encoder = GCNConv(d, d, p).to(device)  # 在线编码器

        self.decoder = GCNConv(d, d, p).to(device)

        self.predictor = MLP_Predictor(d, d, d).to(device)

        self.target_encoder = copy.deepcopy(self.online_encoder).to(device)

        self.fc1 = nn.Linear(d, d).to(device)
        self.fc2 = nn.Linear(d, d).to(device)

        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.enc_mask_token = nn.Parameter(torch.zeros(1, d).to(device))
        self.encoder_to_decoder = nn.Linear(d, d, bias=False).to(device)

    def encoding_mask_noise(self, x, mask_rate=0.3):  # 生成带有掩码噪声的节点特征
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=device)
        num_mask_nodes = int(mask_rate * num_nodes)

        mask_nodes = perm[:num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        num_noise_nodes = int(0.1 * num_mask_nodes)
        perm_mask = torch.randperm(num_mask_nodes, device=device)
        token_nodes = mask_nodes[perm_mask[: int(0.9 * num_mask_nodes)]]
        noise_nodes = mask_nodes[perm_mask[-int(0.1 * num_mask_nodes):]]
        noise_to_be_chosen = torch.randperm(num_nodes, device=device)[:num_noise_nodes]

        out_x = x.clone().to(device)
        out_x[token_nodes] = 0.0
        out_x[noise_nodes] = x[noise_to_be_chosen]

        out_x[token_nodes] += self.enc_mask_token

        return out_x.to(device), mask_nodes.to(device), keep_nodes.to(device)

    def trainable_parameters(self):
        r"""Returns the parameters that will be updated via an optimizer."""
        return list(self.online_encoder.parameters()) + list(self.predictor.parameters())

    @torch.no_grad()
    def update_target_network(self, mm):
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm)

    def project(self, z):  # 将输入嵌入通过两个全连接层进行处理
        z = F.elu(self.fc1(z).to(device))
        return self.fc2(z).to(device)

    def compute_batch_loss(self, x, y):  # 批量损失计算
        z1 = self.project(x).to(device)
        z2 = self.project(y).to(device)

        c1 = F.normalize(z1, dim=-1, p=2).to(device)
        c2 = F.normalize(z2, dim=-1, p=2).to(device)

        batch_size = 15000
        num_nodes = x.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            batch_pos_matrix = glo.get_value('regist_pos_matrix')[mask].to(device)  # batch n
            item_loss = torch.matmul(c1[mask], c2.T).to(device)  # batch n
            item_loss = 2 - 2 * item_loss  # batch n
            need_loss = item_loss * batch_pos_matrix  # batch n

            need_sum = need_loss.sum(dim=-1, keepdims=True).to(device)  # batch 1
            need_mean = need_sum

            losses.append(need_mean)

        return -torch.cat(losses).mean()

    def getGraphMAE_loss(self, x, adj):  # 计算图的重建损失
        mask_rate = 0.3
        use_x, mask_nodes, keep_nodes = self.encoding_mask_noise(x, mask_rate)

        enc_rep = self.online_encoder(use_x, adj).to(device)

        rep = self.encoder_to_decoder(enc_rep).to(device)

        rep[mask_nodes] = 0

        recon = self.decoder(rep, adj).to(device)

        x_init = x[mask_nodes].to(device)
        x_rec = recon[mask_nodes].to(device)
        loss = sce_loss(x_rec, x_init, 3).to(device)
        return enc_rep.to(device), loss

    def forward(self, x, adj, perb=None):
        if perb is None:  # 这里他传的是Q_Q矩阵
            return (x + self.online_encoder(x, adj)).to(device), torch.tensor(0.0).to(device)

        x1, adj1 = x.to(device), copy.deepcopy(adj).to(device)
        x2, adj2 = (x + perb).to(device), copy.deepcopy(adj).to(device)

        embed = (x2 + self.online_encoder(x2, adj2)).to(device)  # 最终 emb

        online_x = self.online_encoder(x1, adj1).to(device)
        online_y = self.online_encoder(x2, adj2).to(device)

        with torch.no_grad():
            target_y = self.target_encoder(x1, adj1).detach().to(device)
            target_x = self.target_encoder(x2, adj2).detach().to(device)

        online_x = self.predictor(online_x).to(device)
        online_y = self.predictor(online_y).to(device)

        loss = (loss_fn(online_x, target_x) + loss_fn(online_y, target_y)).mean().to(device)

        return embed, loss



class ABQR(nn.Module):
    def __init__(self, skill_max, drop_feat1, drop_feat2, drop_edge1, drop_edge2, positive_matrix, pro_max, lamda,
                 contrast_batch, tau, lamda1, top_k, d, p, head=1, graph_aug='knn', gnn_mode='gcn'):
        super(ABQR, self).__init__()

        self.lamda = lamda
        self.head = head

        # BGRL 是对比学习模块
        self.gcl = BGRL(d, p, drop_feat1, drop_feat2, drop_edge1, drop_edge2).to(device)

        # 图卷积网络
        self.gcn = GCNConv(d, d, p).to(device)

        # 问题嵌入和回答嵌入
        self.pro_embed = nn.Parameter(torch.ones((pro_max, d)).to(device))  # 问题嵌入
        nn.init.xavier_uniform_(self.pro_embed)

        self.ans_embed = nn.Embedding(2, d).to(device)  # 回答嵌入

        # 注意力机制
        self.attn = nn.MultiheadAttention(d, 8, dropout=p).to(device)
        self.attn_dropout = nn.Dropout(p).to(device)
        self.attn_layer_norm = nn.LayerNorm(d).to(device)

        # 前馈神经网络
        self.FFN = nn.Sequential(
            nn.Linear(d, d).to(device),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(d, d).to(device),
            nn.Dropout(p),
        ).to(device)
        self.FFN_layer_norm = nn.LayerNorm(d).to(device)

        # 最终预测层
        self.pred = nn.Linear(d, 1).to(device)

        # LSTM 模块
        self.lstm = nn.LSTM(d, d, batch_first=True).to(device)
        self.origin_lstm = nn.LSTM(2 * d, 2 * d, batch_first=True).to(device)
        self.oppo_lstm = nn.LSTM(d, d, batch_first=True).to(device)

        # 输出层
        self.origin_out = nn.Sequential(
            nn.Linear(2 * d, d).to(device),
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.Linear(d, 1).to(device)
        ).to(device)
        self.oppo_out = nn.Sequential(
            nn.Linear(2 * d, d).to(device),
            nn.ReLU(),
            nn.Linear(d, 1).to(device)
        ).to(device)

        self.dropout = nn.Dropout(p=p).to(device)

        # 编码器和解码器 LSTM
        self.encoder_lstm = nn.LSTM(d, d, batch_first=True).to(device)
        self.decoder_lstm = nn.LSTM(d, d, batch_first=True).to(device)

        # 编码器的 token 和映射
        self.enc_token = nn.Parameter(torch.rand(1, d).to(device))
        self.enc_dec = nn.Linear(d, d).to(device)

        # 技能分类
        self.classify = nn.Sequential(
            nn.Linear(d, skill_max).to(device)
        ).to(device)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def compute_loss(self, pro_clas, true_clas):
        """计算分类损失，使用交叉熵损失函数。"""
        pro_clas = pro_clas.view(-1, pro_clas.shape[-1]).to(device)
        true_clas = true_clas.view(-1).to(device)
        loss = F.cross_entropy(pro_clas, true_clas)
        return loss

    def encoding_mask_seq(self, x, mask_rate=0.3):
        """对输入序列进行随机掩码，增加鲁棒性。"""
        num_nodes = x.shape[1]
        perm = torch.randperm(num_nodes, device=device)
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[:num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]
        out_x = x.clone()
        token_nodes = mask_nodes
        out_x[:, token_nodes] = self.enc_token
        return out_x.to(device), mask_nodes.to(device), keep_nodes.to(device)

    def forward(self, last_pro, last_ans, last_skill, next_pro, next_skill, matrix, perb=None):
        """
        last_pro: 上一个问题的索引
        last_ans: 上一个问题的回答嵌入
        next_pro: 下一个问题的索引
        matrix: 图矩阵
        perb: 扰动项
        """
        batch, seq = last_pro.shape[0], last_pro.shape[1]

        # 通过对比学习模块生成问题嵌入和对比损失
        pro_embed, contrast_loss = self.gcl(self.pro_embed, matrix, perb)
        pro_embed = pro_embed.to(device)
        contrast_loss = 0.1 * contrast_loss.to(device)  # 加权的对比损失

        # 获取上一个问题和下一个问题的嵌入
        last_pro_embed = F.embedding(last_pro, pro_embed).to(device)  # [80,199,128]
        next_pro_embed = F.embedding(next_pro, pro_embed).to(device)  # [80,199,128]

        # 获取回答的嵌入
        ans_embed = self.ans_embed(last_ans).to(device)

        X = (last_pro_embed + ans_embed).to(device)

        return X, next_pro_embed, contrast_loss

