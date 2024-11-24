import os

import numpy as np
import torch
from torch import nn as nn

from torch.nn import Module, Embedding, LSTM, Linear, Dropout
from .ABQR_model import ABQR


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载GCN要的技能和问题矩阵
pre_load_gcn = "../data/bridge2algebra2006/ques_skill_gcn_adj.pt"
matrix = torch.load(pre_load_gcn).to(device)
if not matrix.is_sparse:
    matrix = matrix.to_sparse()


class DKT_ABQR(Module):
    def __init__(self, num_q, num_c, emb_size, dropout=0.1, emb_type='qid', emb_path="", pretrain_dim=768):
        super().__init__()
        self.model_name = "dkt_abqr"
        self.num_c = num_c
        self.num_q = num_q
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.emb_type = emb_type
        self.dropout = dropout
        self.matrix = matrix.to(device)  # 确保矩阵在 device 上

        '''
        ABQR  模型初始化的一些参数
        def __init__(self, skill_max, drop_feat1, drop_feat2, drop_edge1, drop_edge2, positive_matrix, pro_max, lamda,
                     contrast_batch, tau, lamda1, top_k, d, p, head=1, graph_aug='knn', gnn_mode='gcn'):
        '''

        drop_feat1 = 0.2
        drop_feat2 = 0.3
        drop_edge1 = 0.3
        drop_edge2 = 0.2
        positive_matrix = self.matrix
        pro_max = self.num_q
        lamda = 5
        contrast_batch = 1000
        tau = 0.8
        lamda1 = 20
        top_k = 25
        d = self.emb_size
        p = 0.4
        head = 8
        graph_aug = 'knn'
        gnn_mode = 'gcn'

        self.abqr = ABQR(self.num_c, drop_feat1, drop_feat2, drop_edge1, drop_edge2, positive_matrix, pro_max, lamda,
                         contrast_batch, tau, lamda1, top_k, d, p, head, graph_aug, gnn_mode).to(device)

        self.lstm_layer = LSTM(self.emb_size, self.hidden_size, batch_first=True).to(device)
        self.dropout_layer = Dropout(dropout).to(device)
        self.out_layer = nn.Sequential(
            nn.Linear(2 * d, d).to(device),
            nn.ReLU(),
            nn.Dropout(p=self.dropout).to(device),
            nn.Linear(d, 1).to(device)
        ).to(device)

    def forward(self, last_pro, last_ans, last_skill, next_pro, next_skill, perb=None):
        '''
        def forward(self, last_pro, last_ans, last_skill, next_pro, next_skill, matrix, perb=None):
        '''
        # 将输入移动到设备
        last_pro = last_pro.to(device)
        last_ans = last_ans.to(device)
        last_skill = last_skill.to(device)
        next_pro = next_pro.to(device)
        next_skill = next_skill.to(device)
        if perb is not None:
            perb = perb.to(device)

        # 调用 ABQR 的 forward 方法
        xemb, next_xemb, contrast_loss = self.abqr(
            last_pro, last_ans, last_skill, next_pro, next_skill, self.matrix, perb
        )
        xemb = xemb.to(device)
        next_xemb = next_xemb.to(device)
        contrast_loss = contrast_loss.to(device)

        # LSTM 层
        h, _ = self.lstm_layer(xemb.to(device))

        # 输出层
        y = torch.sigmoid(self.out_layer(torch.cat([h, next_xemb], dim=-1).to(device))).squeeze(-1)

        return y, contrast_loss
