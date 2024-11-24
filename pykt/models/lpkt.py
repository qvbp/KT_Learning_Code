#!/usr/bin/env python
# coding=utf-8

import torch
from torch import nn

# from models.utils import RobertaEncode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LPKT(nn.Module):
    def __init__(self, n_at, n_it, n_exercise, n_question, d_a, d_e, d_k, gamma=0.03, dropout=0.2, q_matrix="",
                 emb_type="qid", emb_path="", pretrain_dim=768, use_time=True):
        super(LPKT, self).__init__()
        self.model_name = "lpkt"
        self.d_k = d_k  # kc维度
        self.d_a = d_a  # 回答维度
        self.d_e = d_e  # 问题维度
        q_matrix[q_matrix == 0] = gamma
        self.q_matrix = q_matrix.to(device)
        self.n_question = n_question
        print(f"n_question:{self.n_question}")
        self.emb_type = emb_type
        self.use_time = use_time

        # nn.Embedding()里面的参数，第一个是词汇数量，第二维是映射的维度
        self.at_embed = nn.Embedding(n_at + 10, d_k).to(device)
        torch.nn.init.xavier_uniform_(self.at_embed.weight)
        self.it_embed = nn.Embedding(n_it + 10, d_k).to(device)
        torch.nn.init.xavier_uniform_(self.it_embed.weight)
        self.e_embed = nn.Embedding(n_exercise + 10, d_e).to(device)  # +10是为了预留大小
        torch.nn.init.xavier_uniform_(self.e_embed.weight)

        if emb_type.startswith("qidcatr"):
            self.interaction_emb = nn.Embedding(n_exercise * 2, d_k).to(device)
            self.catrlinear = nn.Linear(d_k * 2, d_k).to(device)
            self.pooling = nn.MaxPool1d(2, stride=2).to(device)
            self.avg_pooling = nn.AvgPool1d(2, stride=2).to(device)
        if emb_type.startswith("qidrobertacatr"):
            self.catrlinear = nn.Linear(d_k * 3, d_k).to(device)
            self.pooling = nn.MaxPool1d(3, stride=3).to(device)
            self.avg_pooling = nn.AvgPool1d(3, stride=3).to(device)
        # if emb_type.find("roberta") != -1:
        #     self.roberta_emb = RobertaEncode(self.d_k, emb_path, pretrain_dim)

        self.linear_0 = nn.Linear(d_a + d_e, d_k).to(device)  # 用来编码没有att的知识元组
        torch.nn.init.xavier_uniform_(self.linear_0.weight)
        self.linear_1 = nn.Linear(d_a + d_e + d_k, d_k).to(device)  # 用来编码有att的知识元组
        torch.nn.init.xavier_uniform_(self.linear_1.weight)

        self.linear_2 = nn.Linear(4 * d_k, d_k).to(device)  # 算学习收益（这个是有答题时间间隔）
        torch.nn.init.xavier_uniform_(self.linear_2.weight)

        self.linear_3 = nn.Linear(4 * d_k, d_k).to(device)  # 算学习门，也就是控制学生知识吸收能力的系数（这个是有答题时间间隔）
        torch.nn.init.xavier_uniform_(self.linear_3.weight)

        self.linear_4 = nn.Linear(3 * d_k, d_k).to(device)  # 算遗忘门（这个是有答题时间间隔）
        torch.nn.init.xavier_uniform_(self.linear_4.weight)

        self.linear_5 = nn.Linear(d_e + d_k, d_k).to(device)  # 用来预测下一道题
        torch.nn.init.xavier_uniform_(self.linear_5.weight)

        self.linear_6 = nn.Linear(3 * d_k, d_k).to(device)  # 算学习收益（这个是没有答题时间间隔）
        torch.nn.init.xavier_uniform_(self.linear_6.weight)

        self.linear_7 = nn.Linear(3 * d_k, d_k).to(device)  # 算学习门，也就是控制学生知识吸收能力的系数（这个是没有答题时间间隔）
        torch.nn.init.xavier_uniform_(self.linear_7.weight)

        self.linear_8 = nn.Linear(2 * d_k, d_k).to(device)  # 算遗忘门（这个是没有答题时间间隔）
        torch.nn.init.xavier_uniform_(self.linear_8.weight)

        self.tanh = nn.Tanh().to(device)
        self.sig = nn.Sigmoid().to(device)
        self.dropout = nn.Dropout(dropout).to(device)

    def forward(self, e_data, a_data, it_data=None, at_data=None, qtest=False):
        e_data = e_data.to(device)  # 题目编码 [64,200]
        a_data = a_data.to(device)  # 回答onehot [64,200]
        if it_data is not None:
            it_data = it_data.to(device)  # 每一道题之间的时间间隔 [64,200]
        if at_data is not None:  # assist2012没有at
            at_data = at_data.to(device)

        emb_type = self.emb_type
        batch_size, seq_len = e_data.size(0), e_data.size(1)  # 64，200
        e_embed_data = self.e_embed(e_data)  # 编码题目  [64, 200, 16]

        if self.use_time:  # use_time=True
            if at_data is not None:
                # print("*"*100)
                at_embed_data = self.at_embed(at_data)  # 编码att，即回答时间  2012暂时没有
            it_embed_data = self.it_embed(it_data)  # 编码itt，即间隔时间 [64, 200, 64]

        # 对a_data进行重塑，方便后续计算  a_data代表回答，也就是答案 
        a_data = a_data.view(-1, 1).repeat(1, self.d_a).view(batch_size, -1, self.d_a)  # [64, 200, 64]
    
        # 知识状态 ht-1 [64, 266, 64]
        h_pre = nn.init.xavier_uniform_(torch.zeros(self.n_question + 1, self.d_k)).repeat(batch_size, 1, 1).to(device)
        
        # 相关知识状态 ~ht-1
        h_tilde_pre = None

        if emb_type == "qid":
            if self.use_time and at_data is not None:
                # 编码知识元组: lt=W(et+att+at)+b1  这里是所有的知识元组
                all_learning = self.linear_1(torch.cat((e_embed_data, at_embed_data, a_data), 2))
            else:
                # 编码知识元组: lt=W(et+at)+b1  这里是所有的知识元组
                all_learning = self.linear_0(torch.cat((e_embed_data, a_data), 2))  # [64, 200, 64]
        else:
            # 处理其他 emb_type 的情况
            raise ValueError(f"Unsupported emb_type: {emb_type}")

        # 之前的知识元组lt-1
        learning_pre = torch.zeros(batch_size, self.d_k).to(device)  # [64, 64]
    

        pred = torch.zeros(batch_size, seq_len).to(device)  # [64, 200]
        hidden_state = torch.zeros(batch_size, seq_len, self.d_k).to(device)  # [64, 200, 64]

        # 注意所有可以t-1的张量开始都是初始化为可学习的张量
        for t in range(0, seq_len - 1):
            e = e_data[:, t]  # 取每个批次第t个题目
            # q_e: (bs, 1, n_skill)
            # 习题与知识点的关系矩阵
            q_e = self.q_matrix[e].view(batch_size, 1, -1).to(device)
            # print("*"*100)
            # print(f"q_e shape: {q_e.shape}")
            # # print(q_e)
            # print("*"*100)
            if self.use_time:
                it = it_embed_data[:, t]  # 回答t这道题的时间间隔
                # Learning Module
                # 算出来~ht-1= qet * ht-1
                if h_tilde_pre is None:
                    c_pre = torch.unsqueeze(torch.sum(torch.squeeze(q_e, dim=1), 1), -1)
                    h_tilde_pre = q_e.bmm(h_pre).view(batch_size, self.d_k) / c_pre
                # 获取lt
                learning = all_learning[:, t]
                # 算出lgt（学习收益）= tanh(w2[lt-1 + itt + lt + ~ht-1])
                learning_gain = self.linear_2(torch.cat((learning_pre, it, learning, h_tilde_pre), 1))
                learning_gain = self.tanh(learning_gain)
                # gamma_l：（控制学生知识吸收能力系数）
                gamma_l = self.linear_3(torch.cat((learning_pre, it, learning, h_tilde_pre), 1))
            else:
                # Learning Module
                # 算出来~ht-1
                if h_tilde_pre is None:
                    c_pre = torch.unsqueeze(torch.sum(torch.squeeze(q_e, dim=1), 1), -1)
                    h_tilde_pre = q_e.bmm(h_pre).view(batch_size, self.d_k) / c_pre
                # 获取lt
                learning = all_learning[:, t]
                # 算出lgt（学习收益）
                learning_gain = self.linear_6(torch.cat((learning_pre, learning, h_tilde_pre), 1))
                # 学习收益
                learning_gain = self.tanh(learning_gain)
                # gamma_l：（控制学生知识吸收能力系数）
                gamma_l = self.linear_7(torch.cat((learning_pre, learning, h_tilde_pre), 1))
            # gamma_l 控制学生的知识吸收能力
            gamma_l = self.sig(gamma_l)
            # 实际学习收益
            LG = gamma_l * ((learning_gain + 1) / 2)
            # 相关学习收益
            LG_tilde = self.dropout(q_e.transpose(1, 2).bmm(LG.view(batch_size, 1, -1)))

            # Forgetting Module
            # h_pre: (bs, n_skill, d_k)
            # LG: (bs, d_k)
            # it: (bs, d_k)
            n_skill = LG_tilde.size(1)
            # gamma_f = sig(wt(ht-1 + lGt + itt) + b)  算遗忘系数
            if self.use_time:
                gamma_f = self.sig(self.linear_4(torch.cat((
                    h_pre,
                    LG.repeat(1, n_skill).view(batch_size, -1, self.d_k),
                    it.repeat(1, n_skill).view(batch_size, -1, self.d_k)
                ), 2)))
            else:
                gamma_f = self.sig(self.linear_8(torch.cat((
                    h_pre,
                    LG.repeat(1, n_skill).view(batch_size, -1, self.d_k)
                ), 2)))
            # 获得当前知识状态 ht = ~LG + gamma_f * ht-1
            h = (LG_tilde + gamma_f * h_pre)

            # Predicting Module
            c_tilde = torch.unsqueeze(
                torch.sum(torch.squeeze(self.q_matrix[e_data[:, t + 1]].view(batch_size, 1, -1), dim=1), 1), -1)
            # 算~ht=qet+1 * ht
            h_tilde = self.q_matrix[e_data[:, t + 1]].view(batch_size, 1, -1).bmm(h).view(batch_size, self.d_k) / c_tilde
            # yt+1 = sig(w5[et+1 + ~ht] + b5)  这里算预测结果
            y = self.sig(self.linear_5(torch.cat((e_embed_data[:, t + 1], h_tilde), 1))).sum(1) / self.d_k
            pred[:, t + 1] = y
            hidden_state[:, t + 1, :] = h_tilde

            # prepare for next prediction
            learning_pre = learning
            h_pre = h
            h_tilde_pre = h_tilde
        if not qtest:
            return pred
        else:
            return pred, hidden_state[:, :-1, :], e_embed_data
