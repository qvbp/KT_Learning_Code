import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np
from .utils import transformer_FFN, ut_mask, pos_encode, get_clones
from torch.nn import Module, Embedding, LSTM, Linear, Dropout, LayerNorm, TransformerEncoder, TransformerEncoderLayer, \
    MultiLabelMarginLoss, MultiLabelSoftMarginLoss, CrossEntropyLoss, BCELoss, MultiheadAttention
from torch.nn.functional import one_hot, cross_entropy, multilabel_margin_loss, binary_cross_entropy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2


class fliicbsimpleKT(nn.Module):
    def __init__(self, n_question, n_pid,
                 d_model, n_blocks, dropout, d_ff=256,
                 loss1=0.5, loss2=0.5, loss3=0.5, start=50, num_layers=2, nheads=4, seq_len=200,
                 kq_same=1, final_fc_dim=512, final_fc_dim2=256, num_attn_heads=8, separate_qa=False, l2=1e-5,
                 emb_type="qid", emb_path="", pretrain_dim=768):
        super().__init__()
        """
        Input:
            d_model: dimension of attention block
            final_fc_dim: dimension of final fully connected net before prediction
            num_attn_heads: number of heads in multi-headed attention
            d_ff : dimension for fully conntected net inside the basic block
            kq_same: if key query same, kq_same=1, else = 0
        """
        self.model_name = "simplekt"
        print(f"model_name: {self.model_name}, emb_type: {emb_type}")
        self.n_question = n_question
        self.dropout = dropout
        self.kq_same = kq_same
        self.n_pid = n_pid
        self.l2 = l2
        self.model_type = self.model_name
        self.separate_qa = separate_qa
        self.emb_type = emb_type
        self.cnt = 0
        embed_l = d_model
        self.emb_size = d_model
        if self.n_pid > 0:
            if emb_type.find("scalar") != -1:
                # print(f"question_difficulty is scalar")
                self.difficult_param = nn.Embedding(self.n_pid + 1, 1).to(device)  # 题目难度
            else:
                self.difficult_param = nn.Embedding(self.n_pid + 1, embed_l * 2).to(device)  # 题目难度
            self.q_embed_diff = nn.Embedding(self.n_question + 1,
                                             embed_l * 2).to(device)  # question emb, 总结了包含当前question（concept）的problems（questions）的变化
            # 这个可以不用管，暂时没用上
            self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, embed_l*2).to(device)  # interaction emb, 同上

        if emb_type.startswith("qid"):
            # n_question+1 ,d_model
            self.q_embed = nn.Embedding(self.n_question, embed_l * 2).to(device)
            self.qx_embed = nn.Embedding(self.n_question, embed_l).to(device)
            self.q_pid_embed = nn.Embedding(self.n_pid, embed_l).to(device)
            if self.separate_qa:
                self.qa_embed = nn.Embedding(2 * self.n_question + 1, embed_l).to(device)
            else:  # false default
                self.qa_embed = nn.Embedding(2, embed_l).to(device)
        # Architecture Object. It contains stack of attention block
        self.model = Architecture(n_question=n_question, n_blocks=n_blocks, n_heads=num_attn_heads, dropout=dropout,
                                  d_model=d_model*2, d_feature=(d_model*2) / num_attn_heads, d_ff=d_ff, kq_same=self.kq_same,
                                  model_type=self.model_type, seq_len=seq_len).to(device)

        self.out = nn.Sequential(
            nn.Linear(d_model*2 + embed_l*2,
                      final_fc_dim).to(device), nn.ReLU().to(device), nn.Dropout(self.dropout).to(device),
            nn.Linear(final_fc_dim, final_fc_dim2).to(device), nn.ReLU().to(device),
            nn.Dropout(self.dropout).to(device),
            nn.Linear(final_fc_dim2, 1).to(device)
        )

        self.reset()

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.n_pid + 1 and self.n_pid > 0:
                torch.nn.init.constant_(p, 0.)

    def base_emb(self, q_data, pid_data, target):
        q_data, target = q_data.to(device), target.to(device)
        q_embed_data = self.q_embed(q_data)  # BS, seqlen,  d_model# c_ct  d_model*2
        qa_embed_data = self.qx_embed(q_data)  # d_model
        q_pid_embed_data = self.q_pid_embed(pid_data)
        qa_embed_data = qa_embed_data + q_pid_embed_data
        # print("+"*100)
        # print(f"q_embed_data.shape: {q_embed_data.shape}")
        # print(f"qa_embed_data.shape: {qa_embed_data.shape}")
        # print("+"*100)

        if self.separate_qa:  # 这里没有用到
            qa_data = q_data + self.n_question * target
            qa_embed_data = self.qa_embed(qa_data)
        else:
            # BS, seqlen, d_model # c_ct+ g_rt =e_(ct,rt)
            # qa_embed_data = self.qa_emb(target) + q_embed_data
            qa_embed_data = torch.cat([qa_embed_data.mul((1 - target).unsqueeze(-1).repeat(1, 1, self.emb_size)),
                            qa_embed_data.mul(target.unsqueeze(-1).repeat(1, 1, self.emb_size))], dim=-1)
            # qa_reshape = nn.Linear(self.emb_size*2, self.emb_size).to(device)
            # qa_embed_data = qa_reshape(qa_embed_data)
            # MLP = nn.Sequential(nn.Linear(self.emb_size*2, self.emb_size*4),
            #                     nn.ReLU(),
            #                     nn.Linear(self.emb_size*4, self.emb_size))
            # qa_embed_data = MLP(qa_embed_data)

        return q_embed_data, qa_embed_data

    def get_attn_pad_mask(self, sm):
        sm = sm.to(device)
        batch_size, l = sm.size()
        pad_attn_mask = sm.data.eq(0).unsqueeze(1)
        pad_attn_mask = pad_attn_mask.expand(batch_size, l, l)
        return pad_attn_mask.repeat(self.nhead, 1, 1)

    def forward(self, dcur, qtest=False, train=False):
        dcur = {k: v.to(device) for k, v in dcur.items()}
        q, c, r = dcur["qseqs"].long(), dcur["cseqs"].long(), dcur["rseqs"].long()
        qshft, cshft, rshft = dcur["shft_qseqs"].long(), dcur["shft_cseqs"].long(), dcur["shft_rseqs"].long()
        pid_data = torch.cat((q[:, 0:1], qshft), dim=1)
        q_data = torch.cat((c[:, 0:1], cshft), dim=1)
        target = torch.cat((r[:, 0:1], rshft), dim=1)

        emb_type = self.emb_type

        # Batch First
        if emb_type.startswith("qid"):
            q_embed_data, qa_embed_data = self.base_emb(q_data, pid_data, target)
        if self.n_pid > 0 and emb_type.find("norasch") == -1:  # have problem id
            if emb_type.find("aktrasch") == -1:
                q_embed_diff_data = self.q_embed_diff(q_data)  # d_ct 总结了包含当前question（concept）的problems（questions）的变化
                pid_embed_data = self.difficult_param(pid_data)  # uq 当前problem的难度
                q_embed_data = q_embed_data + pid_embed_data * \
                               q_embed_diff_data  # uq *d_ct + c_ct # question encoder

            else:  # 这里没有用到
                q_embed_diff_data = self.q_embed_diff(q_data)  # d_ct 总结了包含当前question（concept）的problems（questions）的变化
                pid_embed_data = self.difficult_param(pid_data)  # uq 当前problem的难度
                q_embed_data = q_embed_data + pid_embed_data * \
                               q_embed_diff_data  # uq *d_ct + c_ct # question encoder

                qa_embed_diff_data = self.qa_embed_diff(
                    target)  # f_(ct,rt) or #h_rt (qt, rt)差异向量
                qa_embed_data = qa_embed_data + pid_embed_data * \
                                (qa_embed_diff_data + q_embed_diff_data)  # + uq *(h_rt+d_ct) # （q-response emb diff + question emb diff）

        # BS.seqlen,d_model
        # Pass to the decoder
        # output shape BS,seqlen,d_model or d_model//2
        y2, y3 = 0, 0
        if emb_type in ["qid", "qidaktrasch", "qid_scalar", "qid_norasch"]:
            d_output = self.model(q_embed_data, qa_embed_data)

            concat_q = torch.cat([d_output, q_embed_data], dim=-1)
            output = self.out(concat_q).squeeze(-1)
            m = nn.Sigmoid()
            preds = m(output)

        if train:
            # if self.cnt < 20:
            #     print(f"y2: {y2}  y3: {y3}")
            #     self.cnt = self.cnt + 1
            return preds, y2, y3
        else:
            if qtest:
                return preds, concat_q
            else:
                return preds


class Architecture(nn.Module):
    def __init__(self, n_question, n_blocks, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same, model_type, seq_len):
        super().__init__()
        """
            n_block : number of stacked blocks in the attention
            d_model : dimension of attention input/output
            d_feature : dimension of input in each of the multi-head attention part.
            n_head : number of heads. n_heads*d_feature = d_model
        """
        self.d_model = d_model
        self.model_type = model_type

        if model_type in {'simplekt'}:
            self.blocks_2 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same).to(device)
                for _ in range(n_blocks)
            ])
        self.position_emb = CosinePositionalEmbedding(d_model=self.d_model, max_len=seq_len).to(device)
        # self.position_emb_qa = CosinePositionalEmbedding(d_model=self.d_model*2, max_len=seq_len).to(device)

    def forward(self, q_embed_data, qa_embed_data):
        q_embed_data, qa_embed_data = q_embed_data.to(device), qa_embed_data.to(device)
        # target shape  bs, seqlen
        seqlen, batch_size = q_embed_data.size(1), q_embed_data.size(0)

        q_posemb = self.position_emb(q_embed_data)
        q_embed_data = q_embed_data + q_posemb
        qa_posemb = self.position_emb(qa_embed_data)
        # print("+"*100)
        # print(f"qa_embed_data.shape: {qa_embed_data.shape}")
        # print(f"qa_posemb.shape: {qa_posemb.shape}")
        # print("+"*100)
        qa_embed_data = qa_embed_data + qa_posemb

        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data

        y = qa_pos_embed
        seqlen, batch_size = y.size(1), y.size(0)
        x = q_pos_embed

        # encoder

        for block in self.blocks_2:
            x = block(mask=0, query=x, key=x, values=y,
                      apply_pos=True)  # True: +FFN+残差+laynorm 非第一层与0~t-1的的q的attention, 对应图中Knowledge Retriever
            # mask=0，不能看到当前的response, 在Knowledge Retrever的value全为0，因此，实现了第一题只有question信息，无qa信息的目的
            # print(x[0,0,:])
        return x


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same):
        super().__init__()
        """
            This is a Basic Block of Transformer paper. It containts one Multi-head attention object. Followed by layer norm and postion wise feedforward net and dropout layer.
        """
        kq_same = kq_same == 1
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same).to(device)

        # Two layer norm layer and two droput layer
        self.layer_norm1 = nn.LayerNorm(d_model).to(device)
        self.dropout1 = nn.Dropout(dropout).to(device)

        self.linear1 = nn.Linear(d_model, d_ff).to(device)
        self.activation = nn.ReLU().to(device)
        self.dropout = nn.Dropout(dropout).to(device)
        self.linear2 = nn.Linear(d_ff, d_model).to(device)

        self.layer_norm2 = nn.LayerNorm(d_model).to(device)
        self.dropout2 = nn.Dropout(dropout).to(device)
        # self.icb = ICB(d_model, d_model, 1, 3)

    def forward(self, mask, query, key, values, apply_pos=True):
        """
        Input:
            block : object of type BasicBlock(nn.Module). It contains masked_attn_head objects which is of type MultiHeadAttention(nn.Module).
            mask : 0 means, it can peek only past values. 1 means, block can peek only current and pas values
            query : Query. In transformer paper it is the input for both encoder and decoder
            key : Keys. In transformer paper it is the input for both encoder and decoder
            Values. In transformer paper it is the input for encoder and  encoded output for decoder (in masked attention part)

        Output:
            query: Input gets changed over the layer and returned.

        """
        # ensure inputs are on the same device
        query, key, values = query.to(device), key.to(device), values.to(device)

        seqlen, batch_size = query.size(1), query.size(0)
        nopeek_mask = np.triu(
            np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)
        if mask == 0:  # If 0, zero-padding is needed.
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask,
                zero_pad=True)  # 只能看到之前的信息，当前的信息也看不到，此时会把第一行score全置0，表示第一道题看不到历史的interaction信息，第一题attn之后，对应value全0
        else:
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False)

        query = query + self.dropout1((query2))  # 残差1
        query = self.layer_norm1(query)  # layer norm
        if apply_pos:
            query2 = self.linear2(self.dropout(  # FFN
                self.activation(self.linear1(query))))

            # query2 = self.icb(query)  # 将FFN替换成ICB

            query = query + self.dropout2((query2))  # 残差
            query = self.layer_norm2(query)  # lay norm
        return query


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True):
        super().__init__()
        """
        It has projection layer for getting keys, queries and values. Followed by attention and a connected layer.
        """
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same

        self.v_linear = nn.Linear(d_model, d_model, bias=bias).to(device)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias).to(device)
        if kq_same is False:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias).to(device)
        self.dropout = nn.Dropout(dropout).to(device)
        self.proj_bias = bias
        self.out_proj = nn.Linear(d_model, d_model, bias=bias).to(device)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.).to(device)
            constant_(self.v_linear.bias, 0.).to(device)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.).to(device)
            constant_(self.out_proj.bias, 0.).to(device)

    def forward(self, q, k, v, mask, zero_pad):
        q, k, v = q.to(device), k.to(device), v.to(device)
        bs = q.size(0)

        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k,
                           mask, self.dropout, zero_pad)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)

        output = self.out_proj(concat)

        return output


def attention(q, k, v, d_k, mask, dropout, zero_pad):
    """
    This is called by Multi-head atention object to find the values.
    """
    # ensure inputs are on the same device
    q, k, v, mask = q.to(q.device), k.to(k.device), v.to(v.device), mask.to(mask.device)

    # d_k: 每一个头的dim
    scores = torch.matmul(q, k.transpose(-2, -1)) / \
             math.sqrt(d_k)  # BS, 8, seqlen, seqlen
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
    # print(f"before zero pad scores: {scores.shape}")
    # print(zero_pad)
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(q.device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)  # 第一行score置0
    # print(f"after zero pad scores: {scores}")
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    # import sys
    # sys.exit()
    return output


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=True).to(device)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)


class CosinePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False).to(device)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, **kwargs):
        self.__padding = (kernel_size - 1) * dilation
        super().__init__(in_channels, out_channels, kernel_size, stride, padding=self.__padding, dilation=dilation, **kwargs)

    def forward(self, x):
        x = x.to(self.weight.device)  # Ensure input is on the same device as the layer
        result = super().forward(x)
        if self.__padding != 0:
            return result[:, :, :-self.__padding]  # [batch_size, channels, seq_len]
        return result


class ICB(nn.Module):
    def __init__(self, in_features, hidden_features, kernel_size1=1, kernel_size2=3, dropout=0.1):
        super().__init__()
        self.conv1 = CausalConv1d(in_features, hidden_features, kernel_size1).to(device)
        self.conv2 = CausalConv1d(in_features, hidden_features, kernel_size2).to(device)
        self.conv3 = CausalConv1d(hidden_features, in_features, 1).to(device)
        self.activation = nn.GELU().to(device)
        self.dropout = nn.Dropout(dropout).to(device)

    def forward(self, x):
        x = x.transpose(1, 2).to(device)  # (batch_size, feature_dim, sequence_length)

        x1 = self.conv1(x)
        x1_1 = self.activation(x1)
        x1_2 = self.dropout(x1_1)

        x2 = self.conv2(x)
        x2_1 = self.activation(x2)
        x2_2 = self.dropout(x2_1)

        out1 = x1 * x2_2
        out2 = x2 * x1_2

        x = self.conv3(out1 + out2)
        x = x.transpose(1, 2)  # (batch_size, sequence_length, feature_dim)
        return x
