import torch

from torch import nn
from torch.nn import Module, Embedding, LSTM, Linear, Dropout, LayerNorm, TransformerEncoder, TransformerEncoderLayer, CrossEntropyLoss
from .utils import ut_mask

device = "cpu" if not torch.cuda.is_available() else "cuda"

class ATDKT(Module):
    def __init__(self, num_q, num_c, seq_len, emb_size, dropout=0.1, emb_type='qid', 
            num_layers=1, num_attn_heads=5, l1=0.5, l2=0.5, l3=0.5, start=50, emb_path="", pretrain_dim=768):
        super().__init__()
        self.model_name = "atdkt"
        print(f"qnum: {num_q}, cnum: {num_c}")
        print(f"emb_type: {emb_type}")
        self.num_q = num_q
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.emb_type = emb_type
        self.printed = False  # 用于标记是否已经执行过print语句
        self.to(device)  # 将设备信息转到

        self.interaction_emb = Embedding(self.num_c * 2, self.emb_size).to(device)

        self.lstm_layer = LSTM(self.emb_size, self.hidden_size, batch_first=True).to(device)
        self.dropout_layer = Dropout(dropout).to(device)
        self.out_layer = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size//2).to(device), nn.ReLU(), nn.Dropout(dropout).to(device),
                Linear(self.hidden_size//2, self.num_c).to(device))

        if self.emb_type.endswith("predhis"):
            self.l1 = l1
            self.l2 = l2
            if self.emb_type.find("cemb") != -1:
                self.concept_emb = Embedding(self.num_c, self.emb_size).to(device) # add concept emb
            if self.emb_type.find("qemb") != -1:
                self.question_emb = Embedding(self.num_q, self.emb_size).to(device)
            
            self.start = start
            self.hisclasifier = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size//2).to(device), nn.ReLU(), nn.Dropout(dropout).to(device),
                nn.Linear(self.hidden_size//2, 1).to(device))
            self.hisloss = nn.MSELoss().to(device)

        if self.emb_type.endswith("predcurc"): # predict cur question' cur concept
            self.l1 = l1
            self.l2 = l2
            self.l3 = l3
            if self.num_q > 0:
                self.question_emb = Embedding(self.num_q, self.emb_size).to(device) # 1.2
            if self.emb_type.find("trans") != -1:
                self.nhead = num_attn_heads
                d_model = self.hidden_size# * 2
                encoder_layer = TransformerEncoderLayer(d_model, nhead=self.nhead).to(device)
                encoder_norm = LayerNorm(d_model).to(device)
                self.trans = TransformerEncoder(encoder_layer, num_layers=num_layers, norm=encoder_norm).to(device)
            else:    
                self.qlstm = LSTM(self.emb_size, self.hidden_size, batch_first=True).to(device)
            self.qclasifier = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size//2).to(device), nn.ReLU(), nn.Dropout(dropout).to(device),
                Linear(self.hidden_size//2, self.num_c).to(device))
            if self.emb_type.find("cemb") != -1:
                self.concept_emb = Embedding(self.num_c, self.emb_size).to(device) # add concept emb

            self.closs = CrossEntropyLoss().to(device)
            if self.emb_type.find("his") != -1:
                self.start = start
                self.hisclasifier = nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size//2).to(device), nn.ReLU(), nn.Dropout(dropout).to(device),
                    nn.Linear(self.hidden_size//2, 1).to(device))
                self.hisloss = nn.MSELoss().to(device)

    def predcurc(self, dcur, q, c, r, xemb, train):
        emb_type = self.emb_type
        y2, y3 = 0, 0
        # if self.printed == False:
        #     print(f"emb_type: {emb_type}")
        
        if emb_type.find("delxemb") != -1:
            # if self.printed == False:
            #     print("****81****delxemb")
            qemb = self.question_emb(q).to(device)
            cemb = self.concept_emb(c).to(device)
            catemb = qemb + cemb
        else:
            catemb = xemb.to(device)
            if self.num_q > 0:
                qemb = self.question_emb(q).to(device)
                catemb = qemb + xemb
                
            if emb_type.find("cemb") != -1:
                cemb = self.concept_emb(c).to(device)
                catemb += cemb
            else:
                cemb = torch.zeros_like(catemb).to(device)  # 添加默认值以防止cemb未定义

        if emb_type.find("trans") != -1:
            # if self.printed == False:
            #     print("****99****trans")
            mask = ut_mask(seq_len = catemb.shape[1]).to(device)
            qh = self.trans(catemb.transpose(0,1), mask).transpose(0,1).to(device)
        else:
            qh, _ = self.qlstm(catemb)
        if train:
            sm = dcur["smasks"].long().to(device)
            start = 0
            cpreds = self.qclasifier(qh[:,start:,:]).to(device)
            flag = sm[:,start:]==1
            y2 = self.closs(cpreds[flag], c[:,start:][flag])

        # predict response
        xemb = xemb + qh + cemb
        if emb_type.find("qemb") != -1:
            xemb = xemb + qemb
        h, _ = self.lstm_layer(xemb)

        # predict history correctness rates
        rpreds = None
        if train and emb_type.find("his") != -1:
            # print("******120120********")
            sm = dcur["smasks"].long().to(device)
            start = self.start
            rpreds = torch.sigmoid(self.hisclasifier(h)).squeeze(-1)
            rsm = sm[:,start:]
            rflag = rsm==1
            rtrues = dcur["historycorrs"][:,start:].to(device)
            y3 = self.hisloss(rpreds[:,start:][rflag], rtrues[rflag])

        # predict response
        h = self.dropout_layer(h)
        y = self.out_layer(h)
        y = torch.sigmoid(y)
        return y, y2, y3

    def forward(self, dcur, train=False): ## F * xemb
        # print(f"keys: {dcur.keys()}")
        q, c, r = dcur["qseqs"].long().to(device), dcur["cseqs"].long().to(device), dcur["rseqs"].long().to(device)
        # q, c, r = dcur["qseqs"].long(), dcur["cseqs"].long(), dcur["rseqs"].long()
        y2, y3 = 0, 0

        emb_type = self.emb_type
        if emb_type.startswith("qid"):
            x = c + self.num_c * r
            xemb = self.interaction_emb(x).to(device)
        rpreds, qh = None, None
        if emb_type == "qid":
            h, _ = self.lstm_layer(xemb)
            h = self.dropout_layer(h)
            y = torch.sigmoid(self.out_layer(h))
        elif emb_type.endswith("predhis"): # only predict history correct ratios
            # predict response
            if self.emb_type.find("cemb") != -1:
                cemb = self.concept_emb(c).to(device)
                xemb = xemb + cemb
            if emb_type.find("qemb") != -1:
                qemb = self.question_emb(q).to(device)
                xemb = xemb + qemb
            h, _ = self.lstm_layer(xemb)
            # predict history correctness rates
            if train:
                sm = dcur["smasks"].long().to(device)
                start = self.start
                rpreds = torch.sigmoid(self.hisclasifier(h)[:,start:,:]).squeeze(-1)
                rsm = sm[:,start:]
                rflag = rsm==1
                rtrues = dcur["historycorrs"][:,start:].to(device)
                y2 = self.hisloss(rpreds[rflag], rtrues[rflag])

            h = self.dropout_layer(h)
            y = self.out_layer(h)
            y = torch.sigmoid(y)
        elif emb_type.endswith("predcurc"): # predict current question' current concept
            y, y2, y3 = self.predcurc(dcur, q, c, r, xemb, train)

        if train:
            return y, y2, y3
        else:
            return y
