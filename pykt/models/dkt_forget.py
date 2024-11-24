import torch
from torch.nn import Module, Embedding, LSTM, Linear, Dropout

device = "cpu" if not torch.cuda.is_available() else "cuda"

class DKTForget(Module):
    def __init__(self, num_c, num_rgap, num_sgap, num_pcount, emb_size, dropout=0.1, emb_type='qid', emb_path=""):
        super().__init__()
        self.model_name = "dkt_forget"
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.emb_type = emb_type

        if emb_type.startswith("qid"):
            self.interaction_emb = Embedding(self.num_c * 2, self.emb_size)

        self.c_integration = CIntegration(num_rgap, num_sgap, num_pcount, emb_size)  # θin(vt,ct) = [vt*Cct; ct]
        ntotal = num_rgap + num_sgap + num_pcount
    
        self.lstm_layer = LSTM(self.emb_size + ntotal, self.hidden_size, batch_first=True)
        self.dropout_layer = Dropout(dropout)
        self.out_layer = Linear(self.hidden_size + ntotal, self.num_c)
        

    def forward(self, q, r, dgaps):
        # print("!"*100)
        # print(f"q.shape: {q.shape}")
        # print(f"q: {q[0]}")
        # print("-"*100)
        # print(f"r.shape: {r.shape}")
        # print(f"r: {r[0]}")
        # print("!"*100)
        q, r = q.to(device), r.to(device)
        emb_type = self.emb_type
        if emb_type == "qid":
            x = q + self.num_c * r
            xemb = self.interaction_emb(x)
            theta_in = self.c_integration(xemb, dgaps["rgaps"].to(device).long(), dgaps["sgaps"].to(device).long(), dgaps["pcounts"].to(device).long())

        h, _ = self.lstm_layer(theta_in)
        # 这里把t+1的数据传进去，就是把那3个时间数据传进去
        theta_out = self.c_integration(h, dgaps["shft_rgaps"].to(device).long(), dgaps["shft_sgaps"].to(device).long(), dgaps["shft_pcounts"].to(device).long())
        theta_out = self.dropout_layer(theta_out)
        y = self.out_layer(theta_out)
        y = torch.sigmoid(y)

        return y


class CIntegration(Module):
    def __init__(self, num_rgap, num_sgap, num_pcount, emb_dim) -> None:
        super().__init__()
        self.rgap_eye = torch.eye(num_rgap)  # repeated time
        self.sgap_eye = torch.eye(num_sgap)  # sequence time
        self.pcount_eye = torch.eye(num_pcount)  # past trial counts

        ntotal = num_rgap + num_sgap + num_pcount  # 三个参数的总个数
        self.cemb = Linear(ntotal, emb_dim, bias=False)
        print(f"num_sgap: {num_sgap}, num_rgap: {num_rgap}, num_pcount: {num_pcount}, ntotal: {ntotal}")
        # print(f"total: {ntotal}, self.cemb.weight: {self.cemb.weight.shape}")

    def forward(self, vt, rgap, sgap, pcount):
        # print("+"*100)
        # print(f"rgap.shape: {rgap.shape}")
        # print(f"rgap: {rgap[0]}")
        # print("-"*100)
        # print(f"sgap.shape: {sgap.shape}")
        # print(f"sgap: {sgap[0]}")
        # print("-"*100)
        # print(f"pcount.shape: {pcount.shape}")
        # print(f"pcount: {pcount[0]}")
        # print("+"*100)
        rgap, sgap, pcount = self.rgap_eye[rgap].to(device), self.sgap_eye[sgap].to(device), self.pcount_eye[pcount].to(device)
        # print(f"vt: {vt.shape}, rgap: {rgap.shape}, sgap: {sgap.shape}, pcount: {pcount.shape}")
        ct = torch.cat((rgap, sgap, pcount), -1) # bz * seq_len * num_fea
        # print(f"ct: {ct.shape}, self.cemb.weight: {self.cemb.weight.shape}")
        # element-wise mul
        Cct = self.cemb(ct) # bz * seq_len * emb
        # print(f"ct: {ct.shape}, Cct: {Cct.shape}")
        # 这里用了concatenation and multiplication  ------> θin(vt,ct) = [vt*Cct; ct]
        theta = torch.mul(vt, Cct)
        theta = torch.cat((theta, ct), -1)
        return theta
