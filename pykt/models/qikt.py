import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .que_base_model import QueBaseModel, QueEmb
from pykt.utils import debug_print
from sklearn import metrics
from torch.utils.data import DataLoader
from .loss import Loss
from scipy.special import softmax

class MLP(nn.Module):
    '''
    Classifier decoder implemented with MLP
    '''
    def __init__(self, n_layer, hidden_dim, output_dim, dpo):
        super().__init__()
        self.lins = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(n_layer)
        ])
        self.dropout = nn.Dropout(p=dpo)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        x = x.to(next(self.parameters()).device)
        for lin in self.lins:
            x = F.relu(lin(x))
        return self.out(self.dropout(x))

def get_outputs(self, emb_qc_shift, h, data, add_name="", model_type='question'):
    outputs = {}
    emb_qc_shift = emb_qc_shift.to(self.device)
    h = h.to(self.device)
    data = {k: v.to(self.device) for k, v in data.items()}

    if model_type == 'question':
        h_next = torch.cat([emb_qc_shift, h], axis=-1).to(self.device)
        y_question_next = torch.sigmoid(self.out_question_next(h_next))
        y_question_all = torch.sigmoid(self.out_question_all(h))
        outputs["y_question_next" + add_name] = y_question_next.squeeze(-1)
        outputs["y_question_all" + add_name] = (y_question_all * F.one_hot(data['qshft'].long(), self.num_q).to(self.device)).sum(-1)
    else: 
        h_next = torch.cat([emb_qc_shift, h], axis=-1).to(self.device)
        y_concept_next = torch.sigmoid(self.out_concept_next(h_next))
        # all predict
        y_concept_all = torch.sigmoid(self.out_concept_all(h))
        outputs["y_concept_next" + add_name] = self.get_avg_fusion_concepts(y_concept_next, data['cshft'])
        outputs["y_concept_all" + add_name] = self.get_avg_fusion_concepts(y_concept_all, data['cshft'])
    return outputs

class QIKTNet(nn.Module):
    def __init__(self, num_q, num_c, emb_size, dropout=0.1, emb_type='qaid', emb_path="", pretrain_dim=768, device='cpu', mlp_layer_num=1, other_config={}):
        super().__init__()
        self.model_name = "qikt"
        self.num_q = num_q
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.mlp_layer_num = mlp_layer_num
        self.device = device
        self.other_config = other_config
        self.output_mode = self.other_config.get('output_mode', 'an')

        self.emb_type = emb_type
        self.que_emb = QueEmb(num_q=num_q, num_c=num_c, emb_size=emb_size, emb_type=self.emb_type, model_name=self.model_name, device=device,
                             emb_path=emb_path, pretrain_dim=pretrain_dim)
       
        self.que_lstm_layer = nn.LSTM(self.emb_size * 4, self.hidden_size, batch_first=True).to(self.device)
        self.concept_lstm_layer = nn.LSTM(self.emb_size * 2, self.hidden_size, batch_first=True).to(self.device)
        self.dropout_layer = nn.Dropout(dropout).to(self.device)

        self.out_question_next = MLP(self.mlp_layer_num, self.hidden_size * 3, 1, dropout).to(self.device)
        self.out_question_all = MLP(self.mlp_layer_num, self.hidden_size, num_q, dropout).to(self.device)
        self.out_concept_next = MLP(self.mlp_layer_num, self.hidden_size * 3, num_c, dropout).to(self.device)
        self.out_concept_all = MLP(self.mlp_layer_num, self.hidden_size, num_c, dropout).to(self.device)
        self.que_disc = MLP(self.mlp_layer_num, self.hidden_size * 2, 1, dropout).to(self.device)

    def get_avg_fusion_concepts(self, y_concept, cshft):
        """获取知识点 fusion 的预测结果"""
        y_concept = y_concept.to(self.device)
        cshft = cshft.to(self.device)
        max_num_concept = cshft.shape[-1]
        concept_mask = torch.where(cshft.long() == -1, False, True).to(self.device)
        concept_index = F.one_hot(torch.where(cshft != -1, cshft, 0), self.num_c).to(self.device)
        concept_sum = (y_concept.unsqueeze(2).repeat(1, 1, max_num_concept, 1) * concept_index).sum(-1).to(self.device)
        concept_sum = concept_sum * concept_mask
        y_concept = concept_sum.sum(-1) / torch.where(concept_mask.sum(-1) != 0, concept_mask.sum(-1), 1).to(self.device)
        return y_concept

    def forward(self, q, c, r, data=None):
        q, c, r = q.to(self.device), c.to(self.device), r.to(self.device)
        
        _, emb_qca, emb_qc, emb_q, emb_c = self.que_emb(q, c, r)
        emb_qca, emb_qc, emb_q, emb_c = emb_qca.to(self.device), emb_qc.to(self.device), emb_q.to(self.device), emb_c.to(self.device)  #[batch_size,emb_size*4],[batch_size,emb_size*2],[batch_size,emb_size*1],[batch_size,emb_size*1] 

        emb_qc_shift = emb_qc[:, 1:, :].to(self.device)
        emb_qca_current = emb_qca[:, :-1, :].to(self.device)
        
        # question model
        que_h = self.dropout_layer(self.que_lstm_layer(emb_qca_current)[0].to(self.device))
        que_outputs = get_outputs(self, emb_qc_shift, que_h, data, add_name="", model_type="question")
        outputs = que_outputs
        
        # concept model
        emb_ca = torch.cat([emb_c.mul((1 - r).unsqueeze(-1).repeat(1, 1, self.emb_size)),
                            emb_c.mul(r.unsqueeze(-1).repeat(1, 1, self.emb_size))], dim=-1).to(self.device)  # s_t 扩展，分别对应正确的错误的情况
        emb_ca_current = emb_ca[:, :-1, :].to(self.device)
        # emb_c_shift = emb_c[:,1:,:]
        concept_h = self.dropout_layer(self.concept_lstm_layer(emb_ca_current)[0].to(self.device))
        concept_outputs = get_outputs(self, emb_qc_shift, concept_h, data, add_name="", model_type="concept")
        outputs['y_concept_all'] = concept_outputs['y_concept_all'].to(self.device)
        outputs['y_concept_next'] = concept_outputs['y_concept_next'].to(self.device)
        
        return outputs

class QIKT(QueBaseModel):
    def __init__(self, num_q, num_c, emb_size, dropout=0.1, emb_type='qaid', emb_path="", pretrain_dim=768, device='cpu', seed=0, mlp_layer_num=1, other_config={}, **kwargs):
        model_name = "qikt"
        debug_print(f"emb_type is {emb_type}", fuc_name="QIKT")
        super().__init__(model_name=model_name, emb_type=emb_type, emb_path=emb_path, pretrain_dim=pretrain_dim, device=device, seed=seed)
        self.model = QIKTNet(num_q=num_q, num_c=num_c, emb_size=emb_size, dropout=dropout, emb_type=emb_type,
                             emb_path=emb_path, pretrain_dim=pretrain_dim, device=device, mlp_layer_num=mlp_layer_num, other_config=other_config)
        self.model = self.model.to(device)
        self.emb_type = self.model.emb_type
        self.loss_func = self._get_loss_func("binary_crossentropy")
        self.eval_result = {}

    def train_one_step(self, data, process=True, return_all=False):
        outputs, data_new = self.predict_one_step(data, return_details=True, process=process)
        loss_q_all = self.get_loss(outputs['y_question_all'].to(self.device), data_new['rshft'].to(self.device), data_new['sm'].to(self.device))
        loss_c_all = self.get_loss(outputs['y_concept_all'].to(self.device), data_new['rshft'].to(self.device), data_new['sm'].to(self.device))
        loss_q_next = self.get_loss(outputs['y_question_next'].to(self.device), data_new['rshft'].to(self.device), data_new['sm'].to(self.device))  # question level loss
        loss_c_next = self.get_loss(outputs['y_concept_next'].to(self.device), data_new['rshft'].to(self.device), data_new['sm'].to(self.device))  # kc level loss
        loss_kt = self.get_loss(outputs['y'].to(self.device), data_new['rshft'].to(self.device), data_new['sm'].to(self.device))

        loss_q_all_lambda = self.model.other_config.get('loss_q_all_lambda', 1)
        loss_c_all_lambda = self.model.other_config.get('loss_c_all_lambda', 1)
        loss_q_next_lambda = self.model.other_config.get('loss_q_next_lambda', 1)
        loss_c_next_lambda = self.model.other_config.get('loss_c_next_lambda', 1)
        
        if self.model.output_mode == "an":
            loss = loss_q_all_lambda * loss_q_all + loss_c_all_lambda * loss_c_all + loss_q_next_lambda * loss_q_next + loss_c_next_lambda * loss_c_next
        else:
            loss = loss_kt

        outputs['loss'] = loss
        outputs['y'] = outputs['y'].to(self.device)
        
        if return_all:
            return outputs, data_new, loss
        else:
            return outputs['y'], loss  # y_question没用

    def predict(self, dataset, batch_size, return_ts=False, process=True):
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.model.eval()
        with torch.no_grad():
            y_trues = []
            y_pred_dict = {}
            for data in test_loader:
                new_data = self.batch_to_device(data, process=process)
                outputs, data_new = self.predict_one_step(data, return_details=True)
                for key in outputs:
                    if not key.startswith("y") or key in ['y_qc_predict']:
                        continue
                    elif key not in y_pred_dict:
                        y_pred_dict[key] = []
                    y = torch.masked_select(outputs[key].to(self.device), new_data['sm'].to(self.device)).detach().cpu()  # get label
                    y_pred_dict[key].append(y.numpy())
                t = torch.masked_select(new_data['rshft'].to(self.device), new_data['sm'].to(self.device)).detach().cpu()
                y_trues.append(t.numpy())

        results = y_pred_dict
        for key in results:
            results[key] = np.concatenate(results[key], axis=0)
        ts = np.concatenate(y_trues, axis=0)
        results['ts'] = ts
        return results

    def evaluate(self, dataset, batch_size, acc_threshold=0.5):
        results = self.predict(dataset, batch_size=batch_size)
        eval_result = {}
        ts = results["ts"]
        for key in results:
            if not key.startswith("y") or key in ['y_qc_predict']:
                pass
            else:
                ps = results[key]
                kt_auc = metrics.roc_auc_score(y_true=ts, y_score=ps)
                prelabels = [1 if p >= acc_threshold else 0 for p in ps]
                kt_acc = metrics.accuracy_score(ts, prelabels)
                if key != "y":
                    eval_result["{}_kt_auc".format(key)] = kt_auc
                    eval_result["{}_kt_acc".format(key)] = kt_acc
                else:
                    eval_result["auc"] = kt_auc
                    eval_result["acc"] = kt_acc
        
        self.eval_result = eval_result
        return eval_result

    def predict_one_step(self, data, return_details=False, process=True, return_raw=False):
        data_new = self.batch_to_device(data, process=process)
        outputs = self.model(data_new['cq'].long().to(self.device), data_new['cc'].to(self.device), data_new['cr'].long().to(self.device), data=data_new)
        output_c_all_lambda = self.model.other_config.get('output_c_all_lambda', 1)
        output_c_next_lambda = self.model.other_config.get('output_c_next_lambda', 1)
        output_q_all_lambda = self.model.other_config.get('output_q_all_lambda', 1)
        output_q_next_lambda = self.model.other_config.get('output_q_next_lambda', 0)  # not use this
        
        if self.model.output_mode == "an_irt":
            def sigmoid_inverse(x, epsilon=1e-8):
                return torch.log(x / (1 - x + epsilon) + epsilon)
            y = sigmoid_inverse(outputs['y_question_all']).to(self.device) * output_q_all_lambda + sigmoid_inverse(outputs['y_concept_all']).to(self.device) * output_c_all_lambda + sigmoid_inverse(outputs['y_concept_next']).to(self.device) * output_c_next_lambda
            y = torch.sigmoid(y).to(self.device)
        else:
            y = outputs['y_question_all'].to(self.device) * output_q_all_lambda + outputs['y_concept_all'].to(self.device) * output_c_all_lambda + outputs['y_concept_next'].to(self.device) * output_c_next_lambda
            y = y / (output_q_all_lambda + output_c_all_lambda + output_c_next_lambda)
        outputs['y'] = y.to(self.device)

        if return_details:
            return outputs, data_new
        else:
            return y
