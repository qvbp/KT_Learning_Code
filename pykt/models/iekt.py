import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .que_base_model import QueBaseModel, QueEmb
from torch.distributions import Categorical
from .iekt_utils import mygru, funcs
from pykt.utils import debug_print

class IEKTNet(nn.Module): 
    def __init__(self, num_q, num_c, emb_size, max_concepts, lamb=40, n_layer=1, cog_levels=10, acq_levels=10, dropout=0, gamma=0.93, emb_type='qc_merge', emb_path="", pretrain_dim=768, device='cpu'):
        super().__init__()
        self.model_name = "iekt"
        self.emb_size = emb_size
        self.concept_num = num_c
        self.max_concept = max_concepts
        self.device = device
        self.emb_type = emb_type
        self.predictor = funcs(n_layer, emb_size * 5, 1, dropout).to(device)
        self.cog_matrix = nn.Parameter(torch.randn(cog_levels, emb_size * 2).to(self.device), requires_grad=True) 
        self.acq_matrix = nn.Parameter(torch.randn(acq_levels, emb_size * 2).to(self.device), requires_grad=True)
        self.select_preemb = funcs(n_layer, emb_size * 3, cog_levels, dropout).to(device)  # MLP
        self.checker_emb = funcs(n_layer, emb_size * 12, acq_levels, dropout).to(device)
        self.prob_emb = nn.Parameter(torch.randn(num_q, emb_size).to(self.device), requires_grad=True)  # Question representation
        self.gamma = gamma
        self.lamb = lamb
        self.gru_h = mygru(0, emb_size * 4, emb_size).to(device)
        self.concept_emb = nn.Parameter(torch.randn(self.concept_num, emb_size).to(self.device), requires_grad=True)  # Concept representation
        self.sigmoid = torch.nn.Sigmoid().to(device)
        self.que_emb = QueEmb(num_q=num_q, num_c=num_c, emb_size=emb_size, emb_type=self.emb_type, model_name=self.model_name, device=device,
                             emb_path=emb_path, pretrain_dim=pretrain_dim).to(device)

    def get_ques_representation(self, q, c):
        """Get question representation equation 3

        Args:
            q (_type_): question ids
            c (_type_): concept ids

        Returns:
            _type_: _description_
        """
        v = self.que_emb(q, c).to(self.device)
        return v

    def pi_cog_func(self, x, softmax_dim=1):
        return F.softmax(self.select_preemb(x.to(self.device)), dim=softmax_dim)

    def obtain_v(self, q, c, h, x, emb):
        """_summary_

        Args:
            q (_type_): _description_
            c (_type_): _description_
            h (_type_): _description_
            x (_type_): _description_
            emb (_type_): m_t

        Returns:
            _type_: _description_
        """
        v = self.get_ques_representation(q.to(self.device), c.to(self.device))
        predict_x = torch.cat([h.to(self.device), v], dim=1)  # equation 4
        h_v = torch.cat([h.to(self.device), v], dim=1)  # equation 4
        prob = self.predictor(torch.cat([predict_x, emb.to(self.device)], dim=1))  # equation 7
        return h_v, v, prob, x

    def update_state(self, h, v, emb, operate):
        """_summary_

        Args:
            h (_type_): rnn的h
            v (_type_): question 表示
            emb (_type_): s_t knowledge acquistion sensitivity
            operate (_type_): label

        Returns:
            next_p_state {}: _description_
        """
        v_cat = torch.cat([
            v.mul(operate.repeat(1, self.emb_size * 2).to(self.device)),
            v.mul((1 - operate).repeat(1, self.emb_size * 2).to(self.device))], dim=1)  # v_t 扩展，分别对应正确的错误的情况
        e_cat = torch.cat([
            emb.mul((1-operate).repeat(1, self.emb_size * 2).to(self.device)),
            emb.mul((operate).repeat(1, self.emb_size * 2).to(self.device))], dim=1)  # s_t 扩展，分别对应正确的错误的情况
        inputs = v_cat + e_cat  # 起到concat作用
        
        h_t_next = self.gru_h(inputs.to(self.device), h.to(self.device))  # equation 14
        return h_t_next

    def pi_sens_func(self, x, softmax_dim=1):
        return F.softmax(self.checker_emb(x.to(self.device)), dim=softmax_dim)


class IEKT(QueBaseModel):
    def __init__(self, num_q, num_c, emb_size, max_concepts, lamb=40, n_layer=1, cog_levels=10, acq_levels=10, dropout=0, gamma=0.93, emb_type='qid', emb_path="", pretrain_dim=768, device='cpu', seed=0):
        model_name = "iekt"
        super().__init__(model_name=model_name, emb_type=emb_type, emb_path=emb_path, pretrain_dim=pretrain_dim, device=device, seed=seed)

        self.model = IEKTNet(num_q=num_q, num_c=num_c, lamb=lamb, emb_size=emb_size, max_concepts=max_concepts, n_layer=n_layer, cog_levels=cog_levels, acq_levels=acq_levels, dropout=dropout, gamma=gamma, emb_type=emb_type, emb_path=emb_path, pretrain_dim=pretrain_dim, device=device)

        self.model = self.model.to(device)
    
    def train_one_step(self, data, process=True):
        BCELoss = torch.nn.BCEWithLogitsLoss().to(self.device)
        
        data_new, emb_action_list, p_action_list, states_list, pre_state_list, reward_list, predict_list, ground_truth_list = self.predict_one_step(data, return_details=True, process=process)
        data_len = data_new['cc'].shape[0]
        seq_len = data_new['cc'].shape[1]

        seq_num = torch.where(data['qseqs'].to(self.device) != 0, 1, 0).sum(axis=-1) + 1
        emb_action_tensor = torch.stack(emb_action_list, dim=1).to(self.device)
        p_action_tensor = torch.stack(p_action_list, dim=1).to(self.device)
        state_tensor = torch.stack(states_list, dim=1).to(self.device)
        pre_state_tensor = torch.stack(pre_state_list, dim=1).to(self.device)
        reward_tensor = torch.stack(reward_list, dim=1).float().to(self.device) / (seq_num.unsqueeze(-1).repeat(1, seq_len)).float().to(self.device)  # equation 15
        logits_tensor = torch.stack(predict_list, dim=1).to(self.device)
        ground_truth_tensor = torch.stack(ground_truth_list, dim=1).to(self.device)
        loss = []
        tracat_logits = []
        tracat_ground_truth = []
        
        for i in range(0, data_len):
            this_seq_len = seq_num[i]
            this_reward_list = reward_tensor[i]
            this_cog_state = torch.cat([pre_state_tensor[i][0:this_seq_len],
                                    torch.zeros(1, pre_state_tensor[i][0].size()[0]).to(self.device)
                                    ], dim=0).to(self.device)
            this_sens_state = torch.cat([state_tensor[i][0:this_seq_len],
                                    torch.zeros(1, state_tensor[i][0].size()[0]).to(self.device)
                                    ], dim=0).to(self.device)

            td_target_cog = this_reward_list[0:this_seq_len].unsqueeze(1).to(self.device)
            delta_cog = td_target_cog
            delta_cog = delta_cog.detach().cpu().numpy()

            td_target_sens = this_reward_list[0:this_seq_len].unsqueeze(1).to(self.device)
            delta_sens = td_target_sens
            delta_sens = delta_sens.detach().cpu().numpy()

            advantage_lst_cog = []
            advantage = 0.0
            for delta_t in delta_cog[::-1]:
                advantage = self.model.gamma * advantage + delta_t[0]  # equation 17
                advantage_lst_cog.append([advantage])
            advantage_lst_cog.reverse()
            advantage_cog = torch.tensor(advantage_lst_cog, dtype=torch.float).to(self.device)
            
            pi_cog = self.model.pi_cog_func(this_cog_state[:-1].to(self.device))
            pi_a_cog = pi_cog.gather(1, p_action_tensor[i][0:this_seq_len].unsqueeze(1).to(self.device))

            loss_cog = -torch.log(pi_a_cog) * advantage_cog  # equation 16
            
            loss.append(torch.sum(loss_cog).to(self.device))

            advantage_lst_sens = []
            advantage = 0.0
            for delta_t in delta_sens[::-1]:
                advantage = self.model.gamma * advantage + delta_t[0]
                advantage_lst_sens.append([advantage])
            advantage_lst_sens.reverse()
            advantage_sens = torch.tensor(advantage_lst_sens, dtype=torch.float).to(self.device)
            
            pi_sens = self.model.pi_sens_func(this_sens_state[:-1].to(self.device))
            pi_a_sens = pi_sens.gather(1, emb_action_tensor[i][0:this_seq_len].unsqueeze(1).to(self.device))

            loss_sens = -torch.log(pi_a_sens) * advantage_sens  # equation 18
            loss.append(torch.sum(loss_sens).to(self.device))
            
            this_prob = logits_tensor[i][0:this_seq_len].to(self.device)
            this_ground_truth = ground_truth_tensor[i][0:this_seq_len].to(self.device)

            tracat_logits.append(this_prob)
            tracat_ground_truth.append(this_ground_truth)

        bce = BCELoss(torch.cat(tracat_logits, dim=0).to(self.device), torch.cat(tracat_ground_truth, dim=0).to(self.device))   
        y = torch.cat(tracat_logits, dim=0).to(self.device)
        label_len = torch.cat(tracat_ground_truth, dim=0).size()[0]
        loss_l = sum(loss)
        loss = self.model.lamb * (loss_l / label_len).to(self.device) + bce  # equation 21
        return y, loss

    def predict_one_step(self, data, return_details=False, process=True):
        sigmoid_func = torch.nn.Sigmoid().to(self.device)
        data_new = self.batch_to_device(data, process)
        
        data_len = data_new['cc'].shape[0]
        seq_len = data_new['cc'].shape[1]
        h = torch.zeros(data_len, self.model.emb_size).to(self.device)
        batch_probs, uni_prob_list, actual_label_list, states_list, reward_list = [], [], [], [], []
        p_action_list, pre_state_list, emb_action_list, op_action_list, actual_label_list, predict_list, ground_truth_list = [], [], [], [], [], [], []

        rt_x = torch.zeros(data_len, 1, self.model.emb_size * 2).to(self.device)
        for seqi in range(0, seq_len):  # 序列长度
            ques_h = torch.cat([
                self.model.get_ques_representation(q=data_new['cq'][:, seqi].to(self.device), c=data_new['cc'][:, seqi].to(self.device)),
                h], dim=1).to(self.device)  # equation 4

            flip_prob_emb = self.model.pi_cog_func(ques_h.to(self.device))

            m = Categorical(flip_prob_emb)  # equation 5 的 f_p
            emb_ap = m.sample()  # equation 5
            emb_p = self.model.cog_matrix[emb_ap, :].to(self.device)  # equation 6

            h_v, v, logits, rt_x = self.model.obtain_v(q=data_new['cq'][:, seqi].to(self.device), c=data_new['cc'][:, seqi].to(self.device), 
                                                        h=h.to(self.device), x=rt_x.to(self.device), emb=emb_p.to(self.device))  # equation 7
            prob = sigmoid_func(logits).to(self.device)  # equation 7 sigmoid

            out_operate_groundtruth = data_new['cr'][:, seqi].unsqueeze(-1).to(self.device)  # 获取标签
            
            out_x_groundtruth = torch.cat([
                h_v.mul(out_operate_groundtruth.repeat(1, h_v.size()[-1]).float().to(self.device)),
                h_v.mul((1-out_operate_groundtruth).repeat(1, h_v.size()[-1]).float().to(self.device))],
                dim=1).to(self.device)  # equation 9

            out_operate_logits = torch.where(prob > 0.5, torch.tensor(1).to(self.device), torch.tensor(0).to(self.device)) 
            out_x_logits = torch.cat([
                h_v.mul(out_operate_logits.repeat(1, h_v.size()[-1]).float().to(self.device)),
                h_v.mul((1-out_operate_logits).repeat(1, h_v.size()[-1]).float().to(self.device))],
                dim=1).to(self.device)  # equation 10                
            out_x = torch.cat([out_x_groundtruth.to(self.device), out_x_logits.to(self.device)], dim=1).to(self.device)  # equation 11

            ground_truth = data_new['cr'][:, seqi].to(self.device)
            flip_prob_emb = self.model.pi_sens_func(out_x.to(self.device))  # equation 12 中的 f_e

            m = Categorical(flip_prob_emb)
            emb_a = m.sample().to(self.device)
            emb = self.model.acq_matrix[emb_a, :].to(self.device)  # equation 12 s_t
            
            h = self.model.update_state(h.to(self.device), v.to(self.device), emb.to(self.device), ground_truth.unsqueeze(1).to(self.device))  # equation 13-14
           
            uni_prob_list.append(prob.detach().to(self.device))
            
            emb_action_list.append(emb_a)  # s_t 列表
            p_action_list.append(emb_ap)  # m_t
            states_list.append(out_x.to(self.device))
            pre_state_list.append(ques_h.to(self.device))  # 上一个题目的状态
            
            ground_truth_list.append(ground_truth.to(self.device))
            predict_list.append(logits.squeeze(1).to(self.device))
            this_reward = torch.where(out_operate_logits.squeeze(1).float().to(self.device) == ground_truth.to(self.device),
                            torch.tensor(1).to(self.device), 
                            torch.tensor(0).to(self.device))  # if condition x else y,这里相当于统计了正确的数量
            reward_list.append(this_reward.to(self.device))
        prob_tensor = torch.cat(uni_prob_list, dim=1).to(self.device)
        if return_details:
            return data_new, emb_action_list, p_action_list, states_list, pre_state_list, reward_list, predict_list, ground_truth_list
        else:
            return prob_tensor[:, 1:]
