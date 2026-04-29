# Attention-based Feature-level Distillation 
# Copyright (c) 2021-present NAVER Corp.
# Apache License v2.0

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class nn_bn_relu(nn.Module):
    def __init__(self, nin, nout):
        super(nn_bn_relu, self).__init__()
        self.linear = nn.Linear(nin, nout).cuda()
        self.bn = nn.BatchNorm1d(nout).cuda()
        self.relu = nn.ReLU(inplace=False).cuda()

    def forward(self, x, relu=True):
        if relu:
            return self.relu(self.bn(self.linear(x.cuda())))
        return self.bn(self.linear(x.cuda()))


class AFD(nn.Module):
    def __init__(self, n_t,t_shapes,s_shapes,unique_t_shapes,guide_layers,hint_layers):
        super(AFD, self).__init__()
        self.n_t = n_t
        self.t_shapes=t_shapes
        self.s_shapes=s_shapes
        self.unique_t_shapes=unique_t_shapes
        self.guide_layers = guide_layers
        self.hint_layers = hint_layers
        self.attention = Attention(self.n_t,self.t_shapes,self.s_shapes,self.unique_t_shapes).cuda()

    def forward(self, g_s, g_t):
        g_t = [g_t[i] for i in self.guide_layers]
        g_s = [g_s[i] for i in self.hint_layers]
        loss = self.attention(g_s, g_t)
        return sum(loss)


class Attention(nn.Module):
    def __init__(self, n_t,t_shapes,s_shapes,unique_t_shapes):
        super(Attention, self).__init__()
        self.qk_dim = 128
        self.n_t = n_t
        self.t_shapes=t_shapes
        self.s_shapes=s_shapes
        self.unique_t_shapes=unique_t_shapes
        self.linear_trans_s = LinearTransformStudent(self.t_shapes,self.s_shapes,self.unique_t_shapes)
        self.linear_trans_t = LinearTransformTeacher(self.t_shapes)

        self.p_t = nn.Parameter(torch.Tensor(len(t_shapes), 128))
        self.p_s = nn.Parameter(torch.Tensor(len(s_shapes), 128))
        torch.nn.init.xavier_normal_(self.p_t)
        torch.nn.init.xavier_normal_(self.p_s)

    def forward(self, g_s, g_t):
        # 将 Student 和 Teacher 的中间层特征通过线性变换映射到同一空间，得到 Key 和 Query
        bilinear_key, h_hat_s_all = self.linear_trans_s(g_s)
        query, h_t_all = self.linear_trans_t(g_t)

        p_logit = torch.matmul(self.p_t, self.p_s.t())

        logit = torch.add(torch.einsum('bstq,btq->bts', bilinear_key, query), p_logit) / np.sqrt(self.qk_dim)
        atts = F.softmax(logit, dim=2)  # b x t x s
        loss = []

        for i, (n, h_t) in enumerate(zip(self.n_t, h_t_all)):
            # print("i:",i,"\n n:",n,"\n h_t.shape:",h_t.shape)
#             #！！！！！将特定层忽略i: 0 
#  n: 0 
#  h_t.shape: torch.Size([32, 1024])
# i: 1 
#  n: 0 
#  h_t.shape: torch.Size([32, 1024])
# i: 2 
#  n: 1 
#  h_t.shape: torch.Size([32, 256])
# i: 3 
#  n: 1 
#  h_t.shape: torch.Size([32, 256])
# i: 4 
#  n: 2 
#  h_t.shape: torch.Size([32, 64])
# i: 5 
#  n: 2 
#  h_t.shape: torch.Size([32, 64])
# i: 6 
#  n: 3 
#  h_t.shape: torch.Size([32, 16])
# i: 7 
#  n: 3 
#  h_t.shape: torch.Size([32, 16])

            if(i==2 or i==3):
                # print("jump")
                continue
            h_hat_s = h_hat_s_all[n]
            diff = self.cal_diff(h_hat_s, h_t, atts[:, i])
            loss.append(diff)
        # exit()
        return loss

    def cal_diff(self, v_s, v_t, att):
        diff = (v_s - v_t.unsqueeze(1)).pow(2).mean(2)
        diff = torch.mul(diff, att).sum(1).mean()
        return diff


class LinearTransformTeacher(nn.Module):
    def __init__(self, t_shapes):
        super(LinearTransformTeacher, self).__init__()
        self.t_shapes=t_shapes
        self.query_layer = nn.ModuleList([nn_bn_relu(t_shape[1], 128) for t_shape in self.t_shapes])

    def forward(self, g_t):
        bs = g_t[0].size(0)
        channel_mean = [f_t.mean(3).mean(2) for f_t in g_t]
        spatial_mean = [f_t.pow(2).mean(1).view(bs, -1) for f_t in g_t]
        query = torch.stack([query_layer(f_t, relu=False) for f_t, query_layer in zip(channel_mean, self.query_layer)],
                            dim=1)
        value = [F.normalize(f_s, dim=1) for f_s in spatial_mean]
        return query, value


class LinearTransformStudent(nn.Module):
    def __init__(self, t_shapes,s_shapes,unique_t_shapes):
        super(LinearTransformStudent, self).__init__()
        self.t = len(t_shapes)
        self.s = len(s_shapes)
        self.t_shapes=t_shapes
        self.s_shapes=s_shapes
        self.unique_t_shapes=unique_t_shapes
        self.qk_dim = 128
        self.relu = nn.ReLU(inplace=False)
        self.samplers = nn.ModuleList([Sample(self.t_shapes) for self.t_shapes in self.unique_t_shapes])

        self.key_layer = nn.ModuleList([nn_bn_relu(s_shape[1], 128) for s_shape in self.s_shapes])
        self.bilinear = nn_bn_relu(128, 128 * self.t)

    def forward(self, g_s):
        bs = g_s[0].size(0)
        channel_mean = [f_s.mean(3).mean(2) for f_s in g_s]
        spatial_mean = [sampler(g_s, bs) for sampler in self.samplers]

        key = torch.stack([key_layer(f_s) for key_layer, f_s in zip(self.key_layer, channel_mean)],
                                     dim=1).view(bs * self.s, -1) # Bs x h
        bilinear_key = self.bilinear(key, relu=False).view(bs, self.s, self.t, -1)
        value = [F.normalize(s_m, dim=2) for s_m in spatial_mean]
        return bilinear_key, value


class Sample(nn.Module):
    def __init__(self, t_shapes):
        super(Sample, self).__init__()
        t_N, t_C, t_H, t_W = t_shapes
        self.sample = nn.AdaptiveAvgPool2d((t_H, t_W))

    def forward(self, g_s, bs):
        g_s = torch.stack([self.sample(f_s.pow(2).mean(1, keepdim=True)).view(bs, -1) for f_s in g_s], dim=1)
        return g_s
