# Copyright 2022-present, Jin Li, Zhong Ji, Qiang Wang.
"""
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
"""
import torch
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from copy import deepcopy
from torch.nn import functional as F
import numpy as np
import torchvision.transforms as transforms
from utils.scloss import SupConLoss
from utils.afd import AFD
from utils.crd import CRDLoss
import argparse

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via online contrastive distillation')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--ER_weight', type=float, default=1.0)
    parser.add_argument('--Bernoulli_probability', type=float, default=0.70)
    parser.add_argument('--model_t', default='resnet18', type=str)
    parser.add_argument('--qk_dim', default=128, type=int)
    args = parser.parse_args()
    return parser

LAYER = {'resnet20': np.arange(1, (20 - 2) // 2 + 1),  # 9
         'resnet56': np.arange(1, (56 - 2) // 2 + 1),  # 27
         'resnet110': np.arange(2, (110 - 2) // 2 + 1, 2),  # 27
         'resnet18': np.arange(1, (18 - 2) // 2 + 1),  # 8
         }

def rotate_img(img, s):
    transform = transforms.RandomResizedCrop(size=(32, 32), scale=(0.66, 0.67), ratio = (0.99,1.00))
    #img = transform(img)
    return torch.rot90(img, s, [-1, -2])

def unique_shape(s_shapes):
    n_s = []
    unique_shapes = []
    n = -1
    for s_shape in s_shapes:
        if s_shape not in unique_shapes:
            unique_shapes.append(s_shape)
            n += 1
        n_s.append(n)
    return n_s, unique_shapes

class OCDNet1(ContinualModel):
    NAME = 'OCDNet1'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']
    def __init__(self, backbone, loss, args, transform):
        super(OCDNet1, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        # Initialize the teacher model
        self.teacher_model = deepcopy(self.net).to(self.device)
        # Set parameters for experience replay
        self.ER_weight = args.ER_weight
        # Set Bernoulli_probability
        self.Bernoulli_probability = args.Bernoulli_probability
        # Set parameters for the CRD-Module
        self.criterion = SupConLoss(temperature= 0.07)
        self.crd = CRDLoss(temperature= 0.07)
        self.model_iterations = 0
        self.model_t =args.model_t
        #self.criterion_kd = AFD(args1)
        #self.args.guide_layers =args.guide_layers 
        #self.afd = AFD(args)

       

    def observe(self, inputs, labels, not_aug_inputs):

        self.opt.zero_grad()
        _,outputs, _, bat_inv, _= self.net(inputs, return_features=True)
        loss = self.loss(outputs, labels)

        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
            # 1. 获取 Teacher (t_t) 和 Student (t_s) 的所有中间层特征
            t_t,teacher_logits, _, tea_inv, feat= self.teacher_model(buf_inputs, return_features=True)
            t_s,student_outputs, _, buf_inv, _ = self.net(buf_inputs, return_features=True)
            #t_t = [f.detach() for f in t_t]
            #t_t=t_t.to(device)

            # 2. 定义层级映射关系 (LAYER 字典定义了具体的层索引)
            guide_layers = LAYER[self.model_t]  # Teacher 的层索引
            hint_layers = LAYER[self.model_t]   # Student 的层索引
            
            t_shapes = [t_t[i].size() for i in guide_layers]# Teacher每层的size
            s_shapes = [t_s[i].size() for i in hint_layers] # Student每层的size

            n_t, unique_t_shapes = unique_shape(t_shapes) # n_t: [0, 0, 1, 1, 2, 2, 3, 3]
            # print("t_shapes:",t_shapes,"\ntype(n_t):",type(n_t),"\nn_t:",n_t)
            # exit()
            # 3. 初始化 AFD (Attention-based Feature Distillation) 模块
            # 这个模块负责计算层级间的蒸馏损失
            criterion_kd = AFD(n_t , t_shapes , s_shapes , unique_t_shapes , guide_layers , hint_layers)
            # print("criterion_kd:",criterion_kd,"\ntype(criterion_kd):",type(criterion_kd))
            # exit()
            
            omega = 0.1
            #diaN: NxN 的单位矩阵
            diaN = torch.eye(buf_inputs.shape[0], device=self.device)#在当前计算设备上创建一个NxN的单位矩阵，N=buf_inputs.shape[0]即样本数
            # row-wise normalization of affine matrix with zero-diagonal
            #torch.mm(feat, feat.transpose(0, 1)): 计算特征矩阵 feat 及其转置的矩阵乘法。这实际上是在计算 Batch 内所有样本两两之间的内积（相似度）。结果是一个 N×N 的矩阵。
            # -diaN * 1.0: 减去单位矩阵。目的是把对角线上的元素（即样本与自身的相似度）减小（使其变负，因为是内积）。这样做的目的是消除自循环（Feature of self），强迫模型去聚合周围邻居的信息，而不是仅仅关注自己。
            #F.softmax(..., dim=1): 对每一行进行 Softmax 归一化。这样矩阵 A 的每一行之和为 1，变成了概率转移矩阵。A[i][j] 表示从样本 i 转移（或关注）到样本 j 的概率。
            A = F.softmax(torch.mm(feat, feat.transpose(0, 1))- diaN*1.0  , dim = 1)
            T = 1.0 #温度参数
            soft_targets = torch.mm((1 - omega) * torch.inverse(diaN - omega * A), teacher_logits)

            with torch.no_grad(): #上下文管理器: 接下来的计算不需要计算梯度（可以节省显存，防止梯度回传到 Teacher 模型）
                teacher_model_prob = F.softmax(teacher_logits, 1) #归一化
                label_mask = F.one_hot(buf_labels, num_classes=teacher_logits.shape[-1]) > 0
                adaptive_weight = teacher_model_prob[label_mask]
            squared_losses = adaptive_weight * torch.mean((student_outputs - soft_targets.detach()) ** 2 , dim=1)

            loss += 0.1 *  squared_losses.mean()
            #loss += 0.1 * self.crd(buf_inv, tea_inv.detach(), buf_labels)
            loss += self.ER_weight * self.loss(student_outputs, buf_labels)
            loss += self.ER_weight * criterion_kd(t_t, t_s)
            #exit()

            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))
            bat_inv = torch.cat((bat_inv, buf_inv))

        """Collaborative Contrastive Learning https://arxiv.org/pdf/2109.02426.pdf """
        label_shot = torch.arange(0, 4).repeat(inputs.shape[0])
        label_shot = label_shot.type(torch.LongTensor)
        choice = np.random.choice(a=inputs.shape[0], size=inputs.shape[0], replace=False)
        rot_label = label_shot[choice].to(self.device)
        rot_inputs = inputs.cpu()
        for i in range(0, inputs.shape[0]):
            rot_inputs[i] = rotate_img(rot_inputs[i], rot_label[i])
        rot_inputs = rot_inputs.to(self.device)
        _,_, rot_outputs, t_inv, _ = self.net(rot_inputs, return_features=True)
        loss += 0.3 * self.criterion(bat_inv, t_inv, labels)
        loss += 0.3 * self.loss(rot_outputs, rot_label)

        loss.backward()
        self.opt.step()
        self.model_iterations += 1
        self.buffer.add_data(examples=not_aug_inputs, labels=labels[:not_aug_inputs.shape[0]])

        #Updating the teacher model
        if torch.rand(1) < self.Bernoulli_probability:
            # Momentum coefficient m
            m = min(1 - 1 / (self.model_iterations + 1), 0.999)
            for teacher_param, param in zip(self.teacher_model.parameters(), self.net.parameters()):
                teacher_param.data.mul_(m).add_(alpha=1 - m, other=param.data)
        return loss.item()

