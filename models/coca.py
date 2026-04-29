# Copyright 2021-present, Zhong Ji, Jin Li, Qiang Wang, Zhongfei Zhang.
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
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from utils.args import *
import numpy as np
import torchvision.transforms as transforms
from utils.scloss import SupConLoss



def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via complementary calibration.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True, help='Penalty weight.')
    parser.add_argument('--beta', type=float, required=True, help='Penalty weight.')
    return parser

def rotate_img(img, s):

    transform = transforms.RandomResizedCrop(size=(32, 32), scale=(0.66, 0.67), ratio = (0.99,1.00))
    img = transform(img)
    return torch.rot90(img, s, [-1, -2])

class Coca(ContinualModel):
    NAME = 'coca'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Coca, self).__init__(backbone, loss, args, transform)

        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.criterion = SupConLoss()

    def observe(self, inputs, labels, not_aug_inputs):
        #start = time.time()
        self.opt.zero_grad()
        real_batch_size = inputs.shape[0]
        outputs, _, bat_inv, _ = self.net(inputs, return_features = True)

        loss = self.loss(outputs, labels)

        if not self.buffer.is_empty():

            buf_inputs, _, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs , _, inv_feat,feat = self.net(buf_inputs, return_features = True)
            
            omega = 0.1
            diaN = torch.eye(buf_inputs.shape[0], device=self.device)
            # row-wise normalization of affine matrix with zero-diagonal
            A = F.softmax(torch.mm(inv_feat, inv_feat.transpose(0, 1))- diaN*1.0  , dim = 1)
            # for numerical stability
            #logits_max, _ = torch.max(A, dim=1, keepdim=True)

            T = 2.0
            # approximate inference for propagation and ensembling  *(T**2)
            #soft_targets = torch.mm((1 - omega) * torch.inverse(diaN - omega * A), F.softmax(buf_outputs/T, dim=1))
            soft_targets = torch.mm((1 - omega) * torch.inverse(diaN - omega * A), buf_outputs)
            soft_targets = soft_targets.detach()

            gamma = 0.01
            #loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)
            loss += self.args.alpha * F.mse_loss(buf_outputs, (1 - gamma) *buf_logits + gamma * soft_targets)

            """"""
            """"""
            #loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)
            buf_inputs, buf_labels, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs, _, buf_inv, _  = self.net(buf_inputs, return_features = True)
            loss += self.args.beta * self.loss(buf_outputs, buf_labels)

            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))
            bat_inv = torch.cat((bat_inv, buf_inv))

        label_shot = torch.arange(4).repeat(inputs.shape[0])
        label_shot = label_shot.type(torch.LongTensor)
        choice = np.random.choice(a=inputs.shape[0] , size=inputs.shape[0], replace=False)
        rot_label = label_shot[choice].to(self.device)
        rot_inputs = inputs.cpu()
        for i in range(0, inputs.shape[0]):
            rot_inputs[i] = rotate_img(rot_inputs[i],rot_label[i])
        rot_inputs = rot_inputs.to(self.device)
        _ , rot_outputs, t_inv, _= self.net(rot_inputs, return_features = True)
        loss += 0.3 * self.criterion(bat_inv,t_inv, labels)
        loss += 0.3 * self.loss(rot_outputs, rot_label)


        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:real_batch_size],
                             logits=outputs.data)

        return loss.item()
