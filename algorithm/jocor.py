# -*- coding:utf-8 -*-
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from common.utils import accuracy

from algorithm.loss import loss_jocor


class JoCoR:
    def __init__(self, args, train_dataset, device, input_channel, num_classes):

        # Hyper Parameters
        self.batch_size = 64
        learning_rate = args.lr

        if args.forget_rate is None:
            forget_rate = args.noise_rate
        else:
            forget_rate = args.forget_rate

        self.noise_or_not = train_dataset.noise_or_not

        # Adjust learning rate and betas for Adam Optimizer
        # mom1 = 0.9
        # mom2 = 0.1
        # self.alpha_plan = [learning_rate] * args.n_epoch
        # self.beta1_plan = [mom1] * args.n_epoch

        # for i in range(args.epoch_decay_start, args.n_epoch):
        #     self.alpha_plan[i] = float(args.n_epoch - i) / (args.n_epoch - args.epoch_decay_start) * learning_rate
        #     self.beta1_plan[i] = mom2

        # define drop rate schedule
        self.rate_schedule = np.ones(args.n_epoch) * forget_rate
        self.rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate ** args.exponent, args.num_gradual)

        self.device = device
        # self.num_iter_per_epoch = args.num_iter_per_epoch
        self.print_freq = args.print_freq
        self.co_lambda = args.co_lambda
        self.n_epoch = args.n_epoch
        self.train_dataset = train_dataset

        if args.model_type == "R":
            from model.resnet50 import resnet50
            self.model1 = resnet50(pretrained=True,emd_size=128, num_classes=num_classes)
            self.model2 = resnet50(pretrained=True,emd_size=128, num_classes=num_classes)
        elif args.model_type == "B":
            from model.bninception import bninception
            self.model1 = bninception(pretrained=True,emd_size=512, num_classes=num_classes)
            self.model2 = bninception(pretrained=True,emd_size=512, num_classes=num_classes)


        self.model1.to(device)
        # print(self.model1.parameters)

        self.model2.to(device)
        # print(self.model2.parameters)

        self.optimizer = torch.optim.Adam(list(self.model1.parameters()) + list(self.model2.parameters()),
                                          lr=learning_rate)

        self.loss_fn = loss_jocor

        from torch.optim.lr_scheduler import CosineAnnealingLR
        self.scheduler = CosineAnnealingLR(self.optimizer,T_max=args.n_epoch)

    # Evaluate the Model
    def evaluate(self, test_loader):
        print('Evaluating ...')
        self.model1.eval()  # Change model to 'eval' mode.
        self.model2.eval()  # Change model to 'eval' mode

        labels = test_loader.dataset.label_list
        labels = np.array([int(k) for k in labels])
        # print(labels.shape)
        
        from common.utils import retrieval_metric
        from ret_benchmark.utils.feat_extractor import feat_extractor
        feats1 = feat_extractor(self.model1, test_loader, logger=None)
        acc1,map1=retrieval_metric(feats1,labels[:len(feats1)])
        
        feats2 = feat_extractor(self.model2, test_loader, logger=None)
        acc2,map2=retrieval_metric(feats2,labels[:len(feats1)])
    
        return acc1,map1, acc2,map2
    # Train the Model
    def train(self, train_loader, epoch):
        print('Training ...')
        self.model1.train()  # Change model to 'train' mode.
        self.model2.train()  # Change model to 'train' mode


        train_total = 0
        train_correct = 0
        train_total2 = 0
        train_correct2 = 0
        pure_ratio_1_list = []
        pure_ratio_2_list = []

        for i, (images, labels, indexes) in enumerate(train_loader):
            ind = indexes.cpu().numpy().transpose()

            images = Variable(images).to(self.device)
            labels = Variable(labels).to(self.device)

            # Forward + Backward + Optimize
            logits1 = self.model1(images)
            prec1 = accuracy(logits1, labels, topk=(1,))
            train_total += 1
            train_correct += prec1

            logits2 = self.model2(images)
            prec2 = accuracy(logits2, labels, topk=(1,))
            train_total2 += 1
            train_correct2 += prec2

            loss_1, loss_2, pure_ratio_1, pure_ratio_2 = self.loss_fn(logits1, logits2, labels, self.rate_schedule[epoch],
                                                                 ind, self.noise_or_not, self.co_lambda)

            self.optimizer.zero_grad()
            loss_1.backward()
            self.optimizer.step()

            pure_ratio_1_list.append(100 * pure_ratio_1)
            pure_ratio_2_list.append(100 * pure_ratio_2)

            if (i + 1) % self.print_freq == 0:
                print(
                    'Epoch [%d/%d], Iter [%d/%d] Training Accuracy1: %.4F %%, Training Accuracy2: %.4f %%, Loss1: %.4f, Loss2: %.4f, Pure Ratio1 %.4f %% Pure Ratio2 %.4f %%'
                    % (epoch + 1, self.n_epoch, i + 1, len(self.train_dataset) // self.batch_size, prec1, prec2,
                       loss_1.data.item(), loss_2.data.item(), sum(pure_ratio_1_list) / len(pure_ratio_1_list), sum(pure_ratio_2_list) / len(pure_ratio_2_list)))

        train_acc1 = float(train_correct) / float(train_total)
        train_acc2 = float(train_correct2) / float(train_total2)

        self.scheduler.step()
        return train_acc1, train_acc2, pure_ratio_1_list, pure_ratio_2_list
