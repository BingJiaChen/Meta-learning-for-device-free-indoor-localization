import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import *
from data import *
from utils import *
import copy

class Trainer():
    def __init__(self,args):
        self.args = args
        source_path = ['./dataset/EXP1_sub1.npy']
        target_path = ['./dataset/EXP1_sub2.npy']
        self.nway = 16
        self.nshot = 5
        self.tr_dataloader = MyDataLoader(source_path,self.nway,self.nshot)
        self.te_dataloader = MyDataLoader(target_path,self.nway,self.nshot)
        self.in_features = 32 + self.nway
        self.model = gnnModel(self.nway,args)
        self.iter = 5000
        self.sample_size = 32
        self.device = args.device

    def load_model(self,model_path):
        self.model.load_state_dict(torch.load(model_path))

    def load_pretrain(self,model_path):
        self.model.cnn_feature.load_state_dict(torch.load(model_path))
        self.model.cnn_feature.C[13] = Identity()

    def train_batch(self):
        self.model.train()
        data = self.tr_dataloader.load_train_batch(batch_size=16,nway=self.nway,num_shot=self.nshot)
        data = [(_data).to(self.device) for _data in data]
        self.opt.zero_grad()
        logsoft_prob, A = self.model(data)
        label = data[1]
        loss = F.nll_loss(logsoft_prob,label)
        pred = torch.argmax(logsoft_prob,dim=1)
        acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()

        loss.backward()
        self.opt.step()

        return loss.item(), acc

    def train(self):
        self.load_pretrain('./dataset/model_fine_tuning.ckpt')
        for p in self.model.cnn_feature.parameters():
            p.requires_grad = True
        self.model.to(self.device)
        self.opt = torch.optim.Adam(filter(lambda p:p.requires_grad,self.model.parameters()),lr=self.args.lr,weight_decay=1e-6)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt,T_max=100,eta_min=0.0001,last_epoch=-1)
        best_loss = 1e8
        best_acc = 0
        eval_sample = 6400
        train_loss = []
        train_acc = []
        stop = 0
        for i in range(self.iter):
            loss, acc = self.train_batch()
            train_loss.append(loss)
            train_acc.append(acc)
            if i%self.args.log_interval==0:
                print(f"[Train | {i}/{self.iter} ] loss = {np.mean(train_loss):.5f}, acc = {np.mean(train_acc)*100:.5f} %")
                train_loss = []
                train_acc = []
                
            # if (i+1)%self.args.log_eval==0:
                val_loss, val_acc = self.eval(self.te_dataloader,eval_sample)
                print(f"[Val | {i+1}/{self.args.n_iter} ] loss = {val_loss:.5f}, acc = {val_acc:.5f} %")

                if val_acc > best_acc:
                    stop = 0
                    best_loss = val_loss
                    best_acc = val_acc

                stop += 1 

            # self.scheduler.step()

        print(f"[Best Result | for {self.args.n_iter} iterations ] best loss = {best_loss:.5f}, best_acc = {best_acc:.5f} %")

    def fast_adaption(self,dataloader,fast_iter):
        train_loss = []
        train_acc = []
        self.model.train()
        self.model.to(self.device)
        opt = torch.optim.Adam(filter(lambda p:p.requires_grad,self.model.parameters()),lr=0.01,weight_decay=1e-6)
        for i in range(fast_iter):
            data = dataloader.load_fast_batch(batch_size=16,nway=self.nway,num_shot=self.nshot)
            data = [(_data).to(self.device) for _data in data]
            opt.zero_grad()
            logsoft_prob, A = self.model(data)
            label = data[1]
            loss = F.nll_loss(logsoft_prob,label)
            pred = torch.argmax(logsoft_prob,dim=1)
            # print(pred)
            acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()

            loss.backward()
            opt.step()

            train_loss.append(loss.item())
            train_acc.append(acc)

        print(f"[Fast Adaption ] loss = {np.mean(train_loss):.5f}, acc = {np.mean(train_acc)*100:.5f} %")


    def eval(self,dataloader,test_sample):
        
        self.fast_adaption(dataloader,5)

        self.model.eval()
        iteration = int(test_sample/self.args.batch_size)

        total_loss = 0.0
        total_sample = 0
        total_acc = 0
        with torch.no_grad():
            for i in range(iteration):
                data = dataloader.load_test_batch(self.args.batch_size,self.args.way,self.args.shot)
                data = [(_data).to(self.device) for _data in data]
                logsoft_prob, A = self.model(data)
                label = data[1]
                loss = F.nll_loss(logsoft_prob,label)
                total_loss += loss.item() * logsoft_prob.shape[0]
                pred = torch.argmax(logsoft_prob,dim=1)

                total_acc += torch.eq(pred,label).float().sum().item()
                total_sample += pred.shape[0]


        return total_loss/total_sample, total_acc/total_sample*100
        

