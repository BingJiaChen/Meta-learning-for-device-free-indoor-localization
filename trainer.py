import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import *
from data import *
from utils import *
import copy

class Client():
    def __init__(self,client_id,model,dataloader,device,nway,nshot):
        self.client_id = client_id
        self.dataloader = dataloader
        self.device = device
        self.model = model
        self.nway = nway
        self.nshot = nshot
        self.model.cnn_feature.C[13] = Identity()

    def download_model(self,model):
        self.model.load_state_dict(model)

    def load_pretrain(self,model_path):
        self.model.cnn_feature.load_state_dict(torch.load(model_path))

    def get_num_data(self):
        return self.dataloader.get_num()

    def client_update(self,local_iter):
        for p in self.model.cnn_feature.parameters():
            p.requires_grad = True
        self.model.train()
        self.model.to(self.device)
        opt = torch.optim.Adam(filter(lambda p:p.requires_grad,self.model.parameters()),lr=0.01,weight_decay=1e-6)
        train_loss = []
        train_acc = []
        for i in range(local_iter):
            data = self.dataloader.load_train_batch(batch_size=16,nway=self.nway,num_shot=self.nshot)
            data = [(_data).to(self.device) for _data in data]
            opt.zero_grad()
            logsoft_prob, A = self.model(data)
            label = data[1]
            loss = F.nll_loss(logsoft_prob,label)
            pred = torch.argmax(logsoft_prob,dim=1)
            
            acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()

            loss.backward()
            opt.step()

            train_loss.append(loss.item())
            train_acc.append(acc)

        print(f"[Train | client : {self.client_id} ] loss = {np.mean(train_loss):.5f}, acc = {np.mean(train_acc)*100:.5f} %")

        return self.model.state_dict()


class CenterServer():
    def __init__(self,model,dataloader,device,nway,nshot):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.nway = nway
        self.nshot = nshot

    def load_pretrain(self,model_path):
        self.model.cnn_feature.load_state_dict(torch.load(model_path))
        self.model.cnn_feature.C[13] = Identity()
        self.center_model = self.model.state_dict()

    def send_model(self):
        return copy.deepcopy(self.center_model)

    def fast_adaption(self,fast_iter):
        train_loss = []
        train_acc = []
        self.model.train()
        self.model.to(self.device)
        opt = torch.optim.Adam(filter(lambda p:p.requires_grad,self.model.parameters()),lr=0.01,weight_decay=1e-6)
        for i in range(fast_iter):
            data = self.dataloader.load_fast_batch(batch_size=16,nway=self.nway,num_shot=self.nshot)
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

    def aggregation(self,clients):
        update_state = dict()
        total = 0

        for k, client in enumerate(clients):
            # client.load_pretrain('./dataset/model_fine_tuning.ckpt')
            client.download_model(self.send_model())
            local_state = client.client_update(50)
            weights = client.get_num_data()
            total += weights
            if k == 0:
                for key in self.model.state_dict().keys():
                    update_state[key] = local_state[key] * weights
            else:
                for key in self.model.state_dict().keys():
                    update_state[key] += local_state[key] * weights
            # print(weights,total)
            
        for var in update_state.keys():
            update_state[var] = update_state[var] / total

        self.model.load_state_dict(update_state)
        self.center_model = copy.deepcopy(self.model.state_dict())

    def validation(self):
        iter = int(3000/16)
        
        self.model.to(self.device)
        self.model.eval()
        total_loss = 0
        total_sample = 0
        total_acc = 0

        with torch.no_grad():
            for i in range(iter):
                data = self.dataloader.load_test_batch(16,self.nway,self.nshot)
                data = [(_data).to(self.device) for _data in data]
                logsoft_prob, A = self.model(data)
                label = data[1]
                loss = F.nll_loss(logsoft_prob,label)
                total_loss += loss.item() * logsoft_prob.shape[0]
                pred = torch.argmax(logsoft_prob,dim=1)

                total_acc += torch.eq(pred,label).float().sum().item()
                total_sample += pred.shape[0]

        return total_loss/total_sample, total_acc/total_sample*100


class Trainer():
    def __init__(self,args):
        self.args = args
        self.in_feature = 32 + 16
        self.device = self.args.device
        self.num_shot = 5

        # client1 = Client(1,gnnModel(20,self.args),MyDataLoader('./dataset/EXP1_full.npy', 16, self.num_shot),self.device,16,self.num_shot)
        client1 = CenterServer(gnnModel(20,self.args),MyDataLoader('./dataset/EXP1_full.npy', 16, self.num_shot),self.device,16,self.num_shot)

        client2 = Client(2,gnnModel(20,self.args),MyDataLoader('./dataset/EXP2_sub2.npy', 14, self.num_shot),self.device,14,self.num_shot)
        # client2 = CenterServer(gnnModel(20,self.args),MyDataLoader('./dataset/EXP2_full.npy', 14, self.num_shot),self.device,14,self.num_shot)

        client3 = Client(3,gnnModel(20,self.args),MyDataLoader('./dataset/EXP3-r1_full.npy', 18, self.num_shot),self.device,18,self.num_shot)
        # client3 = CenterServer(gnnModel(20,self.args),MyDataLoader('./dataset/EXP3-r1_full.npy', 18, self.num_shot),self.device,18,self.num_shot)

        client4 = Client(4,gnnModel(20,self.args),MyDataLoader('./dataset/EXP3-r2_full.npy', 18, self.num_shot),self.device,18,self.num_shot)
        # client4 = CenterServer(gnnModel(20,self.args),MyDataLoader('./dataset/EXP3-r2_full.npy', 18, self.num_shot),self.device,18,self.num_shot)
        self.clients = [client4, client2, client3]
        self.server = client1

    def train(self,n_epochs):
        self.server.load_pretrain('./dataset/model_fine_tuning.ckpt')
        total_loss = []
        total_acc = []
        for epoch in range(n_epochs):
            self.server.aggregation(self.clients)
            self.server.fast_adaption(5)
            loss, acc = self.server.validation()
            total_acc.append(acc)
            print(f"[Val | {epoch+1}/{n_epochs} ] loss = {loss:.5f}, acc = {acc:.5f} %")
        total_acc = np.array(total_acc)
        print(f"The best result: {np.max(total_acc):.3f} %")
        np.save('./dataset/50_val.npy',total_acc)

