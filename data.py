from utils import data_preproc, count_data
import time
import random
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset

class MyDataLoader(Dataset):
    def __init__(self,path,nway,nshot,train=True):
        super(MyDataLoader,self).__init__()
        self.nway = nway
        self.nshot = nshot
        self.input_channel = 1
        self.size = 120
        self.full_data_dict = self.load_data(path,train)

        print('full_data_num: %d' % count_data(self.full_data_dict))


    def load_data(self,path,train):
        full_data_dict = {}
        data_full, label_full = data_preproc(np.load(path,allow_pickle=True))

        for i in range(len(label_full)):
            if label_full[i] not in full_data_dict:
                full_data_dict[label_full[i]] = [data_full[i]]
            else:
                full_data_dict[label_full[i]].append(data_full[i])

        return full_data_dict

    def load_batch_data(self,batch_size,nway,num_shot,train=True,fast_adapt=False):
        total_way = 20
        data_dict = dict()
        
        if fast_adapt:
            for key in self.full_data_dict.keys():
                data_dict[key] = self.full_data_dict[key][:num_shot+1]
        else:
            data_dict = self.full_data_dict

        x = []
        label_y = []
        one_hot_y = []
        class_y = []

        xi = []
        label_yi = []
        one_hot_yi = []

        map_label2class = []

        for i in range(batch_size):
            sampled_classes = random.sample(data_dict.keys(),nway)
            positive_class = random.randint(0,nway-1)
            label2class = torch.LongTensor(nway)

            single_xi = []
            single_one_hot_yi = []
            single_label_yi = []
            single_class_yi = []
            redun = []

            if train:
                for j, n_class in enumerate(sampled_classes):
                    if j == positive_class:
                        sampled_data = random.sample(data_dict[n_class],num_shot+1)
                        x.append(torch.from_numpy(sampled_data[0]))
                        label_y.append(torch.LongTensor([j]))
                        one_hot = torch.zeros(total_way)
                        one_hot[j] = 1.0
                        one_hot_y.append(one_hot)
                        class_y.append(torch.LongTensor([n_class]))
                        shots_data = torch.Tensor(sampled_data[1:])
                    else:
                        shots_data = torch.Tensor(random.sample(data_dict[n_class],num_shot))

                    single_xi += shots_data.unsqueeze(0)
                    single_label_yi.append(torch.LongTensor([j]).repeat(num_shot))
                    one_hot = torch.zeros(total_way)
                    one_hot[j] = 1.0
                    single_one_hot_yi.append(one_hot.repeat(num_shot,1))

                    label2class[j] = n_class

                single_xi = torch.stack(single_xi,0)
                
                for j in range(total_way-nway):
                    weight = np.random.random(nway)
                    redun.append(torch.Tensor(np.average(single_xi,axis=0,weights=weight)))
                    single_label_yi.append(torch.LongTensor([nway+j]).repeat(num_shot))
                    one_hot = torch.zeros(total_way)
                    one_hot[nway+j] = 1.0
                    single_one_hot_yi.append(one_hot.repeat(num_shot,1))
            else:
                for j, n_class in enumerate(sampled_classes):
                    if j == positive_class:
                        sampled_data = random.sample(data_dict[n_class][:num_shot+1],num_shot)
                        x.append(torch.from_numpy(random.sample(data_dict[n_class][num_shot+1:],1)[0]))
                        label_y.append(torch.LongTensor([j]))
                        one_hot = torch.zeros(total_way)
                        one_hot[j] = 1.0
                        one_hot_y.append(one_hot)
                        class_y.append(torch.LongTensor([n_class]))
                        shots_data = torch.Tensor(sampled_data)
                    else:
                        shots_data = torch.Tensor(random.sample(data_dict[n_class][:num_shot+1],num_shot))

                    single_xi += shots_data.unsqueeze(0)
                    single_label_yi.append(torch.LongTensor([j]).repeat(num_shot))
                    one_hot = torch.zeros(total_way)
                    one_hot[j] = 1.0
                    single_one_hot_yi.append(one_hot.repeat(num_shot,1))

                    label2class[j] = n_class

                single_xi = torch.stack(single_xi,0)
                
                for j in range(total_way-nway):
                    weight = np.random.random(nway)
                    redun.append(torch.Tensor(np.average(single_xi,axis=0,weights=weight)))
                    single_label_yi.append(torch.LongTensor([nway+j]).repeat(num_shot))
                    one_hot = torch.zeros(total_way)
                    one_hot[nway+j] = 1.0
                    single_one_hot_yi.append(one_hot.repeat(num_shot,1))


            redun = torch.stack(redun,dim=0)
            single_xi = torch.cat((single_xi,redun),0)
            single_xi = single_xi.reshape((-1,1,120))
            shuffle_index = torch.randperm(num_shot*total_way)
            xi.append(single_xi[shuffle_index])
            label_yi.append(torch.cat(single_label_yi,dim=0)[shuffle_index])
            one_hot_yi.append(torch.cat(single_one_hot_yi,dim=0)[shuffle_index])
            map_label2class.append(label2class)
        return [torch.stack(x,0), torch.cat(label_y,dim=0), torch.stack(one_hot_y,0), torch.cat(class_y,dim=0), torch.stack(xi,0), torch.stack(label_yi,0), torch.stack(one_hot_yi,0), torch.stack(map_label2class,0)]

    def load_train_batch(self,batch_size,nway,num_shot):
        return self.load_batch_data(batch_size,nway,num_shot,True,False)

    def load_fast_batch(self,batch_size,nway,num_shot):
        return self.load_batch_data(batch_size,nway,num_shot,True,True)

    def load_test_batch(self,batch_size,nway,num_shot):
        return self.load_batch_data(batch_size,nway,num_shot,False,False)

    def get_data_list(self,data_dict):
        data_list = []
        label_list = []
        for i in data_dict.keys():
            for data in data_dict[i]:
                data_list.append(data)
                label_list.append(i)

        now_time = time.time()

        random.Random(now_time).shuffle(data_list)
        random.Random(now_time).shuffle(label_list)

        return data_list, label_list


    def get_full_data_dict(self):
        return self.get_data_list(self.full_data_dict)

    def get_num(self):
        return count_data(self.full_data_dict)




if __name__ == '__main__':
    dataloader = MyDataLoader('./dataset/EXP1_full.npy',16,5)
    batch = dataloader.load_batch_data(16,16,5,True)
    [x,y,one_hot_y,class_y,xi,yi,_,_] = batch
    print(xi.shape)
    print(yi)


