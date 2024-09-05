from asyncore import write
from itertools import count
from platform import release
from pyexpat import features, model
import numpy as np
import matplotlib as plt
import os
import time
from os.path import expanduser


import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from yaml import load


# HYPER PARAM
BATCH_SIZE = 8
MAX_DATA = 10000
EPOCH = 5
PADDING_DATA = 7 #3


class Net(nn.Module):
    def __init__(self, n_channel, n_out):
        super().__init__()
    # Network CNN 3 + FC 2 + fc2
       # nn. is with parameters to be adjusted
        self.conv1 = nn.Conv2d(n_channel, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(960, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(259, 259)
        self.fc7 = nn.Linear(259, n_out)
        self.relu = nn.ReLU(inplace=True)
    # Weight set
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        torch.nn.init.kaiming_normal_(self.fc4.weight)
        torch.nn.init.kaiming_normal_(self.fc5.weight)
        torch.nn.init.kaiming_normal_(self.fc6.weight)
        torch.nn.init.kaiming_normal_(self.fc7.weight)
        #self.fc7.weight = nn.Parameter(torch.zeros(n_channel,260))
        #self.maxpool = nn.MaxPool2d(2,2)
        #self.batch = nn.BatchNorm2d(0.2)
        self.flatten = nn.Flatten()
    # CNN layer
        self.cnn_layer = nn.Sequential(
            self.conv1,
            self.relu,
            self.conv2,
            self.relu,
            self.conv3,
            self.relu,
            # self.maxpool,
            self.flatten
        )
    # FC layer
        self.fc_layer = nn.Sequential(
            self.fc4,
            self.relu,
            self.fc5,
            self.relu
        )
    # Concat layer (CNN output + Cmd data)
        self.concat_layer = nn.Sequential(
            self.fc6,
            self.relu,
            self.fc7
        )
    # forward layer

    def forward(self, x, c):
        x1 = self.cnn_layer(x)
        x2 = self.fc_layer(x1)
        x3 = torch.cat([x2, c], dim=1)
        x4 = self.concat_layer(x3)
        return x4


class deep_learning:
    def __init__(self, n_channel=3, n_action=1):
        # tensor device choiece
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(0)
        self.net = Net(n_channel, n_action)
        self.net.to(self.device)
        print(self.device)
        self.optimizer = optim.Adam(
            self.net.parameters(), eps=1e-2, weight_decay=5e-4)
        self.totensor = transforms.ToTensor()
        self.transform_color = transforms.ColorJitter(
            brightness=0.25, contrast=0.25, saturation=0.25)
        self.n_action = n_action
        self.count = 0
        self.on_count = 0
        self.accuracy = 0
        self.loss_all =0.0
        self.results_train = {}
        self.results_train['loss'], self.results_train['accuracy'] = [], []
        self.loss_list = []
        self.acc_list = []
        self.dir_list = []
        self.datas = []
        self.target_angles = []
        self.criterion = nn.MSELoss()
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.first_flag = True
        torch.backends.cudnn.benchmark = False
        # self.writer = SummaryWriter(log_dir='/home/orne_beta/haruyama_ws/src/nav_cloning/runs')

        #self.writer = SummaryWriter(log_dir="/home/haru/nav_ws/src/nav_cloning/runs",comment="log_1")
    def make_dataset(self,img, dir_cmd, target_angle):        
        if self.first_flag:
            self.x_cat = torch.tensor(
                img, dtype=torch.float32, device=self.device).unsqueeze(0)
            self.x_cat = self.x_cat.permute(0, 3, 1, 2)
            self.c_cat = torch.tensor(
                dir_cmd, dtype=torch.float32, device=self.device).unsqueeze(0)
            self.t_cat = torch.tensor(
                [target_angle], dtype=torch.float32, device=self.device).unsqueeze(0)
            # filename = "zihanki_migi"
            # image_path = '/home/orne_beta/haruyama_ws/src/nav_cloning/data/dataset_with_dir_selected_training/pytorch/image/'+filename+'/image.pt'
            # dir_path = '/home/orne_beta/haruyama_ws/src/nav_cloning/data/dataset_with_dir_selected_training/pytorch/dir/'+filename+'/dir.pt'
            # vel_path = '/home/orne_beta/haruyama_ws/src/nav_cloning/data/dataset_with_dir_selected_training/pytorch/vel/'+filename+'/vel.pt'
            # self.x_cat = torch.load(image_path)
            # self.c_cat = torch.load(dir_path)
            # self.t_cat = torch.load(vel_path)
            # print("x_shape:",self.x_cat.shape)
            # print("c_shape:",self.x_cat.shape)
            # print("t_shape:",self.t_cat.shape)
            self.first_flag = False
        # x= torch.tensor(self.transform(img),dtype=torch.float32, device=self.device).unsqueeze(0)
        # <To tensor img(x),cmd(c),angle(t)>
        x = torch.tensor(img, dtype=torch.float32,
                         device=self.device).unsqueeze(0)
        # <(Batch,H,W,Channel) -> (Batch ,Channel, H,W)>
        x = x.permute(0, 3, 1, 2)
        c = torch.tensor(dir_cmd, dtype=torch.float32,
                         device=self.device).unsqueeze(0)
        t = torch.tensor([target_angle], dtype=torch.float32,
                         device=self.device).unsqueeze(0)
        if dir_cmd == (0,1,0) or dir_cmd == (0,0,1):  
            for i in range(PADDING_DATA):
                self.x_cat = torch.cat([self.x_cat, x], dim=0)
                self.c_cat = torch.cat([self.c_cat, c], dim=0)
                self.t_cat = torch.cat([self.t_cat, t], dim=0)
            print("Padding data!!!!!")
        else:
            self.x_cat = torch.cat([self.x_cat, x], dim=0)
            self.c_cat = torch.cat([self.c_cat, c], dim=0)
            self.t_cat = torch.cat([self.t_cat, t], dim=0)

        # <make dataset>
        #print("train x =",x.shape,x.device,"train c =" ,c.shape,c.device,"tarain t = " ,t.shape,t.device)
        dataset = TensorDataset(self.x_cat, self.c_cat, self.t_cat)
        # <dataloder>
        train_dataset = DataLoader(dataset, batch_size=BATCH_SIZE, generator=torch.Generator(
            'cpu').manual_seed(0), shuffle=True)
        print("dataset_num:",len(dataset))
        return dataset,len(dataset) ,train_dataset
    
    def trains(self):
        if self.first_flag:
            return
        #self.device = torch.device('cuda')
        # print("on_training:",self.on_count)
        self.net.train()
        dataset = TensorDataset(self.x_cat, self.c_cat, self.t_cat)
        train_dataset = DataLoader(dataset, batch_size=BATCH_SIZE, generator=torch.Generator(
            'cpu').manual_seed(0), shuffle=True)
        for x_train, c_train, t_train in train_dataset:
            x_train.to(self.device, non_blocking=True)
            c_train.to(self.device, non_blocking=True)
            t_train.to(self.device, non_blocking=True)
            break
            
        self.optimizer.zero_grad()
        y_train = self.net(x_train, c_train)
        loss = self.criterion(y_train, t_train)
        loss.backward()
        self.loss_all = loss.item()
        self.optimizer.step()
        # self.writer.add_scalar("on_loss", loss_on, self.on_count)
        # self.on_count +=1

        return self.loss_all

    def act_and_trains(self, img, dir_cmd, train_dataset):
        # self.device = torch.device('cuda')
        # print(self.device)
        # <Training mode>
        self.net.train()
        #dataset, dataset_num ,train_dataset = self.make_dataset(img,dir_cmd,target_angle)
        
            # <split dataset and to device>
        for x_train, c_train, t_train in train_dataset:
            x_train.to(self.device, non_blocking=True)
            c_train.to(self.device, non_blocking=True)
            t_train.to(self.device, non_blocking=True)
            break
    # <use data augmentation>
       # x_train = self.transform_color(x_train)
    # <learning>
        self.optimizer.zero_grad()
        y_train = self.net(x_train, c_train)
    # print("y_train=",y_train.shape,"t_train",t_train.shape)
        loss = self.criterion(y_train, t_train)
        loss.backward()
        self.loss_all = loss.item() 
        self.optimizer.step()
        # self.writer.add_scalar("loss",loss,self.count)

        # <test>
        self.net.eval()
        x_act = torch.tensor(
            img, dtype=torch.float32, device=self.device).unsqueeze(0)
        x_act= x_act.permute(0, 3, 1, 2)
        c_act = torch.tensor(dir_cmd, dtype=torch.float32,
                              device=self.device).unsqueeze(0)
        action_value_training = self.net(x_act, c_act)
        # self.writer.add_scalar("loss", self.loss_all, self.count)
        # self.count += 1
        #print("action=" ,action_value_training[0][0].item() ,"loss=" ,loss.item())

        # if self.first_flag:
        #     self.writer.add_graph(self.net,(x,c))
        # self.writer.close()
        # self.writer.flush()
        # <reset dataset>
        # if self.x_cat.size()[0] > MAX_DATA:
        #     self.x_cat = self.x_cat[1:]
        #     self.c_cat = self.c_cat[1:]
        #     self.t_cat = self.t_cat[1:]
            # self.x_cat = torch.empty(1, 3, 48, 64).to(self.device)
            # self.c_cat = torch.empty(1, 4).to(self.device)
            # self.t_cat = torch.empty(1, 1).to(self.device)
            # self.dir_list = torch.empty(1,4).to(self.device)
            # self.dir_list = torch.empty(1, 4).to(self.device)
            # self.target_angles = torch.empty(1, 1).to(self.device)
            # self.first_flag = True
            # print("reset dataset")

        return action_value_training[0][0].item(), self.loss_all

    def act(self, img, dir_cmd):
        self.net.eval()
        # <make img(x_test_ten),cmd(c_test)>
        # x_test_ten = torch.tensor(self.transform(img),dtype=torch.float32, device=self.device).unsqueeze(0)
        x_test_ten = torch.tensor(
            img, dtype=torch.float32, device=self.device).unsqueeze(0)
        x_test_ten = x_test_ten.permute(0, 3, 1, 2)
        c_test = torch.tensor(dir_cmd, dtype=torch.float32,
                              device=self.device).unsqueeze(0)
        # print(x_test_ten.shape,x_test_ten.device,c_test.shape,c_test.device)
        # <test phase>
        action_value_test = self.net(x_test_ten, c_test)

        #print("act = " ,action_value_test.item())
        return action_value_test[0][0].item()

    def result(self):
        accuracy = self.accuracy
        return accuracy

    def save(self, save_path):
        # <model save>
        path = save_path + time.strftime("%Y%m%d_%H:%M:%S")
        os.makedirs(path)
        # torch.save(self.net.state_dict(), path + '/model_gpu.pt')
        torch.save({
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss_all,
        }, path + '/model.pt')
        print("save_model")

    def save_tensor(self,input_tensor,save_path,file_name):
        path = save_path + time.strftime("%Y%m%d_%H:%M:%S")
        os.makedirs(path)
        torch.save(input_tensor, path + file_name)
        print("save_model_tensor:",)

    def load(self, load_path):
        # <model load>
        # self.net.load_state_dict(torch.load(load_path))
        checkpoint = torch.load(load_path)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.loss_all = checkpoint['loss']
        print("load_model =", load_path)


if __name__ == '__main__':
    dl = deep_learning()