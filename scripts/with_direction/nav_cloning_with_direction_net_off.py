from asyncore import write
from itertools import count
from platform import release
from pyexpat import features, model
import numpy as np
import matplotlib as plt
import os
import time
from os.path import expanduser
from tqdm.auto import tqdm


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
BATCH_SIZE = 64
MAX_DATA = 10000
EPOCH = 80
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
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        # torch.manual_seed(0)
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
        self.writer = SummaryWriter(log_dir='/home/takumi/catkin_ws/src/nav_cloning/runs')
    
    def load_dataset(self, image_path, dir_path, vel_path):
        x_tensor = torch.load(image_path)
        c_tensor = torch.load(dir_path)
        t_tensor = torch.load(vel_path)
        print("load_image:", x_tensor.shape)
        print("load_dir:", c_tensor.shape)
        print("load_vel:", t_tensor.shape)
        dataset = TensorDataset(x_tensor, c_tensor, t_tensor)
        return dataset

    def trains(self, dataset):
        #self.device = torch.device('cuda')
        print(self.device)
        # <Training mode>
        self.net.train()
        # <dataloder>
        # train_dataset = DataLoader(dataset, batch_size=BATCH_SIZE, generator=torch.Generator(
            # 'cpu').manual_seed(0), shuffle=True)
        train_dataset = DataLoader(dataset, batch_size=BATCH_SIZE,
        shuffle=True)

        for epoch in range(EPOCH):
            self.loss_all = 0.0
            for x_tensor, c_tensor, t_tensor in tqdm(train_dataset):
                x_tensor = x_tensor.to(self.device, non_blocking=True)
                c_tensor = c_tensor.to(self.device, non_blocking=True)
                t_tensor = t_tensor.to(self.device, non_blocking=True)

            # <use data augmentation>
                # x_tensor = self.transform_color(x_tensor)
            # <learning>
                self.optimizer.zero_grad()
                y_tensor = self.net(x_tensor, c_tensor)
            # print("y_train=",y_train.shape,"t_tensor",t_tensor.shape)
                loss = self.criterion(y_tensor, t_tensor)
                loss.backward()
                self.loss_all += loss.item()
                self.optimizer.step()
                self.writer.add_scalar("loss", self.loss_all, self.count)
                self.count += 1

            average_loss = self.loss_all / len(train_dataset)
            print(f"Epoch {epoch+1}, Average Loss: {average_loss:.4f}")
        return average_loss

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