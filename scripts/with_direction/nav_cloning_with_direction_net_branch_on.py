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
BATCH_SIZE = 16
MAX_DATA = 10000
EPOCH =20
BRANCH = 3

class Net(nn.Module):
    def __init__(self, n_channel, n_out):
        super().__init__()
    # Network CNN 3 + FC 2 + fc2
       # nn. is with parameters to be adjusted
        self.conv1 = nn.Conv2d(n_channel, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate the size after convolution
        # Input size: (n_channel, 48, 64)
        # Conv1: (32, (48-8)/4+1=11, (64-8)/4+1=15) -> (32, 11, 15)
        # Conv2: (64, (11-3)/2+1=5, (15-3)/2+1=7) -> (64, 5, 7)
        # Conv3: (64, (5-3)/1+1=3, (7-3)/1+1=5) -> (64, 3, 5)
        # Flatten: 64*3*5 = 960

        self.fc4 = nn.Linear(960, 512)
        self.fc5 = nn.Linear(512, 512)
        self.fc6 = nn.Linear(512, 256)
        self.fc7 = nn.Linear(256, n_out)
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
        self.branch = nn.ModuleList([
            nn.Sequential(
                self.fc6,
                self.relu,
                self.fc7
            )for i in range(BRANCH)
        ])
    # forward layer
    def forward(self, x, c):
        img_out = self.cnn_layer(x)
        fc_out = self.fc_layer(img_out)
        outputs = torch.stack([branch(fc_out) for branch in self.branch], dim=0)
        cmd_index = torch.argmax(c, dim=1).unsqueeze(0).expand(outputs.size(1), -1).permute(1, 0)
        selected_outputs = torch.gather(outputs, 0, cmd_index.unsqueeze(-1).expand(-1, -1, outputs.size(2))).squeeze(0)
        return selected_outputs

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
            brightness=0.5, contrast=0.5, saturation=0.5)
        self.n_action = n_action
        self.count = 0
        self.count_on = 0
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

        dummy_input = torch.randn(1, 3, 48, 64).to(self.device)
        cmd_dirs = [torch.tensor([[1, 0, 0]]).to(self.device),
                    torch.tensor([[0, 1, 0]]).to(self.device),
                    torch.tensor([[0, 0, 1]]).to(self.device)]
        
        for cmd_dir in cmd_dirs:
            self.writer.add_graph(self.net, (dummy_input, cmd_dir))

    def loss_branch(self, dir_cmd, target, output):
        command = torch.argmax(dir_cmd, dim=1)
        loss_branch = [(output[:, i] - target) ** 2 for i in range(output.size(1))]
        loss_function = sum(loss_branch) / len(loss_branch)
        return torch.mean(loss_function)

    def act_and_trains(self, img, dir_cmd, target_angle):
        # <Training mode>
        self.net.train()
        
        if self.first_flag:
            self.x_cat = torch.tensor(
                img, dtype=torch.float32, device=self.device).unsqueeze(0)
            self.x_cat = self.x_cat.permute(0, 3, 1, 2)
            self.c_cat = torch.tensor(
                dir_cmd, dtype=torch.float32, device=self.device).unsqueeze(0)
            self.t_cat = torch.tensor(
                [target_angle], dtype=torch.float32, device=self.device).unsqueeze(0)
            self.first_flag = False

        # <To tensor img(x),cmd(c),angle(t)>
        x = torch.tensor(img, dtype=torch.float32,
                         device=self.device).unsqueeze(0)
        # <(Batch,H,W,Channel) -> (Batch ,Channel, H,W)>
        x = x.permute(0, 3, 1, 2)
        c = torch.tensor(dir_cmd, dtype=torch.float32,
                         device=self.device).unsqueeze(0)
        t = torch.tensor([target_angle], dtype=torch.float32,
                         device=self.device).unsqueeze(0)
        self.x_cat = torch.cat([self.x_cat, x], dim=0)
        self.c_cat = torch.cat([self.c_cat, c], dim=0)
        self.t_cat = torch.cat([self.t_cat, t], dim=0)

        # <make dataset>
        #print("train x =",x.shape,x.device,"train c =" ,c.shape,c.device,"tarain t = " ,t.shape,t.device)
        dataset = TensorDataset(self.x_cat, self.c_cat, self.t_cat)
        # <dataloder>
        train_dataset = DataLoader(dataset, batch_size=BATCH_SIZE, generator=torch.Generator(
            'cpu').manual_seed(0), shuffle=True)

        # <split dataset and to device>
        for x_train, c_train, t_train in train_dataset:
            x_train.to(self.device, non_blocking=True)
            c_train.to(self.device, non_blocking=True)
            t_train.to(self.device, non_blocking=True)
            break
        # <use data augmentation>
        # x_train = self.transform_color(x_train)
        
        # デバッグ用
        print(f"c_train: {c_train}")
        
        # <learning>
        self.optimizer.zero_grad()
        y_train = self.net(x_train, c_train)
        # print("y_train=",y_train.shape,"t_train",t_train.shape)
        loss = self.loss_branch(c_train, t_train, y_train)
        loss.backward()
        self.loss_all = loss.item()
        self.optimizer.step()
        self.writer.add_scalar("Loss/train", loss.item(), self.count)

        

    # インデックスの調整
        # for i in range(y_train.size(1)):
        #     self.writer.add_scalar(f"Branch_{i}/output", y_train[:, i].mean().item(), self.count)   
        #     self.count += 1

        # <test>
        self.net.eval()
        action_value_training = self.net(x, c)
        return action_value_training.item(), self.loss_all

    def act(self, img, dir_cmd):
        self.net.eval()
        # <make img(x_test_ten),cmd(c_test)>
        x_test_ten = torch.tensor(
            img, dtype=torch.float32, device=self.device).unsqueeze(0)
        x_test_ten = x_test_ten.permute(0, 3, 1, 2)
        c_test = torch.tensor(dir_cmd, dtype=torch.float32,
                              device=self.device).unsqueeze(0)
        # <test phase>
        action_value_test = self.net(x_test_ten, c_test)
        return action_value_test.item()

    def result(self):
        accuracy = self.accuracy
        return accuracy

    def save(self, save_path):
        # <model save>
        path = save_path + time.strftime("%Y%m%d_%H:%M:%S")
        os.makedirs(path)
        torch.save(self.net.state_dict(), path + '/model_gpu.pt')
        print("save_model")

    def load(self, load_path):
        # <model load>
        self.net.load_state_dict(torch.load(load_path))
        print("load_model =", load_path)


if __name__ == '__main__':
    dl = deep_learning()