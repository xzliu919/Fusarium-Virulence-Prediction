#PNNGS Parallel neural network for genomic selection
#https://colab.research.google.com/drive/1o8lfWHvr4WoyTA5Y9b4mSCSw2TEbXJb7?usp=sharing#scrollTo=15ae67a2

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchinfo import summary
import numpy as np
import math
import pandas as pd
import time
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import Ridge
from sklearn import decomposition
from permetrics.regression import RegressionMetric
from imblearn.over_sampling import RandomOverSampler
import scipy.stats as stats
import os

# 禁用Scipy的警告
np.seterr(all='ignore')
torch.backends.cudnn.enabled = False


class Inception1d(nn.Module):
    def __init__(self, in_channels, parallel_number, branch_out_channel=3, stride = 1):
        super(Inception1d, self).__init__()
        self.parallel_number = parallel_number
        # 第一条线路
        # self.branch1 = nn.Conv1d(in_channels= in_channels, out_channels = branch_out_channel, kernel_size= 1,
        #                              stride= stride, padding= 0)
        #
        # # 第二条线路
        # self.branch3 = nn.Conv1d(in_channels = in_channels, out_channels = branch_out_channel, kernel_size= 3,
        #                              stride= stride, padding= 1)
        #
        # # 第三条线路
        # self.branch5 = nn.Conv1d(in_channels = in_channels, out_channels = branch_out_channel, kernel_size= 5,
        #                              stride= stride, padding= 2)
        #
        # # 第四条线路
        # self.branch7 = nn.Conv1d(in_channels=in_channels, out_channels= branch_out_channel, kernel_size=7,
        #                          stride=stride, padding=3)
        #
        # # 第五条线路
        # self.branch9 = nn.Conv1d(in_channels=in_channels, out_channels= branch_out_channel, kernel_size=9,
        #                          stride=stride, padding=4)
        #
        # # 第六条线路
        # self.branch11 = nn.Conv1d(in_channels=in_channels, out_channels= branch_out_channel, kernel_size=11,
        #                          stride=stride, padding= 5)
        #
        # # 第七条线路
        # self.branch13 = nn.Conv1d(in_channels=in_channels, out_channels= branch_out_channel, kernel_size=13,
        #                           stride=stride, padding=6)
        #
        # # 第八条线路
        # self.branch15 = nn.Conv1d(in_channels=in_channels, out_channels= branch_out_channel, kernel_size=15,
        #                           stride=stride, padding=7)

        # 第一条线路
        self.branch1 = nn.Conv1d(in_channels=in_channels, out_channels=1, kernel_size=1,
                                 stride=stride, padding=0)

        # 第二条线路
        self.branch3 = nn.Conv1d(in_channels=in_channels, out_channels=5, kernel_size=3,
                                 stride=stride, padding=1)

        # 第三条线路
        self.branch5 = nn.Conv1d(in_channels=in_channels, out_channels=5, kernel_size=5,
                                 stride=stride, padding=2)

        # 第四条线路
        self.branch7 = nn.Conv1d(in_channels=in_channels, out_channels=5, kernel_size=7,
                                 stride=stride, padding=3)

        # 第五条线路
        self.branch9 = nn.Conv1d(in_channels=in_channels, out_channels=5, kernel_size=9,
                                 stride=stride, padding=4)

        # 第六条线路
        self.branch11 = nn.Conv1d(in_channels=in_channels, out_channels=5, kernel_size=11,
                                  stride=stride, padding=5)

        # 第七条线路
        self.branch13 = nn.Conv1d(in_channels=in_channels, out_channels=5, kernel_size=13,
                                  stride=stride, padding=6)

        # 第八条线路
        self.branch15 = nn.Conv1d(in_channels=in_channels, out_channels=5, kernel_size=15,
                                  stride=stride, padding=7)


    def forward(self, x):
        f1 = self.branch1(x)
        f2 = self.branch3(x)
        f3 = self.branch5(x)
        f4 = self.branch7(x)
        f5 = self.branch9(x)
        f6 = self.branch11(x)
        f7 = self.branch13(x)
        f8 = self.branch15(x)

        if self.parallel_number == 2:
            output = torch.cat((f1, f2), dim=1)
        elif self.parallel_number == 3:
            output = torch.cat((f1, f2, f3), dim=1)
        elif self.parallel_number == 4:
            output = torch.cat((f1, f2, f3, f4), dim=1)
        elif self.parallel_number == 5:
            output = torch.cat((f1, f2, f3, f4, f5), dim=1)
        elif self.parallel_number == 6:
            output = torch.cat((f1, f2, f3, f4, f5, f6), dim=1)
        elif self.parallel_number == 7:
            output = torch.cat((f1, f2, f3, f4, f5, f6, f7), dim=1)
        elif self.parallel_number == 8:
            output = torch.cat((f1, f2, f3, f4, f5, f6, f7, f8), dim=1)
        else:
            output = "error"
        return output


class Inception1d_residual(nn.Module):
    def __init__(self, in_channels, parallel_number, branch_out_channel=3, stride = 1):
        super(Inception1d_residual, self).__init__()
        self.parallel_number = parallel_number
        # 第一条线路
        # self.branch1 = nn.Conv1d(in_channels= in_channels, out_channels = branch_out_channel, kernel_size= 1,
        #                              stride= stride, padding= 0)
        #
        # # 第二条线路
        # self.branch3 = nn.Conv1d(in_channels = in_channels, out_channels = branch_out_channel, kernel_size= 3,
        #                              stride= stride, padding= 1)
        #
        # # 第三条线路
        # self.branch5 = nn.Conv1d(in_channels = in_channels, out_channels = branch_out_channel, kernel_size= 5,
        #                              stride= stride, padding= 2)
        #
        # # 第四条线路
        # self.branch7 = nn.Conv1d(in_channels=in_channels, out_channels= branch_out_channel, kernel_size=7,
        #                          stride=stride, padding=3)
        #
        # # 第五条线路
        # self.branch9 = nn.Conv1d(in_channels=in_channels, out_channels= branch_out_channel, kernel_size=9,
        #                          stride=stride, padding=4)
        #
        # # 第六条线路
        # self.branch11 = nn.Conv1d(in_channels=in_channels, out_channels= branch_out_channel, kernel_size=11,
        #                          stride=stride, padding= 5)
        #
        # # 第七条线路
        # self.branch13 = nn.Conv1d(in_channels=in_channels, out_channels= branch_out_channel, kernel_size=13,
        #                           stride=stride, padding=6)
        #
        # # 第八条线路
        # self.branch15 = nn.Conv1d(in_channels=in_channels, out_channels= branch_out_channel, kernel_size=15,
        #                           stride=stride, padding=7)

        # 第一条线路
        self.branch1 = nn.Conv1d(in_channels=in_channels, out_channels=1, kernel_size=1,
                                 stride=stride, padding=0, bias= False)

        # 第二条线路
        self.branch3 = nn.Conv1d(in_channels=in_channels, out_channels=5, kernel_size=3,
                                 stride=stride, padding=1, bias= False)

        # 第三条线路
        self.branch5 = nn.Conv1d(in_channels=in_channels, out_channels=5, kernel_size=5,
                                 stride=stride, padding=2, bias= False)

        # 第四条线路
        self.branch7 = nn.Conv1d(in_channels=in_channels, out_channels=5, kernel_size=7,
                                 stride=stride, padding=3, bias= False)

        # 第五条线路
        self.branch9 = nn.Conv1d(in_channels=in_channels, out_channels=5, kernel_size=9,
                                 stride=stride, padding=4, bias= False)

        # 第六条线路
        self.branch11 = nn.Conv1d(in_channels=in_channels, out_channels=5, kernel_size=11,
                                  stride=stride, padding=5, bias= False)

        # 第七条线路
        self.branch13 = nn.Conv1d(in_channels=in_channels, out_channels=5, kernel_size=13,
                                  stride=stride, padding=6, bias= False)

        # 第八条线路
        self.branch15 = nn.Conv1d(in_channels=in_channels, out_channels=5, kernel_size=15,
                                  stride=stride, padding=7, bias= False)



    def forward(self, x):
        f1 = self.branch1(x)
        f2 = self.branch3(x)

        # x_filp = torch.flip(x, dims=[-1])
        # f3 = self.branch3(x_filp)


        f3 = self.branch5(x)
        f4 = self.branch7(x)
        f5 = self.branch9(x)
        f6 = self.branch11(x)
        f7 = self.branch13(x)
        f8 = self.branch15(x)

        # print("x.shape：", x.shape)
        x = torch.mean(x, dim= 1, keepdim= True)
        # print("x.shape：", x.shape)
        f1 = f1 + x
        x3 = x.repeat(1,5,1)
        # f2_8 = [f2, f3, f4, f5, f6, f7, f8]

        f2 = f2 + x3

        # x3_filp = torch.flip(x3, dims=[-1])
        # f3 = f3 + x3_filp


        f3 = f3 + x3
        f4 = f4 + x3
        f5 = f5 + x3
        f6 = f6 + x3
        f7 = f7 + x3
        f8 = f8 + x3

        if self.parallel_number == 2:
            output = torch.cat((f1, f2), dim=1)
        elif self.parallel_number == 3:
            output = torch.cat((f1, f2, f3), dim=1)
        elif self.parallel_number == 4:
            output = torch.cat((f1, f2, f3, f4), dim=1)
        elif self.parallel_number == 5:
            output = torch.cat((f1, f2, f3, f4, f5), dim=1)
        elif self.parallel_number == 6:
            output = torch.cat((f1, f2, f3, f4, f5, f6), dim=1)
        elif self.parallel_number == 7:
            output = torch.cat((f1, f2, f3, f4, f5, f6, f7), dim=1)
        elif self.parallel_number == 8:
            output = torch.cat((f1, f2, f3, f4, f5, f6, f7, f8), dim=1)
        else:
            output = "error"
        return output


class PNNGSModel(nn.Module):
    def __init__(self, parallel_number, X_train):
        super(PNNGSModel, self).__init__()
        self.conv1 = Inception1d_residual(in_channels=1, parallel_number=parallel_number, stride=1)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(0.5)
        self.batch = nn.BatchNorm1d(5 * parallel_number - 4)
        # self.conv2 = Inception1d(in_channels= 3 * parallel_number - 2, parallel_number= parallel_number, stride=1)

        self.conv2 = Inception1d_residual(in_channels= 5 * parallel_number - 4, parallel_number=parallel_number, stride=1)
        # self.batch2 = nn.BatchNorm1d(3 * parallel_number - 2)

        self.conv3 = nn.Conv1d(in_channels=5 * parallel_number - 4, out_channels=1, kernel_size=3, stride=1, padding=1, bias= False)

        # self.conv2 = nn.Conv1d(in_channels=3 * parallel_number - 2, out_channels=1, kernel_size=3, stride=1, padding=1)
        # self.batch2 = nn.BatchNorm1d(1)
        # self.conv3 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)

        self.fc = nn.Linear(X_train.shape[1], 1)


    def forward(self, input):
        # print("input.shape:", input.shape)
        x = self.conv1(input)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.batch(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout(x)

        # x = self.conv1(input)
        # x = self.relu(x)
        # x = self.dropout(x)
        # x = self.batch(x)
        #
        # x = self.conv2(x)
        # x = self.relu(x)
        # x = self.dropout(x)

        x = self.conv3(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv1d):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.Linear):
    #             nn.init.normal_(m.weight, 0, 0.01)
    #             nn.init.constant_(m.bias, 0)


class PNNGSModel1(nn.Module):
    def __init__(self, parallel_number, X_train):
        super(PNNGSModel1, self).__init__()
        self.conv1 = Inception1d(in_channels=1, parallel_number=parallel_number, stride=1)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(0.5)
        self.batch = nn.BatchNorm1d(3 * parallel_number - 2)
        # self.conv2 = Inception1d(in_channels= 3 * parallel_number - 2, parallel_number= parallel_number, stride=1)
        self.conv2 = nn.Conv1d(in_channels=3 * parallel_number - 2, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.batch2 = nn.BatchNorm1d(1)
        self.conv3 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(X_train.shape[1], 1)


    def forward(self, input):
        # print("input.shape:", input.shape)
        x = self.conv1(input)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.batch(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv3(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x




class PNNGSModel2(nn.Module):
    def __init__(self, parallel_number, X_train):
        super(PNNGSModel2, self).__init__()
        self.conv1 = Inception1d(in_channels=1, parallel_number=parallel_number, branch_out_channel=8, stride=2)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(0.3)
        self.batch = nn.BatchNorm1d(8 * parallel_number)
        self.conv2 = Inception1d(in_channels=8 * parallel_number, parallel_number= parallel_number, branch_out_channel=8,
                                 stride=1)


        self.conv3 = nn.Conv1d(in_channels=8 * parallel_number, out_channels=3, kernel_size=3, stride=1, padding=1)

        strided = 2
        length = math.ceil(X_train.shape[1] / 2)
        length = math.ceil(length / 1)
        length = length * 3
        self.fc = nn.Linear(length, 1)

        # self.fc = nn.Linear(5 * X_train.shape[1] // 4 + 2, 1)


        # self.fc2 = nn.Linear(2, 1)


    def forward(self, x):
        # print("x.shape:", x.shape)
        # print("x.shape[1]//4:", x.shape[1]//4)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.batch(x)

        # x = self.downsampling1(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        # x = self.batch(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout(x)

        # x = self.downsampling2(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        # x = self.batch(x)

        x = self.conv3(x)
        # x = self.relu(x)
        # #
        # x = self.conv4(x)

        # x = self.downsampling3(x)
        # x = self.relu(x)


        x = torch.flatten(x, 1)
        # print("x.shape:", x.shape)

        # x = self.fc1(x)
        # x = self.relu(x)
        # x = self.fc2(x)
        x = self.fc(x)
        # x = self.relu(x)
        # x = self.fc2(x)
        return x

    def _initialize_weights(self):
        seed = torch.randint(0, 1000, (1,))  # seed必须是int，可以自行设置
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)  # 让显卡产生的随机数一致
        print("seed:", seed)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class PNNGSModel3(nn.Module):
    def __init__(self, parallel_number, X_train):
        super(PNNGSModel3, self).__init__()
        length = 33163
        width = 376
        self.conv1 = Inception1d(in_channels=1, parallel_number=parallel_number, branch_out_channel=4,
                                 stride= length // width)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(0.3)
        self.batch = nn.BatchNorm1d(4 * parallel_number)
        self.conv2 = nn.Conv1d(in_channels=4 * parallel_number, out_channels=1, kernel_size=3, stride=1, padding=1)


        self.conv3 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)

        length1 = math.ceil(length / (length // width))
        length2 = length1 + width
        self.fc = nn.Linear(753, 2)
        self.fc2= nn.Linear(2, 1)

        # self.fc = nn.Linear(5 * X_train.shape[1] // 4 + 2, 1)


        # self.fc2 = nn.Linear(2, 1)


    def forward(self, x):
        # print("x.shape:", x.shape)
        # print("x.shape[1]//4:", x.shape[1]//4)


        x1 = x[:, :, :33163]
        x2 = x[:, :, 33163:]

        x1 = self.conv1(x1)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)
        x1 = self.batch(x1)
        x1 = self.conv2(x1)
        x1 = self.relu(x1)
        # print("x1 = self.relu(x1).shape:", x1.shape)
        # print("x2.shape:", x2.shape)

        x3 = torch.cat((x1, x2), dim=2)
        # print("x3 = torch.cat((x1, x2), dim=2).shape:", x3.shape)
        # x = self.downsampling1(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        # x = self.batch(x)

        # x3 = self.conv3(x3)
        # x3 = self.relu(x3)

        # x = self.downsampling2(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        # x = self.batch(x)

        # x3 = self.conv3(x3)
        # x = self.relu(x)
        # #
        # x = self.conv4(x)

        # x = self.downsampling3(x)
        # x = self.relu(x)


        x3 = torch.flatten(x3, 1)
        # print("x3 = torch.flatten(x3, 1).shape:", x.shape)

        # x = self.fc1(x)
        # x = self.relu(x)
        # x = self.fc2(x)
        x3 = self.fc(x3)
        x3 = self.fc2(x3)
        # x = self.relu(x)
        # x = self.fc2(x)
        return x3

    def _initialize_weights(self):
        seed = torch.randint(0, 1000, (1,))  # seed必须是int，可以自行设置
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)  # 让显卡产生的随机数一致
        print("seed:", seed)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)



def keepOnlyTheLargestCategory(data_file, cluster_file):
    data_file = pd.read_csv(data_file, header=0, index_col=0)
    cluster_file = pd.read_csv(cluster_file, header=0, index_col=0)
    cluster_number = cluster_file["cluster"]
    print(cluster_file["cluster"].value_counts())
    print(cluster_file["cluster"].value_counts().index[0])
    largest_category = cluster_file["cluster"].value_counts().index[0]
    data_file = pd.merge(data_file, cluster_number, how= 'inner', left_index= True, right_index= True)
    data_file = data_file[data_file["cluster"] == largest_category]
    data_file = data_file.drop("cluster", axis= "columns")
    return data_file

def discardDataBycluster(data_file, cluster_file, cluster = 1, NumberOfDiscards = 100):
    data_file = pd.read_csv(data_file, header=0, index_col=0)
    cluster_file = pd.read_csv(cluster_file, header=0, index_col=0)
    cluster_number = cluster_file["cluster"]
    # print(cluster_file["cluster"].value_counts())
    # print(cluster_file["cluster"].value_counts().index[0])
    # largest_category = cluster_file["cluster"].value_counts().index[0]
    data_file = pd.merge(data_file, cluster_number, how='inner', left_index=True, right_index=True)

    condition = data_file["cluster"] == cluster
    drop_line = data_file.loc[condition]
    drop_line = drop_line.sample(n = NumberOfDiscards, random_state = 0)

    data_file = data_file.append(drop_line)
    data_file = data_file.drop_duplicates(keep= False)
    return data_file


def Cluster2Test(data_file, cluster_file, cluster = 1, NumberOfDiscards = 100):
    data_file = pd.read_csv(data_file, header=0, index_col=0)
    cluster_file = pd.read_csv(cluster_file, header=0, index_col=0)
    cluster_number = cluster_file["cluster"]
    print(cluster_file["cluster"].value_counts())
    # print(cluster_file["cluster"].value_counts().index[0])
    # largest_category = cluster_file["cluster"].value_counts().index[0]
    data_file = pd.merge(data_file, cluster_number, how='inner', left_index=True, right_index=True)

    condition = data_file["cluster"] == cluster
    drop_line = data_file.loc[condition]
    drop_line = drop_line.sample(n = NumberOfDiscards, random_state = 0)

    data_file = data_file.append(drop_line)
    data_file = data_file.drop_duplicates(keep= False)
    return data_file, drop_line


def loss3(output, target):
    loss = torch.mean((torch.abs(output - target))**3)
    return loss

def loss05(output, target):
    loss = torch.mean(torch.sqrt(torch.abs(output - target)))
    return loss

def loss033(output, target):
    loss = torch.mean((torch.abs(output - target))**0.33)
    return loss

def loss_infinite(output, target):
    loss = torch.mean(torch.max(torch.abs(output - target)))
    return loss


def PNNGS(pheno,
          data_file,
          x_validation,
          n_splits = 10,
          epoch = 1500,
          CUDA_VISIBLE_DEVICES = '0',
          parallel_number = 3,
          Batch_Size = 32,
          save_path = '../save_model/PNNGS.pth'):
    '''

    :param pheno: pheno name
    :param data_file: phenotype+genotype
    :param n_splits: the split number
    :param epoch:
    :param CUDA_VISIBLE_DEVICES: the number of CUDA_VISIBLE_DEVICES
    :param parallel_number:
    :param Batch_Size:
    :param save_path:
    :return: average prediction and prediction list
    '''
    # os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
    # print("★CUDA_VISIBLE_DEVICES:", CUDA_VISIBLE_DEVICES)
    repeat_times = 2
    early_stop = 50
    print("2个损失函数")

    # data_file = pd.read_csv(data_file, header=0, index_col=0)

    y = data_file[pheno]
    X = data_file.drop([pheno], axis=1)

    X = np.array(X)
    print("X.shape:", X.shape)
    print("Dropout:", 0.5)



    # kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)





    train_pearns_list, test_pearns_list = [], []
    NRMSES = []
    # train_loss_list, test_loss_list = [], []
    fold = 0
    best_net = 0
    best_intercept = 0
    best_pearns_10folds = 0

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    for train_index, test_index in kf.split(X):
        print("---------------fold: "+ str(fold) + "---------------")
        fold += 1
        # if fold == 1 or fold == 2:
        #     continue
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index].values, y[test_index].values

        ridge_model = Ridge(random_state=0)
        ridge_model.fit(X_train, y_train)
        y_pre = ridge_model.predict(X_test)
        pearn_ridge, _ = stats.pearsonr(y_pre, y_test)
        print("pearn_ridge:", pearn_ridge)
        print("ridge_model.intercept_:", ridge_model.intercept_)


        y_train = y_train - ridge_model.intercept_
        y_test = y_test - ridge_model.intercept_



        X_train_torch = torch.from_numpy(X_train.astype(np.float32))
        y_train_torch = torch.from_numpy(y_train.astype(np.float32))
        X_test_torch = torch.from_numpy(X_test.astype(np.float32))
        y_test_torch = torch.from_numpy(y_test.astype(np.float32))

        X_train_torch = torch.unsqueeze(X_train_torch, 1)
        y_train_torch = torch.unsqueeze(y_train_torch, 1)
        X_test_torch = torch.unsqueeze(X_test_torch, 1)
        y_test_torch = torch.unsqueeze(y_test_torch, 1)

        trainset = torch.utils.data.TensorDataset(X_train_torch, y_train_torch)
        testset = torch.utils.data.TensorDataset(X_test_torch, y_test_torch)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=Batch_Size, shuffle=True, num_workers=8)
        testloader = torch.utils.data.DataLoader(testset, batch_size=Batch_Size, shuffle=False, num_workers=8)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("torch.cuda.is_available():", torch.cuda.is_available())
        print("device:", device)

        # if pearn_mean <= 0.6:
        #     net = PNNGSModel1(parallel_number, X_train)
        #     print("pearn_mean:", pearn_mean)
        #     print("net is PNNGSModel1")
        # else:
        #     net = PNNGSModel2(parallel_number, X_train)
        #     print("pearn_mean:", pearn_mean)
        #     print("net is PNNGSModel2")
        #
        net = PNNGSModel(parallel_number, X_train)
        print("net is PNNGSModel")
        print("total_params:", sum(p.numel() for p in net.parameters()))

        if device == 'cuda':
            net = nn.DataParallel(net)
            torch.backends.cudnn.benchmark = True



        best_pearns = 0
        repeat_time = 0
        loss_selection = 0
        NRMSE = 0

        # lr_list = []
        while repeat_time < repeat_times:
        # for repeat_time in range(repeat_times):
            seed = loss_selection  # seed必须是int，可以自行设置
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)  # 让显卡产生的随机数一致
            print("seed:", seed)

            # if repeat_time == 0:
            #     net = PNNGSModel1(parallel_number, X_train)
            #     print("net is PNNGSModel1")
            # else:
            #     net = PNNGSModel2(parallel_number, X_train)
            #     print("net is PNNGSModel2")
            # print("total_params:", sum(p.numel() for p in net.parameters()))

            # if device == 'cuda':
            #     net = nn.DataParallel(net)
            #     torch.backends.cudnn.benchmark = True

            optimizer = optim.Adam(net.parameters(), weight_decay= 0.1,
                                   amsgrad= True)

            criterion1 = nn.L1Loss()
            criterion2 = nn.MSELoss()

            best_epoch = 0
            nan_number = 0

            print("\rrepeat_time:", repeat_time)
            print("early_stop:", early_stop)



            for i in range(epoch):
                # start = time.time()
                # train_loss = 0
                if torch.cuda.is_available():
                    net = net.to(device)
                net.train()
                for step, data in enumerate(trainloader, start=0):
                    # if torch.rand(1) > training_rate:
                    #     continue
                    im, label = data
                    im = im.to(device)
                    label = label.to(device)

                    optimizer.zero_grad()
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()

                    outputs = net(im)

                    if loss_selection % 2 == 0:
                        loss = criterion1(outputs, label)
                    elif loss_selection % 2 == 1:
                        loss = criterion2(outputs, label)


                    loss.backward()
                    optimizer.step()

                    # train_loss += loss.data

                # train_outputs = net(X_train_torch)
                # train_outputs = train_outputs.cpu().detach().numpy().squeeze()
                # train_pearns = stats.pearsonr(train_outputs, y_train)[0]
                #
                # train_pearns_list.append(train_pearns)

                net.eval()
                test_outputs = []
                with torch.no_grad():
                    for step, data in enumerate(testloader, start=0):
                        X_test_input, Y_test_label = data
                        X_test_input = X_test_input.to(device)
                        test_output = net(X_test_input)
                        test_output = test_output.cpu().detach().numpy().squeeze()
                        test_output = list(test_output)
                        test_outputs = test_outputs + test_output


                #
                test_outputs = np.array(test_outputs)

                if np.isnan(test_outputs.any()):
                    nan_number += 1
                    print("nan_number:", nan_number)
                    repeat_time -= 1

                    if loss_selection > 100:
                        repeat_time += 1
                    break

                test_pearns = stats.pearsonr(test_outputs + ridge_model.intercept_,
                                             y_test + ridge_model.intercept_)[0]
                # print("\rEpoch:", i+1, "test_pearns:", test_pearns)

                # test_outputs = net(X_test_torch)
                # print("test_outputs:", test_outputs)

                if test_pearns > best_pearns:
                    # torch.save(net, save_path)
                    evaluator = RegressionMetric(test_outputs + ridge_model.intercept_,
                                             y_test + ridge_model.intercept_)
                    NRMSE = evaluator.NRMSE(multi_output="raw_values")
                    best_pearns = test_pearns

                    # best_net = net
                    # best_intercept = ridge_model.intercept_

                    best_epoch = i
                    print("\rEpoch:", i+1, "best_pearns:", best_pearns)

                if test_pearns > best_pearns_10folds:
                    best_pearns_10folds = test_pearns

                    best_net = net
                    best_intercept = ridge_model.intercept_


                    print("save model")

                if i - best_epoch > early_stop:
                    break

                if i > 60 and best_pearns < pearn_ridge:
                    break


                if best_pearns > pearn_ridge + 0.11 and i - best_epoch > 40:
                    break

                if np.isnan(test_pearns):
                    nan_number += 1
                    print("nan_number:", nan_number)
                    repeat_time -= 1

                    if loss_selection > 100:
                        repeat_time += 1
                    break


            print("repeat_time: ", repeat_time)
            repeat_time += 1
            loss_selection += 1
            print("final epoch:", i)

        print("best_pearns:", best_pearns)
        test_pearns_list.append(best_pearns)
        NRMSES.append(NRMSE)

    average_pearns = np.mean(test_pearns_list)
    print("np.mean(pearns):", np.mean(test_pearns_list))
    # print("pearns:", test_pearns_list)
    # print("np.std(test_pearns_list):", np.std(test_pearns_list))
    #
    # print("np.mean(NRMSES):", np.mean(NRMSES))
    # print("NRMSES:", NRMSES)
    # print("np.std(NRMSES):", np.std(NRMSES))

    x_validation_index = x_validation.index
    x_validation = np.array(x_validation)
    x_validation = torch.from_numpy(x_validation.astype(np.float32))
    x_validation = x_validation.to(device)
    x_validation = torch.unsqueeze(x_validation, 1)
    validation_output = best_net(x_validation)
    validation_output = validation_output.cpu().detach().numpy().squeeze()
    validation_output = validation_output + best_intercept


    for i in range(len(validation_output)):
        if validation_output[i] < 0:
            validation_output[i] = 0
    print("validation_output:", validation_output)
    validation_output = pd.DataFrame(data= validation_output,
                                     columns= [pheno],
                                     index= x_validation_index)
    validation_output.to_csv("data/fusariumAllSamples/233Fusarium_infecting_plants_phenotype_PNNGS_" + pheno + ".csv")


    return average_pearns, test_pearns_list


