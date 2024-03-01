import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm


class UserDataset(Dataset):
    def __init__(self, path):
        pd.set_option('display.max_columns', None)  # 显示所有列
        self.df = pd.read_csv(path)
        # self.df = self.df.tail(int(len(self.df) * 0.1))  # 取后百分之10行
        self.df = self.df.drop(['city_name', 'county_name'], axis=1)

        # 将第一列设置为字符串类型
        self.df.iloc[:, 0] = self.df.iloc[:, 0].astype(str)
        # 将其他列设置为浮点数类型
        self.df.iloc[:, 1:] = self.df.iloc[:, 1:].astype('float32')

        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        array = imp.fit_transform(self.df.iloc[:, 1:])  # 补充缺失值
        self.df.iloc[:, 1:] = pd.DataFrame(array, dtype='float32')

    def __getitem__(self, item):
        phone = self.df.iloc[item].values[0]
        lst = self.df.iloc[item].values[1:-1].astype('float32')
        label = self.df.iloc[item].values[-1].astype('float32')
        return phone, torch.tensor(lst), torch.tensor(label)

    def __len__(self):
        return len(self.df)


class AppDataset(Dataset):
    def __init__(self, path):
        self.df = pd.read_csv(path)
        # self.df = self.df.tail(int(len(self.df) * 0.1))  # 取后百分之10行
        self.df = self.df.drop(['month_id', 'busi_name'], axis=1)

        # 将第一列设置为字符串类型
        self.df.iloc[:, 0] = self.df.iloc[:, 0].astype(str)
        # 将其他列设置为浮点数类型
        self.df.iloc[:, 1:] = self.df.iloc[:, 1:].astype('float32')

        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        array = imp.fit_transform(self.df.iloc[:, 1:])  # 补充缺失值
        self.df.iloc[:, 1:] = pd.DataFrame(array, dtype='float32')

    def __getitem__(self, item):
        phone = self.df.iloc[item].values[0]
        lst = self.df.iloc[item].values[1:-1].astype('float32')
        label = self.df.iloc[item].values[-1].astype('float32')
        return phone, torch.tensor(lst), torch.tensor(label)

    def __len__(self):
        return len(self.df)


class VocDataset(Dataset):
    def __init__(self, path):
        self.df = pd.read_csv(path)
        self.df = self.df.iloc[:, [0, 4, 8]]

        # 将第一列设置为字符串类型
        self.df.iloc[:, 0] = self.df.iloc[:, 0].astype(str)
        # 将其他列设置为浮点数类型
        self.df.iloc[:, 1:] = self.df.iloc[:, 1:].astype('float32')

        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        array = imp.fit_transform(self.df.iloc[:, 1:])  # 补充缺失值
        self.df.iloc[:, 1:] = pd.DataFrame(array, dtype='float32')

    def __getitem__(self, item):
        phone = self.df.iloc[item].values[0]
        lst = self.df.iloc[item].values[1:-1].astype('float32')
        label = self.df.iloc[item].values[-1].astype('float32')
        return phone, torch.tensor(lst), torch.tensor(label)

    def __len__(self):
        return len(self.df)

class UserTestDataset(Dataset):
    def __init__(self, path):
        pd.set_option('display.max_columns', None)  # 显示所有列
        self.df = pd.read_csv(path)
        self.df = self.df.tail(int(len(self.df) * 0.1))  # 取后百分之10行
        self.df = self.df.drop(['city_name', 'county_name'], axis=1)

        # 将第一列设置为字符串类型
        self.df.iloc[:, 0] = self.df.iloc[:, 0].astype(str)
        # 将其他列设置为浮点数类型
        self.df.iloc[:, 1:] = self.df.iloc[:, 1:].astype('float32')

        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        array = imp.fit_transform(self.df.iloc[:, 1:])  # 补充缺失值
        self.df.iloc[:, 1:] = pd.DataFrame(array, dtype='float32')

    def __getitem__(self, item):
        phone = self.df.iloc[item].values[0]
        lst = self.df.iloc[item].values[1:-1].astype('float32')
        label = self.df.iloc[item].values[-1].astype('float32')
        return phone, torch.tensor(lst), torch.tensor(label)

    def __len__(self):
        return len(self.df)


class UserModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(9, 512),
            nn.BatchNorm1d(512),
            nn.Sigmoid(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Sigmoid(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.Sigmoid(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.linear(x)


class AppModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(1, 64),
            nn.Sigmoid(),
            nn.LayerNorm(64),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.linear(x)

class VocModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(1, 64),
            nn.Sigmoid(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.linear(x)

def train(model, training_dataloader, hyper, loss, optim, device):
    model.train()
    for epoch in range(hyper['epochs']):
        for phone, feature, label in tqdm(training_dataloader, desc=f"epoch: {epoch}"):
            feature = feature.to(device)
            label = label.to(device)
            y = model(feature).flatten()
            optim.zero_grad()
            cost = loss(y, label)
            print(cost.item())
            cost.backward()
            optim.step()
        print(f"loss: {cost}")


def test(model, testing_dataloader, hyper, device, dic):
    '''
    初步测试，将当前模型的预测结果保存在字典里返回，键是用户，值是预测的label
    '''
    model = model.to(device)
    model.eval()
    for phone, feature, label in tqdm(testing_dataloader):
        phone = phone[0]
        feature = feature.to(device)
        label = label.to(device)
        y = model(feature)
        # print(feature, y)
        if phone not in dic:
            dic[phone] = [y]
        else:
            dic[phone].append(y)

    for phone, lst in dic.items():
        zeros = 0
        ones = 0
        for n in lst:
            if n < 0.5:
                zeros += 1
            else:
                ones += 1
        y = 0 if zeros > ones else 1
        dic[phone] = y
    return dic

def run(hyper, device, model, table_name, TableDataset, dic):
    '''
    创建数据集，DataLoader，训练模型，模型测试
    '''
    # Define loss function and optimizer
    loss = nn.MSELoss().to(device)
    optim = torch.optim.Adam(model.parameters(), hyper['lr'])
    # optim = torch.optim.RMSprop(model.parameters(), lr=hyper['lr'], momentum=hyper['momentum'])

    # Load data and split dataset
    dataset = TableDataset("dataset/train/"+table_name+'.csv')
    training_set, testing_set = torch.utils.data.random_split(dataset, [0.8, 0.2])
    training_loader = DataLoader(training_set, batch_size=hyper['batch_size'], shuffle=True)
    testing_loader = DataLoader(testing_set, shuffle=True, batch_size=1)

    # train & save
    train(model, training_loader, hyper, loss, optim, device)
    torch.save(model, table_name+".pth")

    # load & test
    # model = torch.load("models/" + table_name + ".pth")
    test(model, testing_loader, hyper, 'cuda', dic)
    del model


def train_models():
    hyper = {'epochs': 500, 'batch_size': 64, 'lr': 0.01, 'device': 'cuda', 'momentum': 0.1}
    device = torch.device(hyper['device'])

    '''
    创建三个字典，分别代表当前用户在各个表里的预测结果
    '''
    # train_user
    # user_dic = {}
    # user_model = UserModel().to(device)
    # run(hyper, device, user_model, 'train_user', UserDataset, user_dic)

    # train_app
    app_dic = {}
    hyper['epochs'] = 1
    app_model = AppModel().to(device)
    run(hyper, device, app_model, 'train_app', AppDataset, app_dic)

    # train_voc
    # voc_dic = {}
    # voc_model = VocModel().to(device)
    # run(hyper, device, voc_model, 'train_voc', VocDataset, voc_dic)

    # return user_dic, app_dic, voc_dic

def load_label_map():
    df = pd.read_csv("dataset/label_map.csv")
    # print(df)
    label_map = {}
    for k, v in df.iloc[:, :].values:
        label_map[k] = v
    return label_map

# def judge(dic, label_map):
#     acc = 0
#     for phone, lst in dic.items():
#         y = sum(lst) / len(lst) # 用均值代表最终结果
#         y = 1 if y >= 0.5 else 0
#         if label_map[phone] == y:
#             acc += 1
#     print(f"Accuracy: {acc / len(dic)}")

def judge(dicts, label_map):
    acc = 0
    n = len(dicts[0])
    for phone, label in dicts[0].items():
        y = 0
        y += 1 if label == 1 else -1
        y += 1 if phone in dicts[1] and dicts[1][phone] == 1 else -1
        y += 1 if phone in dicts[2] and dicts[2][phone] == 1 else -1
        if y > 0:
            y = 1
        elif y < 0:
            y = 0
        else:   # 如果1和0出现的次数一样，则忽略
            n -= 1
            continue
        if y == label:
            acc += 1
    print(f"Accuracy: {acc / n}")

if __name__ == '__main__':

    dicts = train_models()
    # torch.save(dicts, 'dic.pth')
    # dicts = torch.load('dic.pth')
    #
    # label_map = load_label_map()
    # judge(dicts, label_map)
