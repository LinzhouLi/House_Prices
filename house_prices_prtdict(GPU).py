import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

#导入数据
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

#数据处理：
#删除部分列
train_features = train_data.drop(['Id','Summary', 'Address','Elementary School','Middle School','High School','State','Parking features','Cooling features','Heating features','City','Region'], axis=1)
test_features = test_data.drop(['Id', 'Summary','Address','Elementary School','Middle School','High School','State','Parking features','Cooling features','Heating features','City','Region'], axis=1)

#用Sold Price填充Listed Price,Tax assessed value
train_features['Tax assessed value'].fillna(train_features['Sold Price'], inplace=True)
train_features['Listed Price'].fillna(train_features['Sold Price'], inplace=True)

train_features = train_features.drop(['Sold Price'], axis=1)
all_features = pd.concat((train_features, test_features))

#将日期转化为int

all_features['Listed On'] = all_features['Listed On'].apply(pd.to_datetime,format='%Y-%m-%d %H:%M:%S.%f')
all_features['Last Sold On'] = all_features['Last Sold On'].apply(pd.to_datetime,format='%Y-%m-%d %H:%M:%S.%f')

all_features['Listed On'] = all_features['Listed On'].apply(
    lambda x: (x.year-2000)*12 + x.month)
all_features['Last Sold On'] = all_features['Last Sold On'].apply(
    lambda x: (x.year-2000)*12 + x.month)

#将Bedrooms转化为int
def n_bedrooms(s):
    if type(s) is float:
        return s
    else:
        l = len(s)
        if l <= 2:
            return int(s)
        s = s.lower()
        l1 = s.count(',') + s.count('/')
        l2 = s.count('more')
        return l1 + 1 + int(1.5 * l2)


all_features['Bedrooms'] = all_features['Bedrooms'].apply(
    lambda s: n_bedrooms(s))

#数据标准化
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
all_features[numeric_features] = all_features[numeric_features].fillna(0)

#分词处理Heating,Cooling,Parking,Type,Flooring,Appliances included,Laundry features
features = ['Heating','Cooling','Parking','Type','Flooring','Appliances included','Laundry features']

all_features[features] = all_features[features].fillna('none')
all_features[features] = all_features[features].apply(
    lambda x: x.astype(str).str.lower())

Heating = all_features.Heating.str.get_dummies(sep=',')#244
Cooling = all_features.Cooling.str.get_dummies(sep=',')#196
Parking = all_features.Parking.str.get_dummies(sep=',')#332
Type = all_features.Type.str.get_dummies(sep=',')#114
Flooring = all_features.Flooring.str.get_dummies(sep=',')#141
Appliance = all_features['Appliances included'].str.get_dummies(sep=',')#162
Laundry = all_features['Laundry features'].str.get_dummies(sep=',')#247

all_features = all_features.drop(features, axis = 1)
all_features = pd.concat([all_features, Heating, Cooling, Parking, Type, Flooring, Appliance, Laundry], axis = 1)

#训练：
def try_gpu(i=0):  
    """如果存在，则返回gpu(i)，否则返回cpu()。"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():  
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]。"""
    devices = [
        torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32, device=try_gpu())
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(train_data['Sold Price'].values.reshape(-1, 1), dtype=torch.float32, device=try_gpu())
train_labelslog = torch.log(train_labels)#取对数后的label

loss = nn.MSELoss()
in_features = train_features.shape[1]

def get_net():
    net = nn.Sequential(nn.Linear(in_features, 1024), nn.ReLU(), nn.Dropout(drop),
                    nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(drop),
                    nn.Linear(256, 32), nn.ReLU(), nn.Dropout(drop),
                    nn.Linear(32, 1))
    return net

def log_rmse(preds, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    clipped_preds = torch.clamp(preds, 1, float('inf'))
    return torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
  
def myloss(preds, labels):
    return torch.sqrt(sum((torch.log(preds) - torch.log(labels))**2) / len(labels))

def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # 这里使用的是Adam优化算法
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            #l = log_rmse(net(X), y)
            #l = myloss(net(X), y)
            l.backward()
            optimizer.step()
        #train_ls.append(log_rmse(net(train_features), train_labels).item())
        train_ls.append(loss(net(train_features), train_labels).item())
        if test_labels is not None:
            #test_ls.append(log_rmse(net(test_features), test_labels).item())
            test_ls.append(loss(net(test_features), test_labels).item())
    return train_ls, test_ls

def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid

def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
        print(f'fold {i + 1}, train log rmse {float(train_ls[-1]):f}, '
              f'valid log rmse {float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k

#超参数：
k, num_epochs, lr, weight_decay, batch_size, drop = 5, 300, 0.0001, 0.001, 4096, 0

net = get_net()
net = net.to(device=try_gpu())

train_l, valid_l = k_fold(k, train_features, train_labelslog, num_epochs, lr, weight_decay, batch_size)
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
   f'平均验证log rmse: {float(valid_l):f}')

#保存数据：
net1 = net.to(torch.device('cpu'))
preds = torch.exp(net1(test_features)).detach().numpy()
test_data['Sold Price'] = pd.Series(preds.reshape(1, -1)[0])
submission = pd.concat([test_data['Id'], test_data['Sold Price']], axis=1)
submission.to_csv('prediction.csv', index=False)