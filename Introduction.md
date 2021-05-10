# 基于多层感知机的房价预测模型

项目地址：[LinzhouLi/House_Prices (github.com)](https://github.com/LinzhouLi/House_Prices)

## 1 数据集

本项目的数据集来自Kaggle上的一个比赛：[California House Prices | Kaggle](https://www.kaggle.com/c/california-house-prices/data)。整个数据集描述了2020年交易的美国加州的房价及其对应信息。

这个数据集分为测试集(test.csv)和训练集(train.csv)，训练集有47325条数据，41条维度。其中交易价格(Sold Price)是需要预测的数据，序号(Id)是所有信息的标号。剩下39条维度分别为：地址(Address)、描述(Summary)、类型(Type)、建造年份(Year built)、取暖(Heating)、空调(Cooling)、车库(Parking)、庭院面积(Lot)、卧室(Bedrooms)、洗手间(Bathrooms)、浴室(Full bathrooms)、室内居住面积(Total interior livable area)、总建筑面积(Total spaces)、车库面积(Garage spaces)、地区(Region)、小学(Elementary School)、小学评分(Elementary School Score)、小学距离(Elementary School Distance)、中学(Middle School)、中学评分(Middle School Score)、中学距离(Middle School Distance)、高中(High School)、高中评分(High School Score)、高中距离(High School Distance)、地板(Flooring)、取暖信息(Heating features)、空调信息(Cooling features)、家具(Appliances included)、洗衣信息(Laundry features)、车库信息(Parking features)、缴税评估价(Tax assessed value)、全年房产税(Annual tax amount)、上架信息(Listed On)、报价(Listed Price)、上次售卖信息(Last Sold On)、上次售卖价格(Last Sold Price)、城市(City)、邮编(Zip)、州(State)。

## 2 数据预处理

整个数据集的39个维度中，有大量保存字符串信息的维度，还有大量缺失数据，需要经过一定的编码和处理才能用于机器学习模型的训练。

### 2.1 丢弃数据

首先，数据集中有一些维度的数据和房价相关性不大，可以直接丢弃，比如地址、描述上架时间以及周边小学中学和高中的名字。

其次，取暖信息(Heating)、空调信息(Cooling)、车库信息(Parking)分别与取暖(Heating features)、空调(Cooling features)、车库(Parking)相似、可以只保留后三个维度。

最后，与地理信息有关的数据，包括州(State)、城市(City)和地区(Region)都是与邮编(Zip)完全相关的，只保留邮编这一维度即可。

### 2.2 处理Bedrooms维度

Bedrooms维度中部分数据是整形，表示该房子有几个卧室，有些数据是字符串，描述了该房子每个卧室的情况。所以可以把字符串型数据中的“,” “/”以及"more"出现的次数统计出来，用来估算卧室的个数，从而将所有数据均转化为数值型。代码如下：

```python
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
```

### 2.3 处理其他文本数据

对于文本数据，每一个维度基本都是由几个单词描述，并由逗号分隔所组成的字符串。所以我们可以把每一个维度的数据的单词都提取出来，并以one-hot的方式编码。

```python
features = ['Heating','Cooling','Parking','Type','Flooring','Appliances included','Laundry features']

#处理缺失值并将字符数据全部转化为小写
all_features[features] = all_features[features].fillna('none')
all_features[features] = all_features[features].apply(
    lambda x: x.astype(str).str.lower())

#one-hot编码
Heating = all_features.Heating.str.get_dummies(sep=',')
Cooling = all_features.Cooling.str.get_dummies(sep=',')
Parking = all_features.Parking.str.get_dummies(sep=',')
Type = all_features.Type.str.get_dummies(sep=',')
Flooring = all_features.Flooring.str.get_dummies(sep=',')
Appliance = all_features['Appliances included'].str.get_dummies(sep=',')
Laundry = all_features['Laundry features'].str.get_dummies(sep=',')

#将编码后的数据帧和原本的数值型数据合成一个数据帧
all_features = all_features.drop(features, axis = 1)
all_features = pd.concat([all_features, Heating, Cooling, Parking, Type, Flooring, Appliance, Laundry], axis = 1)
```

**one-hot编码：**

one-hot编码是一种特征数字化的方法，它采用n个状态位对n个状态进行编码。如果一列离散特征值有n个互异的值，就将一列特征扩展为n列，每一列保存0/1数据，表示表示这一行是否有此特征。从而one-hot编码可以把离散的字符数据转化成计算机能够处理的数值型数据，同时还保存了离散值蕴含的信息。

但one-hot编码也有缺点，如果每一列的离散特征值几乎都是互异的，那经过one-hot编码后就会增加大量的列，存储大量0，非常消耗存储和计算资源。在本项目中，如果直接对所有文本数据进行one-hot编码，会导致内存不足。而通过将文本分词后再进行编码，增加的属性只有1400多列，计算机仍然能够处理

### 2.4 数值型数据标准化

首先，对于数据集中的缺失数据，用其他数据的平均值填充。但是对于标价(Listed Price)和缴税评估价(Tax assessed value)这种对房价预测过程权重非常高的数据，则直接用成交价格(Sold Price)来替换。代码如下：

```python
train_features['Tax assessed value'].fillna(train_features['Sold Price'], inplace=True)
train_features['Listed Price'].fillna(train_features['Sold Price'], inplace=True)
```

其次，由于每个维度的数据范围相差很大，甚至量纲和数量级都不同，所有如果直接使用原始数据来训练模型，就相当于给较大的数据分配了一个较大的初始权重，并且“惩罚”那些较小的数据。所以需要数据标准化，从而将所有特征都放在一个共同的尺度上。
$$
x\leftarrow\frac{x-\mu}{\sigma}\;\;\;其中\mu=\frac{1}{n}\sum_{i=1}^n x_i,\sigma=\sqrt{\frac{1}{n-1}\sum_{i=1}^n(x_i-\overline x)^2}
$$
通过z-score标准化，计算每个维度数据的均值和方差，然后将数据都映射到[0,1]的区间上，且均值为0，方差为1。代码如下：

```python
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
all_features[numeric_features] =
all_features[numeric_features].fillna(0) #用0填充缺失值，相当于用均值填充
```

将数据预处理之后，数据集的维度是(79065, 1464)。

## 3 训练

### 3.1 训练模型

本项目训练模型采用了多层感知机(MLP)，输入的数据维度是1464维，第一层有1024个结点，第二层有64个结点，输出层为1。每一个线性层后使用的激活函数是ReLU函数，同时还加入了Dropout正则化。

```python
in_features = train_features.shape[1]

def get_net():
    net = nn.Sequential(nn.Linear(in_features, 1024), nn.ReLU(), nn.Dropout(drop),
                        nn.Linear(1024, 64), nn.ReLU(), nn.Dropout(drop),
                        nn.Linear(64, 1))
    return net
```



### 3.2 损失函数

对于房价而言，差距可能非常大，如果使用方均误差作为损失函数，会使得对高低房价的相等的偏移得出相等的误差。但显然，对于10w的房子和1000w的房子同样预测偏差1w元所对应的误差应该是大不相同的。所以我们更关心预测结果的相对误差,
$$
\frac{y-\widehat y}{y}
$$
解决这个问题的一个方法是用价格的对数衡量差异，即将
$$
e^{-\delta}\le \frac{\widehat y}{y}\le e^{\delta}\;\;转化为\;\;|log\,y-log\,\widehat y|\le \delta
$$
所以最终的损失函数为：
$$
\sqrt{\frac{1}{n}\sum_{i=1}^{n}(log\,y-log\,\widehat y)^2}
$$
对应的代码为：

```python
loss = nn.MSELoss()#loss()为均方误差

def log_rmse(preds, labels):
    rmse = torch.sqrt(loss(torch.log(preds), torch.log(labels)))
    return rmse
```

### 3.3 优化方法

本项目中采用的优化方法是Adam优化算法，它是一种对随机梯度下降算法的扩展，引入了自适应学习率的机制，所以对学习率的设置不太敏感。训练的代码如下：

```python
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # Adam优化算法
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = log_rmse(net(X), y) # 取对数后的方均误差
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net(train_features), train_labels).item())
        if test_labels is not None:
            test_ls.append(log_rmse(net(test_features), test_labels).item())
    return train_ls, test_ls # 返回训练集上的损失和测试集上的损失
```

### 3.4 训练结果

```python
k, num_epochs, lr, weight_decay, batch_size, drop = 2, 300, 0.0001, 0.001, 4096, 0

train_l, valid_l = k_fold(k, train_features, train_labelslog, num_epochs, lr, weight_decay, batch_size)
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
   f'平均验证log rmse: {float(valid_l):f}')
```

学习率设置为0.0001，每批数据为4096个，训练集共扫描300轮，训练结果如下图：

![image-20210430010815100](D:\Documents\人工智能导论\房价预测训练结果.png)

## 4 总结与体会

本项目基于pytorch，深度学习框架已经完善地封装了各种模型，优化方法，自动求导等，非常易于使用。但是对于实际问题，数据处理和设置合理的超参数更为重要。

在数据处理方面，我最开始对所有文本数据进行one-hot编码，但是直接导致数据帧过大，训练时内存不足。之后我删除了所有文本数据，虽然完成了训练，但由于去掉了许多有用信息，所以训练后的模型偏差较大，平均的log-rmse损失在0.3以上。最后通过观察数据的特征，对每一列文本数据经过","分割后再用one-hot编码，处理后的数据维度小了很多，同时仍保留了原数据集的信息。

调整超参数时，我刚开始将学习率设置在0~10，训练后的曲线波动非常大，效果不好。之后又将学习率调整到1e-3的数量级，训练后的损失曲线平滑了很多，但是产生了严重的过拟合，训练集上的损失小于0.1但测试集损失仍大于0.3。最后我进一步调小学习率，并加入了Dropout的正则化，缓解了过拟合现象。