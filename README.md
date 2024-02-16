# ---Finally---
I origionally published the post here (https://blog.csdn.net/WildCatFish/article/details/116228950) (Chinese).

---
# ---Rank Results---
Rank 26 on the A leaderboard, Rank 16 on the B leaderboard.

---

# Tianchi Beginner Contest - Heartbeat Signal Classification Prediction - PyTorch CNN Model

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210428183335448.png)
## Competition Introduction

The competition's task is to predict the category of electrocardiogram heartbeat signals. The dataset, visible and downloadable after registration, comes from a platform's electrocardiogram data records, with a total data volume of over 200,000. It mainly consists of a single column of heartbeat signal sequence data, where each sample's signal sequence has consistent sampling frequency and length. To ensure fairness, 100,000 records will be drawn as the training set, 20,000 as test set A, and 20,000 as test set B. The heartbeat signal category (label) information will be anonymized.
| Field | Description |
|--|--|
|id  | Unique identifier assigned to the heartbeat signal |
|heartbeat_signals  |  Heartbeat signal sequence|
|label| Heartbeat signal category (0, 1, 2, 3) |



 [^1]: [mermaid语法说明](https://mermaidjs.github.io/)

## Evaluation Criteria
Participants need to submit the probabilities of four different heartbeat signal predictions. The results submitted by the participants will be compared with the actual heartbeat type results, calculating the absolute value of the difference between the predicted probability and the real value (the smaller, the better).

The specific formula is as follows:
For a certain signal, if the real value is [$y_1$, $y_2$, $y_3$, $y_4$], and the model's predicted probability values are[$a_1$, $a_2$, $a_3$, $a_4$], then the model's average index $abs-sum$ is

$abs-sum$ = $\displaystyle\sum_{y=1}^{n}\displaystyle\sum_{i=1}^{4} |y_i -a_i|$
For example, if the heartbeat signal is 1, it will be encoded as [0, 1, 0, 0]. If the predicted probabilities for different heartbeat signals are [0.1, 0.7, 0.1, 0.1], then the prediction result's $abs-sum$ is
$abs-sum$ = ∣0.1−0∣+∣0.7−1∣+∣0.1−0∣+∣0.1−0∣=0.6
## Data Analysis
This part is already provided by authors on Tianchi notebook, hereby cited. Link is as follows:
[Task 2 Data Analysis](https://tianchi.aliyun.com/forum/postDetail?spm=5176.12586969.1002.27.3cf2170e522P1a&postId=195918)

## CNN Model
This CNN model is implemented with the PyTorch framework.
The general idea is as follows:
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210428171205926.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1dpbGRDYXRGaXNo,size_16,color_FFFFFF,t_70#pic_center)
The above image is intended to help beginners understand what convolutional and sampling layers mean. Due to the limited image size, it was not possible to illustrate kernels and their movements. For the shape of each layer's input, refer to the comments in the author's code.
You can also refer to PyTorch's documentation and source code for understanding, link is as follows:
[nn.Conv1d](https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html#torch.nn.Conv1d)
```python
class Model(nn.Module):
    def __init__(self):
        """
            CNN模型构造
        """
        super(Model, self).__init__()
        self.conv_layer1 = nn.Sequential(
            # input shape(32, 1, 205) -> [batch_size, channel, features]
            # 参考->https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html#torch.nn.Conv1d
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),   # 卷积后(16, 1, 205)
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        # 下采样down-sampling
        self.sampling_layer1 = nn.Sequential(
            # input shape(32, 16, 205)
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # size随便选的, 这里output应该是(32, 32, 102)
        )

        self.conv_layer2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),   # 输出(32, 64, 102)
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.sampling_layer2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),  # 输出(32, 128, 102)
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 输出(32, 64, 51)
        )

        self.conv_layer3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),  # 输出(32, 256, 51)
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.sampling_layer3 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1),  # 输出(32, 512, 51)
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 输出(32, 512, 25)
        )
        # 全连接层
        self.full_layer = nn.Sequential(
            nn.Linear(in_features=512*25, out_features=256*25),
            nn.ReLU(),
            nn.Linear(in_features=256*25, out_features=128*25),
            nn.ReLU(),
            nn.Linear(in_features=128*25, out_features=64*25),
            nn.ReLU(),
            nn.Linear(in_features=64*25, out_features=4)
        )
        # 这个是输出label预测概率, 不知道这写法对不对
        self.pred_layer = nn.Softmax(dim=1)

    def forward(self, x):
        """
            前向传播
        :param x: batch
        :return: training == Ture 返回的是全连接层输出， training == False 加上一个Softmax(), 返回各个label概率.
        """
        x = x.unsqueeze(dim=1)  # 升维. input shape(32, 205), output shape(32, 1, 205)
        x = self.conv_layer1(x)
        x = self.sampling_layer1(x)
        x = self.conv_layer2(x)
        x = self.sampling_layer2(x)
        x = self.conv_layer3(x)
        x = self.sampling_layer3(x)
        x = x.view(x.size(0), -1)   # output(32, 12800)
        x = self.full_layer(x)

        if self.training:
            return x	# CrossEntropyLoss自带LogSoftmax, 所以训练的时候不用输出概率(我也不知道这个写法对不对, 我是试错出来的.)
        else:
            return self.pred_layer(x)
   
```



## Loss Function

**Cross Entropy Loss**:
This loss function is commonly used for multi-class problems. Note that Cross Entropy Loss combines LogSoftmax and NLLLoss. Using Softmax in your output layer might prevent your model from fitting.
Reference: [Cross Entropy Loss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html?highlight=crossentropy#torch.nn.CrossEntropyLoss)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210428174329570.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1dpbGRDYXRGaXNo,size_16,color_FFFFFF,t_70)
**L1 Loss**:
As mentioned in the scoring criteria, this competition uses	$abs-sum$ = $\displaystyle\sum_{y=1}^{n}\displaystyle\sum_{i=1}^{4} |y_i -a_i|$
This is essentially nn.L1Loss() or F.l1_loss() in PyTorch. PyTorch defaults to mean absolute error (MAE), but according to the documentation, you can set reduction='sum' to get sum absolute error (SAE).
L1 Loss的参考链接: [nn.L1Loss](https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html?highlight=loss#torch.nn.L1Loss)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210428181021900.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1dpbGRDYXRGaXNo,size_16,color_FFFFFF,t_70)
## Main Code
Below is the main code for your reference.
```python
def train_loop(dataloader, model, loss_fn, optimizer):
    """
        模型训练部分
    :param dataloader: 训练数据集
    :param model: 训练用到的模型
    :param loss_fn: 评估用的损失函数
    :param optimizer: 优化器
    :return: None
    """
    for batch, x_y in enumerate(dataloader):
        X, y = x_y[:, :205].type(torch.float64), torch.tensor(x_y[:, 205], dtype=torch.long, device='cuda:0')
        # 开启梯度
        with torch.set_grad_enabled(True):
            # Compute prediction and loss
            pred = model(X.float())
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            # Backpropagation
            loss.backward()
            optimizer.step()


def test_loop(dataloader, model, loss_fn):
    """
        模型测试部分
    :param dataloader: 测试数据集
    :param model: 测试模型
    :param loss_fn: 损失函数
    :return: None
    """
    size = len(dataloader.dataset)
    test_loss, correct, l1_loss = 0, 0, 0
    # 用来计算abs-sum. 等于PyTorch L1Loss-->
    # https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html#torch.nn.L1Loss
    l1loss_fn = AbsSumLoss()
    with torch.no_grad():   # 关掉梯度
        model.eval()
        for x_y in dataloader:
            X, y = x_y[:, :205].type(torch.float64), torch.tensor(x_y[:, 205], dtype=torch.long, device='cuda:0')
            # 注意Y和y的区别, Y用来计算L1 loss, y是CrossEntropy loss.
            Y = torch.zeros(size=(len(y), 4), device='cuda:0')
            for i in range(len(Y)):
                Y[i][y[i]] = 1

            pred = model(X.float())
            test_loss += loss_fn(pred, y).item()    # 这个是CrossEntropy loss
            l1_loss += l1loss_fn(pred, Y).item()    # 这个是abs-sum/L1 loss
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()  # 这个是计算准确率的, 取概率最大值的下标.

    test_loss /= size   # 等于CrossEntropy的reduction='mean', 这里有些多此一举可删掉.
    correct /= size
    print(f"Test Results:\nAccuracy: {(100*correct):>0.1f}% abs-sum loss: {l1_loss:>8f} CroEtr loss: {test_loss:>8f}")


def prediction(net, loss):
    """
        对数据进行预测
    :param net: 训练好的模型
    :param loss: 模型的测试误差值, 不是损失函数. 可以去掉, 这里是用来给预测数据命名方便区分.
    :return: None
    """
    with torch.no_grad():
        net.eval()
        pred_loader = torch.utils.data.DataLoader(dataset=pred_data)
        res = []
        for x in pred_loader:
            x = torch.tensor(x, device='cuda:0', dtype=torch.float64)
            output = net(x.float())
            res.append(output.cpu().numpy().tolist())

        res = [i[0] for i in res]
        res_df = pd.DataFrame(res, columns=['label_0', 'label_1', 'label_2', 'label_3'])
        res_df.insert(0, 'id', value=range(100000, 120000))

        res_df.to_csv('res-loss '+str(loss)+'.csv', index=False)


class Model(nn.Module):
    def __init__(self):
        """
            CNN模型构造
        """
        super(Model, self).__init__()
        self.conv_layer1 = nn.Sequential(
            # input shape(32, 1, 205) -> [batch_size, channel, features]
            # 参考->https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html#torch.nn.Conv1d
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),   # 卷积后(32, 16, 205)
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        # 下采样down-sampling
        self.sampling_layer1 = nn.Sequential(
            # input shape(32, 16, 205)
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # size随便选的, 这里output应该是(32, 32, 102)
        )

        self.conv_layer2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),   # 输出(32, 64, 102)
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.sampling_layer2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),  # 输出(32, 128, 102)
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 输出(32, 64, 51)
        )

        self.conv_layer3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),  # 输出(32, 256, 51)
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.sampling_layer3 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1),  # 输出(32, 512, 51)
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 输出(32, 512, 25)
        )
        # 全连接层
        self.full_layer = nn.Sequential(
            nn.Linear(in_features=512*25, out_features=256*25),
            nn.ReLU(),
            nn.Linear(in_features=256*25, out_features=128*25),
            nn.ReLU(),
            nn.Linear(in_features=128*25, out_features=64*25),
            nn.ReLU(),
            nn.Linear(in_features=64*25, out_features=4)
        )
        # 这个是输出label预测概率, 不知道这写法对不对
        self.pred_layer = nn.Softmax(dim=1)

    def forward(self, x):
        """
            前向传播
        :param x: batch
        :return: training == Ture 返回的是全连接层输出， training == False 加上一个Softmax(), 返回各个label概率.
        """
        x = x.unsqueeze(dim=1)  # 升维. input shape(32, 205), output shape(32, 1, 205)
        x = self.conv_layer1(x)
        x = self.sampling_layer1(x)
        x = self.conv_layer2(x)
        x = self.sampling_layer2(x)
        x = self.conv_layer3(x)
        x = self.sampling_layer3(x)
        x = x.view(x.size(0), -1)   # output(32, 12800)
        x = self.full_layer(x)

        if self.training:
            return x    # CrossEntropyLoss自带LogSoftmax, 训练的时候不用输出概率(我也不知道这个写法对不对, 我是试错出来的.)
        else:
            return self.pred_layer(x)


class AbsSumLoss(nn.Module):
    def __init__(self):
        """
            可以直接用PyTorch的nn.L1Loss, 这个我写的时候不知道。
        """
        super(AbsSumLoss, self).__init__()

    def forward(self, output, target):
        loss = F.l1_loss(target, output, reduction='sum')

        return loss


if __name__ == '__main__':
    set_random_seed(1996)   # 设定随机种子
    # 加载数据集
    data = pd.read_csv('train.csv')
    data = process_data(data)
    pred_data = pd.read_csv('testA.csv')
    pred_data = get_pred_x(pred_data)

    # 初始化模型
    lr_rate = 1e-5
    w_decay = 1e-6
    n_epoch = 100
    b_size = 32
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = Model()
    net.to(device)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr_rate, weight_decay=w_decay)
    loss_fn = nn.CrossEntropyLoss(reduction='sum')

    # 拆分训练测试集
    train, test = train_test_split(data, test_size=0.2)
    train, test = torch.cuda.FloatTensor(train), torch.cuda.FloatTensor(test)
    train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=b_size)
    test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=b_size)

    for epoch in range(n_epoch):
        start = time.time()
        print(f"\n----------Epoch {epoch + 1}----------")
        train_loop(train_loader, net, loss_fn, optimizer)
        test_loop(test_loader, net, loss_fn)
        end = time.time()
        print('training time: ', end-start)

    # predict

```

## Conclusion
I'm also a novice, without much experience, so mistakes and oversights are inevitable. Please feel free to correct me. The competition is still ongoing, and if there are new discoveries and experiences, I will continue to share with everyone later.

