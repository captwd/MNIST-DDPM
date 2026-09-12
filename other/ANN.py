import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# 定义网络
class ANN(nn.Module):
    def __init__(self, input_size=784, hidden_size=15, output_size=10):
        super(ANN, self).__init__()
        self.flatten = nn.Flatten()           # 将多维张量转换为一维张量
        self.fc1 = nn.Linear(input_size, hidden_size)       #后面有设置具体数量
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        return x  # 不加 softmax，CrossEntropyLoss 会处理


# 加载数据
def loadData(batch_size=64):                    #这个size不重要，如果后面没有定义才会使用这个
    transform = transforms.Compose([               #compose简化操作
        transforms.ToTensor(),                      #	把 PIL 图像 或 NumPy 数组 变成 PyTorch 张量
        transforms.Normalize((0.5,), (0.5,))           #平均值，标准差，标准化，像素点为[0,1]
    ])                                                            #，名称标准化，实际归一化，[-1,1]
    train_dataset = datasets.MNIST(root=REPO_ROOT, train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root=REPO_ROOT, train=False, transform=transform, download=True)
                                  #root:仓库根目录（内含 MNIST/raw 官方数据），本地没数据就自动从官网镜像下载

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,)#每次循环都要打乱顺序
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0,)
    return train_loader, test_loader          #准备后续每次循环要使用的小批量数据


##
#划分验证集train_x,val_x,train_y,val_y =,数据太少，没必要


# 计算准确率
def computeAcc(model, data_loader):
    model.eval()        #切换到评估模式
    correct, total = 0, 0
    with torch.no_grad():#关闭自动求图，节省显存，加速前向
        for X, y in data_loader:
            outputs = model(X)#前向一次，拿到原始分数，tensor，二维矩阵，每个小矩阵中有10个数字，代表0到9对应的类似概率值
            _, predicted = torch.max(outputs, 1)#在第一维(类别，0到9)取最大值下标，即获得预测类别
            total += y.size(0)#获得0维长度(即样本个数)增加已看的样本数
            correct += (predicted == y).sum().item()#累加预测正确的样本数
    return correct / total#获得准确率


if __name__ == "__main__":#只有当这个文件被直接运行，才执行后面的语句
                         #如果该文件被当作库函数import则不会执行后面的
    start_time = time.time()#记录开始的时间
    print(" [Init] 开始初始化参数和模型...")

    # ================= 超参数配置（可优化空间） =================
    input_size = 784               #输入维度
    hidden_size = 15               #隐藏层输出维度，或者说中间的维度？
    output_size = 10               #输出层输出维度
    learning_rate = 0.1            #学习率
    batch_size = 64                #每个小批量数据中的元素个数
    num_epochs = 5                 #训练循环次数
    # =========================================================

    # 数据
    train_loader, test_loader = loadData(batch_size=batch_size)#依次返回train test

    # 模型
    model = ANN(input_size, hidden_size, output_size)

    # 损失函数 & 优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
                #随机梯度下降   打包所有要更新的参数

    # 训练过程
    loss_list, acc_list = [], []         #准备两个空列表
    acc_max, acc_max_epoch = 0.0, 0

    for epoch in range(num_epochs):
        model.train()                 #训练模式
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        #设置可视化进度条            在最左侧加入描述性文字
        for X, y in loop:
            optimizer.zero_grad()          #清除梯度
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()                      #仅取出数字

        avg_loss = running_loss / len(train_loader)            #计算平均数
        acc_now = computeAcc(model, test_loader)               #获得测试集的准确度
        loss_list.append(avg_loss)
        acc_list.append(acc_now)

        if acc_now > acc_max:
            acc_max = acc_now
            acc_max_epoch = epoch

        tqdm.write(f"[Result] Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Test Acc: {acc_now:.4f}")

    print(" [Done] 模型训练完成")
    print(f"总训练用时: {time.time() - start_time:.2f} 秒")

    # 保存模型
    # model_path = "mnist_ann_cuda.pt"                     定义文件名
    # torch.save(model.state_dict(), model_path)       把模型当前的模型参数序列化，写入文件
    # print(f"[Info] 模型已保存为: {model_path}")

    # 绘制曲线
    plt.figure(figsize=(12, 5))                #新建一张画布，指定宽12英寸，高5英寸
    plt.subplot(121)                           #把画布分成一行两列的两个格子，并选第一个作为绘画区域
    plt.plot(range(len(loss_list)), loss_list, "r")           #横坐标与纵坐标，用红线
    plt.xlabel("Epochs")                                  #横坐标标题
    plt.ylabel("Loss")
    plt.grid(True)                             #打开网格线

    plt.subplot(122)
    plt.plot(range(len(acc_list)), acc_list, "r")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.annotate(f"Max acc: {acc_max:.4f}",             #保留四位小数验证集最高准确率
                 xy=(acc_max_epoch, acc_max),               #箭头指向的坐标
                 xytext=(acc_max_epoch * 0.7, acc_max - 0.02),       #文字框出现位置
                 arrowprops=dict(facecolor='black', shrink=0.05),   #画一条从文字框指向xy的黑色箭头
                 fontsize=12)                                      #箭头两端个缩短%5，字体大小12
    plt.savefig("ANN_plot.png")                  #把当前图像保存为"ANN_plot.png"
    plt.show()                                 #显示图片
