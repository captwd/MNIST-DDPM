import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import random
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ================= 定义网络 =================
class ANN(nn.Module):
    def __init__(self,input_size = 784,hidden_size = (1,2,3),output_size = 10):
        super().__init__()
        self.net = nn.Sequential(
        nn.Flatten(),              # 将多维张量转换为一维张量
        nn.Linear(input_size, hidden_size[0]),           #后面有设置具体数量
        nn.ReLU(),                      #改用ReLU
        nn.Linear(hidden_size[0], hidden_size[1]),
        nn.ReLU(),
        nn.Linear(hidden_size[1], hidden_size[2]),
        nn.ReLU(),
        nn.Linear(hidden_size[2], output_size))

    def forward(self, x):
        return self.net(x)  # CrossEntropyLoss 会自动处理 softmax

# ================= 数据加载 =================
def loadData(batch_size=64):                    #这个size不重要，如果后面没有定义才会使用这个
    transform = transforms.Compose([               #compose简化操作
        transforms.ToTensor(),                      #	把 PIL 图像 或 NumPy 数组 变成 PyTorch 张量，并除以255
        transforms.Normalize((0.5,), (0.5,))           #平均值，标准差，标准化，像素点为[0,1]
    ])                                                            #标准化[-1,1]
    train_dataset = datasets.MNIST(root=REPO_ROOT, train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root=REPO_ROOT, train=False, transform=transform, download=True)
                                  #root:仓库根目录（内含 MNIST/raw 官方数据），本地没数据就自动从官网镜像下载
    #train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,)#每次循环都要打乱顺序
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0,)
    return train_loader, test_loader          #准备后续每次循环要使用的小批量数据

# ================= 计算准确率 =================
def computeAcc(model, data_loader, device):
    model.eval()                   #切换到评估模式
    correct, total = 0, 0
    with torch.no_grad():        #关闭自动求图，节省显存，加速前向
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)  # 前向一次，拿到原始分数，tensor，二维矩阵，每个小矩阵中有10个数字，代表0到9对应的类似概率值
            _, predicted = torch.max(outputs, 1)  # 在第一维(类别，0到9)取最大值下标，即获得预测类别
            total += y.size(0)  # 获得0维长度(即样本个数)增加已看的样本数
            correct += (predicted == y).sum().item()  # 累加预测正确的样本数
    return correct / total  # 获得准确率

# ================= 主程序 =================
if __name__ == "__main__":
    start_time = time.time()
    print(" [Init] 开始初始化参数和模型...")

    # ================= 超参数区 =================
    input_size = 784            #输入维度
    hidden_size = (256,64,32)            #隐藏层输出维度，或者说中间的维度？
    output_size = 10            #输出层输出维度
    batch_size = 256            #每个小批量数据中的元素个数
    num_epochs = 10              #训练循环次数
    learning_rate = 0.001        #学习率
    criterion = nn.CrossEntropyLoss()  # 损失函数，可换成 nn.MSELoss()
    optimizer_choice = "Adam"  # 可选 "SGD" / "Adam"
    momentum = 0.9
    # =================================================

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"[Info] 使用设备: {device}")

    train_loader, test_loader = loadData(batch_size=batch_size)    #依次返回train test
    model = ANN(input_size, hidden_size, output_size).to(device)

    # 优化器初始化
    if optimizer_choice.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=1e-4)
               # 随机梯度下降   打包所有要更新的参数        #引入动量，正则化
    elif optimizer_choice.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas = (0.9,0.999), weight_decay=1e-4  )
    else:
        raise ValueError(f"未知优化器: {optimizer_choice}")
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # ================= 训练 =================

    loss_list, acc_list = [], []
    acc_max, acc_max_epoch = 0.0, 0

    for epoch in range(num_epochs):
        model.train()                  #训练模式
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        #设置可视化进度条            在最左侧加入描述性文字
        for X, y in loop:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()                   #清除梯度
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        acc_now = computeAcc(model, test_loader, device)
        loss_list.append(avg_loss)
        acc_list.append(acc_now)
        scheduler.step()

        if acc_now > acc_max:
            acc_max = acc_now
            acc_max_epoch = epoch                      #得到最优的训练次数，？？要自己手动在下一次更改循环次数吗

        tqdm.write(f"[Result] Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Test Acc: {acc_now:.4f}")

    print(" [Done] 模型训练完成")
    print(f"总训练用时: {time.time() - start_time:.2f} 秒")

    # ================= 绘图 =================
    rand_id = random.randint(1000, 9999)
    fname = f"ANN-调参-{rand_id}.png"                #随机取名

    plt.figure(figsize=(12, 5))                  #新建一张画布，指定宽12英寸，高5英寸

    # Loss 曲线
    # Loss 曲线
    # 绘制曲线
    plt.subplot(121)                           #把画布分成一行两列的两个格子，并选第一个作为绘画区域
    plt.plot(range(len(loss_list)), loss_list, "r",label = "Loss")           #横坐标与纵坐标，用红线
    plt.xlabel("Epochs")                                  #横坐标标题
    plt.ylabel("Loss")
    plt.grid(True)                             #打开网格线
    plt.title("Training Loss")

    # Accuracy 曲线
    plt.subplot(122)
    plt.plot(range(len(acc_list)), acc_list, "b", label="Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.title("Test Accuracy")
    plt.annotate(f"Max acc: {acc_max:.4f}",                 #保留四位小数验证集最高准确率
                 xy=(acc_max_epoch, acc_max),                     #箭头指向的坐标
                 xytext=(acc_max_epoch * 0.7, acc_max - 0.003),   #文字框出现位置
                 arrowprops=dict(facecolor='black', shrink=0.05), #画一条从文字框指向xy的黑色箭头
                 fontsize=12)                                      #箭头两端个缩短%5，字体大小12

    # 底部标题显示超参数
    param_text = (f"hidden_size={hidden_size}, lr={learning_rate}, "
                  f"optimizer={optimizer_choice}, loss={criterion._get_name()}, "
                  f"batch_size={batch_size}, epochs={num_epochs}")
    plt.gcf().subplots_adjust(bottom=0.15)  # 留出底部空间
      #get current figure 得到整张画布
    plt.figtext(0.5, 0.01, param_text, ha="center", fontsize=10)
      #在figure上写text
    plt.savefig(fname)
     #难不成是save figure
    plt.show()
    print(f"[Saved] {fname}")
