import torch
from torchvision import datasets, transforms
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ================= 数据加载 =================
#？？？？？我拿到的数据集就已经是张量了？
def load_mnist_numpy():
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5,), (0.5,))
    # ])           #这个用于深度学习后面要搭配dataloader才能用
    # train_dataset = datasets.MNIST(root='./MNIST', train=True, transform=transform, download=True) transform只会在取元素的时候生效
    # test_dataset = datasets.MNIST(root='./MNIST', train=False, transform=transform, download=True)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)#每次循环都要打乱顺序
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    train_dataset = datasets.MNIST(root=REPO_ROOT, train=True, download=True)#没有太大变化
    test_dataset = datasets.MNIST(root=REPO_ROOT, train=False, download=True)#加载测试集


    # 与深度模型保持一致：先 /255 归到 [0,1]，再标准化到 [-1,1]
    X_train = train_dataset.data.numpy().astype(np.float32).reshape(-1, 28*28) / 255.0       #怎么要对数据进行两次处理
    X_train = (X_train - 0.5) / 0.5          #统一与深度模型相同的 [-1,1] 归一化，保证对比实验对齐
    #怎么理解只有一次归一化
    y_train = train_dataset.targets.numpy()     #60000 张训练图片对应的 0–9 标签的张量,转化为numpy数组
    X_test = test_dataset.data.numpy().astype(np.float32).reshape(-1, 28*28) / 255.0   #reshape把二维图片拉直
    X_test = (X_test - 0.5) / 0.5             #同样标准化到 [-1,1]
    y_test = test_dataset.targets.numpy()

    return X_train, y_train, X_test, y_test

# ================= 传统 ML 模型 =================
def run_svm(X_train, y_train, X_test, y_test):
    model = SVC(kernel='rbf', gamma='scale')    #核函数，拟合非线性边界   根据实际情况选择数值
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return "SVM", acc

def run_knn(X_train, y_train, X_test, y_test):
    model = KNeighborsClassifier(n_neighbors=5)    #k临近算法，取众数
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return "KNN", acc

def run_rf_progress(X_train, y_train, X_test, y_test, n_estimators=20):
    """逐树训练 RandomForest"""
    model = RandomForestClassifier(n_estimators=1, warm_start=True, random_state=42)
          #可以逐渐增加树，并保留前面训练好的树，即便i增加依然每轮只用训练一棵树      控制随机种子
    print(f"[Training] RandomForest 开始训练 {n_estimators} 棵树...")
    for i in tqdm(range(1, n_estimators+1), desc="Fitting RF"):
             #转换成进度条                  前缀
        model.set_params(n_estimators=i)  # ✅ 用 set_params 动态地更改模型参数，动态设置最大迭代次数
        model.fit(X_train, y_train)
    print("[Predicting] RandomForest 开始预测...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return "RandomForest", acc

def run_logistic_progress(X_train, y_train, X_test, y_test, max_iter=50):
    """逐轮训练 LogisticRegression"""
    model = LogisticRegression(max_iter=1, solver='lbfgs', warm_start=True)
                        #                优化算法,换成拟牛顿法
    print(f"[Training] LogisticRegression 开始训练 {max_iter} 轮...")
    for i in tqdm(range(1, max_iter+1), desc="Fitting Logistic"):
        model.set_params(max_iter=i)  # ✅ 用 set_params
        model.fit(X_train, y_train)
    print("[Predicting] LogisticRegression 开始预测...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return "LogisticRegression", acc

def run_logistic(X_train, y_train, X_test, y_test,max_iter=50):
    print("[Predicting] LogisticRegression 开始预测...")
    model = LogisticRegression(max_iter=max_iter, solver='lbfgs')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return "LogisticRegression", acc

# ================= 绘图函数 =================
def plot_results(model_name, acc):
    plt.figure(figsize=(6, 4))
    plt.bar([model_name], [acc], color="skyblue")   #画柱状图
    plt.ylim(0, 1)          #y轴的显示范围
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} on MNIST (Acc={acc:.4f})")
    fname = f"MNIST_{model_name}.png"
    plt.savefig(fname)
    plt.show()
    print(f"[Saved] {fname}")

# ================= 主程序 =================
if __name__ == "__main__":
    start_time = time.time()
    print("[Init] 加载 MNIST 数据...")
    X_train, y_train, X_test, y_test = load_mnist_numpy()

    # 模型池
    model_pool = {
        "svm": run_svm,
        "knn": run_knn,
        "rf": run_rf_progress,
        "logistic": run_logistic_progress,
        "logistic1": run_logistic
    }

    # 选择模型
    choice = "svm"  # 可改成 "svm" / "knn" / "rf" / "logistic"/"logistic1"
    print(f"[Info] 选择模型: {choice}")

    # 可调参数
    if choice == "rf":
        n_estimators = 15  # 少量树，演示用
        model_name, acc = model_pool[choice](X_train, y_train, X_test, y_test, n_estimators=n_estimators)
    elif choice == "logistic":
        max_iter = 30  # 少量迭代
        model_name, acc = model_pool[choice](X_train, y_train, X_test, y_test, max_iter=max_iter)
    else:
        model_name, acc = model_pool[choice](X_train, y_train, X_test, y_test)

    print(f"[Result] {model_name} Test Accuracy: {acc:.4f}")

    # 绘图
    plot_results(model_name, acc)

    print(f"[Done] 总耗时: {time.time() - start_time:.2f} 秒")
