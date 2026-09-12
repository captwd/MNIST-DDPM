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


# ================= Vision Transformer =================
class ViT(nn.Module):
    def __init__(self, img_size=28, patch_size=7, in_channels=1,
                 num_classes=10, dim=128, depth=6, heads=4,
                 mlp_ratio=4.0, dropout=0.1):
        super(ViT, self).__init__()
        # 28x28、patch=7 -> 4x4=16 个图块
        self.num_patches = (img_size // patch_size) ** 2

        # 图块嵌入：用 stride=patch_size 的卷积一次完成切分+线性投影
        self.patch_embed = nn.Conv2d(in_channels, dim,
                                     kernel_size=patch_size, stride=patch_size)

        # 可学习的 [class] token 和位置编码
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, dim))
        self.pos_drop = nn.Dropout(dropout)

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # 分类头：LayerNorm + Linear
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # (B,1,28,28) -> (B,dim,4,4) -> (B,16,dim)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        # 拼上 [class] token，再加位置编码
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)

        x = self.encoder(x)
        x = self.norm(x[:, 0])   # 只取 [class] token 的输出做分类
        return self.head(x)      # 输出 logits，不做 softmax


# ================= 数据加载 =================
def loadData(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(root=REPO_ROOT, train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root=REPO_ROOT, train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# ================= 计算准确率 =================
def computeAcc(model, data_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total


# ================= 主程序 =================
if __name__ == "__main__":
    start_time = time.time()
    print(" [Init] 开始初始化参数和模型...")

    # ================= 超参数 =================
    batch_size = 64
    num_epochs = 5            # 和 ANN/CNN 相同的训练轮数，保证公平
    learning_rate = 3e-4      # ViT 用小学习率 + AdamW
    weight_decay = 0.05
    # ==========================================

    # GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] 使用设备: {device}")

    # 数据
    train_loader, test_loader = loadData(batch_size=batch_size)

    # 模型
    model = ViT().to(device)
    print(f"[Info] ViT 参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 学习率调度
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # 训练
    loss_list, acc_list = [], []
    acc_max, acc_max_epoch = 0.0, 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for X, y in loop:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()

        avg_loss = running_loss / len(train_loader)
        acc_now = computeAcc(model, test_loader, device)
        loss_list.append(avg_loss)
        acc_list.append(acc_now)

        if acc_now > acc_max:
            acc_max = acc_now
            acc_max_epoch = epoch

        tqdm.write(f"[Result] Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Test Acc: {acc_now:.4f}")

    print(" [Done] 模型训练完成")
    print(f"总训练用时: {time.time() - start_time:.2f} 秒")

    # 保存模型
    model_path = "mnist_vit.pt"
    torch.save(model.state_dict(), model_path)
    print(f"[Info] 模型已保存为: {model_path}")

    # 绘图
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.plot(range(len(loss_list)), loss_list, "r")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)

    plt.subplot(122)
    plt.plot(range(len(acc_list)), acc_list, "r")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.annotate(f"Max acc: {acc_max:.4f}",
                 xy=(acc_max_epoch, acc_max),
                 xytext=(max(acc_max_epoch * 0.7, 0), acc_max - 0.003),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=12)
    plt.savefig("ViT_plot.png")
    plt.show()
