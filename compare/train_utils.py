import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            predicted = model(X).argmax(dim=1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total


def train_deep(model, train_loader, test_loader, device,
               num_epochs=10, lr=1e-3, desc=''):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    history = {'loss': [], 'acc': []}
    prefix = f'[{desc}] ' if desc else ''
    start = time.time()
    for epoch in range(num_epochs):
        model.train()
        running = 0.0
        loop = tqdm(train_loader, desc=f'{prefix}Epoch {epoch + 1}/{num_epochs}', leave=False)
        for X, y in loop:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            running += loss.item()
        scheduler.step()
        acc = evaluate(model, test_loader, device)
        history['loss'].append(running / max(len(train_loader), 1))
        history['acc'].append(acc)
        tqdm.write(f'{prefix}[Result] Epoch {epoch + 1}/{num_epochs}, '
                   f'Loss: {history["loss"][-1]:.4f}, Test Acc: {acc:.4f}')
    elapsed = time.time() - start
    return model, history, elapsed


def plot_curves(history, title, out_path):
    acc_max = max(history['acc'])
    acc_max_epoch = history['acc'].index(acc_max)
    plt.figure(figsize=(12, 5))

    plt.subplot(121)
    plt.plot(range(len(history['loss'])), history['loss'], 'r')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.title('Training Loss')

    plt.subplot(122)
    plt.plot(range(len(history['acc'])), history['acc'], 'b')
    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracy')
    plt.grid(True)
    plt.title('Test Accuracy')
    plt.annotate(f'Max acc: {acc_max:.4f}',
                 xy=(acc_max_epoch, acc_max),
                 xytext=(0.40, 0.10), textcoords='axes fraction',
                 ha='center', fontsize=10,
                 arrowprops=dict(facecolor='black', shrink=0.10,
                                 width=1.5, headwidth=7))
    plt.suptitle(title)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
