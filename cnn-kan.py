import os
import warnings
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from modules.cnn_kan import CNNKAN
from modules.kan import KAN
from modules.cnn import CNN

def load_mnist(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    valset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)
    eval_loader = DataLoader(valset, batch_size=len(valset), shuffle=False)
    return trainloader, valloader, eval_loader

def load_cifar10(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    valset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)
    eval_loader = DataLoader(valset, batch_size=len(valset), shuffle=False)
    return trainloader, valloader, eval_loader

def setup_model(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4) # type: ignore
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    return model, optimizer, scheduler, criterion, device

def save_checkpoint(model, optimizer, scheduler, epoch, val_accuracy, filename):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    file_path = os.path.join('checkpoints', filename)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_accuracy': val_accuracy
    }
    torch.save(checkpoint, file_path)

def load_checkpoint(filename, model, optimizer, scheduler):
    file_path = os.path.join('checkpoints', filename)
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'], checkpoint['val_accuracy']

def train_epoch(model, trainloader, optimizer, criterion, device):
    model.train()
    with tqdm(trainloader, desc="Training") as pbar:
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            if isinstance(model, KAN):
                images = images.view(images.size(0), -1)
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            accuracy = (output.argmax(dim=1) == labels).float().mean()
            pbar.set_postfix(loss=f"{loss.item():.4f}", accuracy=f"{accuracy.item():.4f}")
    return loss.item(), accuracy.item() # type: ignore

def validate(model, valloader, criterion, device):
    model.eval()
    val_loss = 0
    val_accuracy = 0
    with torch.no_grad():
        for images, labels in valloader:
            images, labels = images.to(device), labels.to(device)
            if isinstance(model, KAN):
                images = images.view(images.size(0), -1)
            output = model(images)
            val_loss += criterion(output, labels).item()
            val_accuracy += (output.argmax(dim=1) == labels).float().mean().item()
    return val_loss / len(valloader), val_accuracy / len(valloader)

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            if isinstance(model, KAN):
                images = images.view(images.size(0), -1)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def plot_comparison(results, dataset, epochs):
    if not os.path.exists('figures'):
        os.makedirs('figures')

    file_path_train = os.path.join('figures', f'{dataset}_{epochs}_training_comparison.png')
    plt.figure(figsize=(16, 8))

    plt.subplot(1, 2, 1)
    for model, data in results.items():
        plt.plot(np.arange(1, epochs + 1), data['train_losses'], label=f'{model} Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss Comparison ({dataset})')
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.subplot(1, 2, 2)
    for model, data in results.items():
        plt.plot(np.arange(1, epochs + 1), data['train_accuracies'], label=f'{model} Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Training Accuracy Comparison ({dataset})')
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(file_path_train)
    plt.close()

    file_path_val = os.path.join('figures', f'{dataset}_{epochs}_validation_comparison.png')
    plt.figure(figsize=(16, 8))

    plt.subplot(1, 2, 1)
    for model, data in results.items():
        plt.plot(np.arange(1, epochs + 1), data['val_losses'], label=f'{model} Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Validation Loss Comparison ({dataset})')
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.subplot(1, 2, 2)
    for model, data in results.items():
        plt.plot(np.arange(1, epochs + 1), data['val_accuracies'], label=f'{model} Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Validation Accuracy Comparison ({dataset})')
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(file_path_val)
    plt.close()

def main(dataset, models, epochs):
    warnings.filterwarnings("ignore")

    if dataset == 'mnist':
        trainloader, valloader, eval_loader = load_mnist()
        input_size = (1, 28, 28)
    elif dataset == 'cifar10':
        trainloader, valloader, eval_loader = load_cifar10()
        input_size = (3, 32, 32)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    results = {}

    for modelname in models:
        print(f"\nTraining {modelname} on {dataset}")
        if modelname == 'cnn':
            model = CNN(input_size)
        elif modelname == 'kan':
            input_size_flat = input_size[0] * input_size[1] * input_size[2]
            model = KAN([input_size_flat, 64, 10])
        elif modelname == 'cnn-kan':
            model = CNNKAN(input_size)
        else:
            raise ValueError(f"Unknown model: {modelname}")

        model, optimizer, scheduler, criterion, device = setup_model(model)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{modelname} parameters: {total_params}")

        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        best_val_accuracy = 0

        start_time = time.time()
        filename = f'{modelname}_{dataset}_{epochs}_checkpoint.pth'
        for epoch in range(epochs):
            train_loss, train_acc = train_epoch(model, trainloader, optimizer, criterion, device)
            val_loss, val_accuracy = validate(model, valloader, criterion, device)
            scheduler.step()

            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
            print(f"Epoch {epoch + 1}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                save_checkpoint(model, optimizer, scheduler, epoch, val_accuracy, filename)

        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total training time for {modelname}: {total_time:.2f} seconds")

        best_epoch, best_accuracy = load_checkpoint(filename, model, optimizer, scheduler)
        print(f"Loaded best model from epoch {best_epoch+1} with validation accuracy {best_accuracy:.4f}")

        test_accuracy = evaluate(model, eval_loader, device)
        print(f"Test Accuracy for {modelname}: {test_accuracy:.4f}")

        results[modelname] = {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'test_accuracy': test_accuracy
        }

    plot_comparison(results, dataset, epochs)

if __name__ == "__main__":
    main(dataset='cifar10', models=['cnn', 'kan', 'cnn-kan'], epochs=10)
    # main(dataset='mnist', models=['cnn'], epochs=1)