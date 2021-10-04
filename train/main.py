import os
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, models
from torchvision.datasets import ImageFolder

from train.dataset import CustomDataset
from train.utils import save_checkpoint, binary_acc


def resnet50(pretrained=False):
    model = models.resnet50(pretrained=pretrained)
    if pretrained:
        for name, param in model.named_parameters():
            if 'bn' not in name:  # DON'T freeze BN layers
                param.requires_grad = False

    model.fc = nn.Sequential(
        OrderedDict(
            [
                ('dropout1', nn.Dropout(0.5)),
                ('fc1', nn.Linear(2048, 1024)),
                ('activation1', nn.ReLU()),
                ('dropout2', nn.Dropout(0.3)),
                ('fc2', nn.Linear(1024, 256)),
                ('activation2', nn.ReLU()),
                ('dropout3', nn.Dropout(0.3)),
                ('fc3', nn.Linear(256, 128)),
                ('activation3', nn.ReLU()),
                ('fc4', nn.Linear(128, 1))
            ]
        )
    )

    return model


def train():
    model_dir = "output_dir"
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    writer = SummaryWriter(model_dir)

    def send_stats(i, module, input, output):
        writer.add_scalar(f"{i}-mean", output.data.std())
        writer.add_scalar(f"{i}-stddev", output.data.std())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ImageFolder(
        root="/data/mhassan/2d-dataset/images",
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor()
        ])
    )
    test_split = int(len(dataset) * 0.15)
    validation_split = int(len(dataset) * 0.15)
    train_split = len(dataset) - test_split - validation_split
    batch_size = 128
    num_workers = 24
    epochs = 50

    train, test, validation = random_split(dataset, [train_split, test_split, validation_split],
                                           torch.Generator().manual_seed(42))

    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    loaders = {
        "train": DataLoader(
            CustomDataset(
                train,
                transform
            ),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        ),
        "test": DataLoader(
            CustomDataset(
                train,
                transform
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        ),
        "valid": DataLoader(
            CustomDataset(
                train,
                transform
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    }

    model = resnet50(pretrained=True)

    # writer.add_graph(model, torch.rand([1, 3, 224, 224]))
    start_epoch = 1
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # model, optimizer, start_epoch = load_checkpoint(
    #     model,
    #     optimizer,
    #     os.path.join(model_dir, "checkpoint-best.pth.tar")
    # )

    if torch.cuda.device_count() > 1:
        print("Using ", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)

    best_accuracy = 0.0
    for epoch in range(start_epoch, epochs + 1):  # loop over the dataset multiple times
        epoch_loss, epoch_acc = 0., 0.
        running_loss, running_acc = 0., 0.
        model.train()
        for i, (images, labels) in enumerate(loaders['train'], 0):
            images, labels = images.to(device), labels.to(torch.float).to(device).unsqueeze(1)
            # add to tensorboard
            # grid = torchvision.utils.make_grid(images[:8])
            # writer.add_image('train/images', grid, epoch)
            # writer.add_graph(model, inputs)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            acc = binary_acc(outputs, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            running_loss += loss.item()
            running_acc += acc.item()
            if i % 10 == 0:
                print(f'iter: {i:04}: Running loss: {running_loss / 10:.3f} | Running acc: {running_acc / 10:.3f}')
                running_loss = 0.0
                running_acc = 0.0

        model.eval()
        val_accuracy = 0.0
        val_loss = 0.0
        with torch.no_grad():
            for i, (images, labels) in enumerate(loaders['valid'], 0):
                images, labels = images.to(device), labels.to(torch.float).to(device).unsqueeze(1)
                # add to tensorboard
                # grid = torchvision.utils.make_grid(images[:8])
                # writer.add_image('validation/images', grid, epoch)

                outputs = model(images)
                loss = criterion(outputs, labels)
                acc = binary_acc(outputs, labels)
                val_accuracy += acc.item()
                val_loss += loss.item()

        val_accuracy = val_accuracy / len(loaders['valid'])
        val_loss = val_loss / len(loaders['valid'])
        is_best = bool(val_accuracy > best_accuracy)
        best_accuracy = max(val_accuracy, best_accuracy)
        # Save checkpoint if is a new best
        save_checkpoint(
            {
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'state_dict': model.module.state_dict(),
                'loss': val_loss,
                'best_accuracy': best_accuracy
            },
            is_best,
            filename=os.path.join(model_dir, f'checkpoint-{epoch:03d}-val-{val_accuracy:.3f}.pth.tar')
        )

        train_loss = epoch_loss / len(loaders['train'])
        train_acc = epoch_acc / len(loaders['train'])
        print(
            f'Epoch {epoch:03}: Loss: {train_loss:.3f} | Acc:'
            f' {train_acc:.3f} | Val Loss: {val_loss:.3f} | Val Acc: {best_accuracy:.3f}'
        )

        # add to tensorboard
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/accuracy", train_acc, epoch)
        writer.add_scalar("validation/loss", val_loss, epoch)
        writer.add_scalar("validation/accuracy", val_accuracy, epoch)

        for name, weight in model.named_parameters():
            writer.add_histogram(name, weight, epoch)
            writer.add_histogram(f'{name}.grad', weight, epoch)

        for i, m in enumerate(model.children()):
            m.register_forward_hook(partial(send_stats, i))

        writer.flush()

    print('Finished Training.')
