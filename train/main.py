import os
from argparse import ArgumentParser
from collections import OrderedDict
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torchmetrics
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, models
from torchvision.datasets import ImageFolder

from train.dataset import CustomDataset
from train.utils import save_checkpoint, load_checkpoint, FocalLoss, optimizer_to


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
    parser = ArgumentParser("Train models.")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory containing positive and negative images")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory to save the models")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--loss", type=str, default="bce", help="Loss function. bce or focal")
    parser.add_argument("--num_workers", type=int, default=12, help="Number of workers")
    parser.add_argument("--test_split", type=float, default=0.15, help="Fraction of the dataset to use as test set")
    parser.add_argument("--valid_split", type=float, default=0.15,
                        help="Fraction of the dataset to use as validation set")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--eval", action="store_true", default=False, help="Evaluate the test set.")
    args = parser.parse_args()

    if not os.path.isdir(args.model_dir):
        os.makedirs(args.model_dir)
    writer = SummaryWriter(args.model_dir)

    def send_stats(i, module, input, output):
        writer.add_scalar(f"{i}-mean", output.data.std())
        writer.add_scalar(f"{i}-stddev", output.data.std())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ImageFolder(root=args.image_dir)
    test_split = int(len(dataset) * args.test_split)
    validation_split = int(len(dataset) * args.valid_split)
    train_split = len(dataset) - test_split - validation_split
    batch_size = args.batch_size
    num_workers = args.num_workers
    epochs = args.epochs

    train, test, validation = random_split(dataset, [train_split, test_split, validation_split],
                                           torch.Generator().manual_seed(42))

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    loaders = {
        "train": DataLoader(
            CustomDataset(
                train,
                transform=transforms.Compose(
                    [
                        transforms.Resize(256),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomVerticalFlip(p=0.5),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]
                        )
                    ]
                )
            ),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        ),
        "test": DataLoader(
            CustomDataset(
                test,
                transform
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        ),
        "valid": DataLoader(
            CustomDataset(
                validation,
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
    if args.loss == "bce":
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss == "focal":
        criterion = FocalLoss()
    else:
        criterion = None

    assert criterion is not None

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    model, optimizer, start_epoch, best_f1, best_acc = load_checkpoint(
        model,
        optimizer,
        os.path.join(args.model_dir, "checkpoint-best.pth.tar")
    )

    if torch.cuda.device_count() > 1:
        print("Using ", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)
    optimizer_to(optimizer, device)
    print(f"Initial values: start epoch: {start_epoch + 1}, best_f1: {best_f1:.3f}, best_acc: {best_acc:.3f}")

    if args.eval:
        # Load the model checkpoint
        # checkpoint = torch.load(os.path.join(args.model_dir, "checkpoint-best.pth.tar"))
        # model.load_state_dict(checkpoint["state_dict"])

        acc_metric = torchmetrics.Accuracy().to(device)
        f1_metric = torchmetrics.F1().to(device)
        cf_metric = torchmetrics.ConfusionMatrix(num_classes=2).to(device)

        model.eval()
        all_y = []
        all_pred = []
        indices = []
        with torch.no_grad():
            for i, (idxs, images, labels) in enumerate(loaders['train'], 0):
                images, labels = images.to(device), labels.to(torch.int).to(device).unsqueeze(1)
                outputs = model(images)
                pred_proba = torch.sigmoid(outputs)
                pred_labels = torch.round(pred_proba).to(torch.int)
                ids = idxs.numpy().tolist()
                indices += [os.path.split(f[0])[1] for f in [dataset.samples[i] for i in ids]]
                all_y += labels.cpu().numpy().tolist()
                all_pred += pred_proba.cpu().numpy().tolist()
                acc = acc_metric(pred_labels, labels)
                f1 = f1_metric(pred_labels, labels)
                cf = cf_metric(pred_labels, labels)
                # print(f"Current batch acc: {acc}")
        np.save(os.path.join(args.model_dir, 'y'), np.array(all_y))
        np.save(os.path.join(args.model_dir, 'y_pred'), np.array(all_pred))
        np.save(os.path.join(args.model_dir, 'indices'), np.array(indices))
        test_accuracy = acc_metric.compute()
        test_f1 = f1_metric.compute()
        test_cf = cf_metric.compute().cpu().numpy()
        np.save(os.path.join(args.model_dir, 'cf'), test_cf)
        print(f"Test accuracy: {test_accuracy}\nTest F1: {test_f1}\nConfusion matrix: {test_cf}")

        # Reset metrics
        acc_metric.reset()
        f1_metric.reset()
        cf_metric.reset()
        return

    for epoch in range(start_epoch + 1, epochs + 1):  # loop over the dataset multiple times
        train_f1_metric = torchmetrics.F1().to(device)
        train_acc_metric = torchmetrics.Accuracy().to(device)

        running_acc, running_f1 = 0., 0.
        epoch_loss, running_loss = 0., 0.
        model.train()
        for i, (idxs, images, labels) in enumerate(loaders['train'], 0):
            images, labels = images.to(device), labels.to(torch.float).to(device).unsqueeze(1)
            # add to tensorboard
            # grid = torchvision.utils.make_grid(images[:8])
            # writer.add_image('train/images', grid, epoch)
            # writer.add_graph(model, inputs)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            pred_proba = torch.sigmoid(outputs)
            pred_labels = torch.round(pred_proba).to(torch.int)
            epoch_loss += loss.item()
            running_loss += loss.item()
            labels = labels.to(torch.int)
            running_f1 += train_f1_metric(pred_labels, labels)
            running_acc += train_acc_metric(pred_labels, labels)
            if i % 10 == 0:
                print(
                    f'Iter: {i:04}: Running loss: {running_loss / 10:.3f} | Running acc: {running_acc / 10:.3f} | Running f1: {running_f1 / 10:.3f}')
                running_loss = 0.
                running_f1 = 0.
                running_acc = 0.

        train_loss = epoch_loss / len(loaders['train'])
        train_acc = train_acc_metric.compute()
        train_f1 = train_f1_metric.compute()

        # reset
        train_f1_metric.reset()
        train_acc_metric.reset()

        model.eval()

        val_acc_metric = torchmetrics.Accuracy().to(device)
        val_f1_metric = torchmetrics.F1().to(device)
        val_loss = 0.0
        with torch.no_grad():
            for i, (idxs, images, labels) in enumerate(loaders['valid'], 0):
                images, labels = images.to(device), labels.to(torch.float).to(device).unsqueeze(1)
                # add to tensorboard
                # grid = torchvision.utils.make_grid(images[:8])
                # writer.add_image('validation/images', grid, epoch)

                outputs = model(images)
                loss = criterion(outputs, labels)
                pred_proba = torch.sigmoid(outputs)
                pred_labels = torch.round(pred_proba).to(torch.int)
                labels = labels.to(torch.int)
                acc = val_acc_metric(pred_labels, labels)
                f1 = val_f1_metric(pred_labels, labels)
                val_loss += loss.item()

        val_acc = val_acc_metric.compute()
        val_f1 = val_f1_metric.compute()
        val_loss = val_loss / len(loaders['valid'])
        best_f1 = max(best_f1, val_f1)
        is_best = bool(val_acc > best_acc)
        best_acc = max(best_acc, val_acc)

        val_acc_metric.reset()
        val_f1_metric.reset()
        # Save checkpoint if is a new best
        save_checkpoint(
            {
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'state_dict': model.module.state_dict(),
                'loss': val_loss,
                'best_f1': best_f1,
                'best_acc': best_acc
            },
            is_best,
            filename=os.path.join(args.model_dir, f'checkpoint-{epoch:03d}-val-{val_acc:.3f}.pth.tar')
        )

        print(
            f'Epoch {epoch:03}: | Train: Loss: {train_loss:.3f}, Acc: {train_acc:.3f}, F1: {train_f1:.3f} | '
            f'Val: Loss: {val_loss:.3f}, Acc: {val_acc:.3f}, F1: {val_f1:.3f} | '
            f'Best: Acc: {best_acc:.3f}, F1: {best_f1:.3f}'
        )

        # add to tensorboard
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/accuracy", train_acc, epoch)
        writer.add_scalar("train/f1", train_f1, epoch)
        writer.add_scalar("validation/loss", val_loss, epoch)
        writer.add_scalar("validation/accuracy", val_acc, epoch)
        writer.add_scalar("validation/f1", val_f1, epoch)

        for name, weight in model.named_parameters():
            writer.add_histogram(name, weight, epoch)
            writer.add_histogram(f'{name}.grad', weight, epoch)

        for i, m in enumerate(model.children()):
            m.register_forward_hook(partial(send_stats, i))

        writer.flush()

    print('Finished Training.')


if __name__ == '__main__':
    train()
