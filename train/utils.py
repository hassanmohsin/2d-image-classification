import os

import torch


def get_mean_std(loader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for _, data, _ in loader:
        channels_sum += torch.mean(data, dim=(0, 2, 3))
        channels_squared_sum += torch.mean(data ** 2, dim=(0, 2, 3))
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


# TODO: Enable loading epoch, loss and accuracy also
def load_checkpoint(model, optimizer, filename='checkpoint-best.pth.tar'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 1
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch


def save_checkpoint(state, is_best, filename):
    weight_dir = os.path.dirname(filename)
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print("=> Saving a new best")
        torch.save(state, os.path.join(weight_dir, "checkpoint-best.pth.tar"))  # save checkpoint
    else:
        print("=> Validation Accuracy did not improve")
    torch.save(state, filename)


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc
