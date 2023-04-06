# ========================================================
#             Media and Cognition
#             Homework 2 Convolutional Neural Network
#             train.py - train traffic sign classification model
#             Student ID:
#             Name:
#             Tsinghua University
#             (C) Copyright 2023
# ========================================================

# ==== Part 1: import libs
import argparse  # argparse is used to conveniently set our configurations
import glob
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Import our data loader function and Classifier model defined in other files
from datasets import get_data_loader
from networks import Classifier


# ==== Part 2: training and validation
def train(data_root, augment, in_channels, num_class,
          batch_norm, dropout, stn, n_epochs, batch_size, lr,
          momentum, weight_decay, optim_type, ckpt_path, max_ckpt_save_num,
          ckpt_save_interval, val_interval, resume, device='cpu'):
    """
    The main training procedure
    ----------------------------
    :param data_root: path to the root directory of dataset
    :param augment: whether to use data augmentation
    :param in_channels: channel number of image
    :param num_class: number of classes, in this task it is 26 English letters
    :param batch_norm: whether to use batch normalization in convolutional layers and linear layers
    :param dropout: dropout ratio of dropout layer which ranges from 0 to 1
    :param stn: whether to use spatial transformer network
    :param n_epochs: number of training epochs
    :param batch_size: batch size of training
    :param lr: learning rate
    :param momentum: only used if optim_type == 'sgd'
    :param weight_decay: the factor of L2 penalty on network weights
    :param optim_type: optimizer, which can be set as 'sgd', 'adagrad', 'rmsprop', 'adam', or 'adadelta'
    :param ckpt_path: path to save checkpoint models
    :param max_ckpt_save_num: maximum number of saving checkpoint models
    :param ckpt_save_interval: intervals of saving checkpoint models, e.g., if ckpt_save_interval = 2, then save checkpoint models every 2 epochs
    :param val_interval: intervals of validation, e.g., if val_interval = 5, then do validation after each 5 training epochs
    :param resume: path to resume model
    :param device: 'cpu' or 'cuda', we can use 'cpu' for our homework if GPU with cuda support is not available
    """

    # construct training and validation data loader
    train_loader = get_data_loader(data_root, 'train', image_size=(32, 32), batch_size=batch_size,
                                   num_workers=2, augment=augment)
    val_loader = get_data_loader(data_root, 'val', image_size=(32, 32), batch_size=batch_size,
                                 num_workers=2)

    model = Classifier(in_channels, num_class, batch_norm, stn, dropout_prob=dropout)

    # put the model on CPU or GPU
    model = model.to(device)

    # define loss function and optimizer
    loss_func = nn.CrossEntropyLoss()

    if optim_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
    elif optim_type == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr, weight_decay=weight_decay)
    elif optim_type == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr, weight_decay=weight_decay)
    elif optim_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr, weight_decay=weight_decay)
    elif optim_type == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr, weight_decay=weight_decay)
    else:
        print('[Error] optim_type should be one of sgd, adagrad, rmsprop, adam, or adadelta')
        raise NotImplementedError

    if resume is not None:
        print(f'[Info] resuming model from {resume} ...')
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])

    # training
    # to save loss of each training epoch in a python "list" data structure
    losses = []
    # to save accuracy on validation set of each training epoch in a python "list" data structure
    accuracy_list = []
    val_epochs = []

    print('training...')
    for epoch in range(n_epochs):
        # set the model in training mode
        model.train()

        # to save total loss in one epoch
        total_loss = 0.

        for step, (input, label) in enumerate(train_loader):  # get a batch of data

            # set data type and device
            input, label = input.type(torch.float).to(device), label.type(torch.long).to(device)

            # clear gradients in the optimizer
            optimizer.zero_grad()

            # run the model which is the forward process
            out = model(input)

            # compute the CrossEntropy loss, and call backward propagation function
            loss = loss_func(out, label)
            loss.backward()

            # update parameters of the model
            optimizer.step()

            # sum up of total loss, loss.item() return the value of the tensor as a standard python number
            # this operation is not differentiable
            total_loss += loss.item()

        # average of the total loss for iterations
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)

        # evaluate model on validation set
        if (epoch + 1) % val_interval == 0:
            val_accuracy = eval_one_epoch(model, val_loader, device)
            accuracy_list.append(val_accuracy)
            val_epochs.append(epoch)
            print('Epoch {:02d}: loss = {:.3f}, accuracy on validation set = {:.3f}'.format(epoch + 1, avg_loss,
                                                                                            val_accuracy))

        if (epoch + 1) % ckpt_save_interval == 0:
            # get info of all saved checkpoints
            ckpt_list = glob.glob(os.path.join(ckpt_path, 'ckpt_epoch_*.pth'))
            # sort checkpoints by saving time
            ckpt_list.sort(key=os.path.getmtime)
            # remove surplus ckpt file if the number is larger than max_ckpt_save_num
            if len(ckpt_list) >= max_ckpt_save_num:
                for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                    os.remove(ckpt_list[cur_file_idx])

            # save model parameters in a file
            ckpt_name = os.path.join(ckpt_path, 'ckpt_epoch_%d.pth' % (epoch + 1))
            save_dict = {'model_state': model.state_dict(),
                         'optimizer_state': optimizer.state_dict(),
                         'configs': {
                             'in_channels': in_channels,
                             'num_class': num_class,
                             'use_batch_norm': batch_norm,
                             'use_stn': stn,
                             'dropout_prob': dropout}
                         }

            torch.save(save_dict, ckpt_name)
            print('Model saved in {}\n'.format(ckpt_name))

    plot(losses, accuracy_list, val_epochs, ckpt_path)


def eval_one_epoch(model, val_loader, device):
    """
    Evaluate model performance.
    --------------------------
    :param model: model
    :param val_loader: validation dataloader
    :param device: 'cpu' or 'cuda'
    :return accuracy: performance of model
    """

    # enter the evaluation mode
    model.eval()
    correct = 0  # number of images that are correctly classified
    n_samples = 0
    with torch.no_grad():  # we do not need to compute gradients during validation
        for input, label in val_loader:
            # set data type and device
            input, label = input.type(torch.float).to(device), label.type(torch.long).to(device)
            # get the prediction result
            pred = model(input)
            pred = torch.argmax(pred, dim=-1)
            correct += torch.sum(pred == label).item()
            n_samples += len(label)

        # calculate accuracy
        accuracy = correct / n_samples

    return accuracy


def plot(losses, accuracy_list, val_epochs, ckpt_path):
    """
    Draw loss and accuracy curve
    ------------------
    :param losses: a list with loss of each training epoch
    :param accuracy_list: a list with accuracy on validation set of each training epoch
    """

    # create a plot
    f, ax = plt.subplots()

    # draw loss
    ax.plot(losses)
    ax.plot(val_epochs, accuracy_list, 'r')

    # set labels
    ax.set_xlabel('training epoch')
    ax.legend(['loss', 'accuracy'])

    # show the image
    plt.savefig(os.path.join(ckpt_path, 'loss_and_acc.jpg'), dpi=300)
    plt.show()


if __name__ == '__main__':
    # set random seed for reproducibility
    seed = 2023
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    # set configurations of the model and training process
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data', help='file list of training image paths and labels')
    parser.add_argument('--augment', action='store_true', help='whether to use data augmentation')
    parser.add_argument('--epoch', type=int, default=15, help='number of training epochs')
    parser.add_argument('--batchsize', type=int, default=32, help='training batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum of SGD optimizer')
    parser.add_argument('--weight_decay', type=float, default=0, help='the factor of L2 penalty on network weights')
    parser.add_argument('--optim_type', type=str, default='adam',
                        help='type of optimizer, can be sgd, adagrad, rmsprop, adam, or adadelta')
    parser.add_argument('--bn', action='store_true', help='whether to use batch normalization')
    parser.add_argument('--stn', action='store_true', help='whether to use spatial transformer network')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout ratio')
    parser.add_argument('--ckpt_path', type=str, default='checkpoints/default', help='path to save checkpoints')
    parser.add_argument('--max_ckpt_save_num', type=int, default=10, help='maximum number of saving checkpoints')
    parser.add_argument('--val_interval', type=int, default=1, help='intervals of validation')
    parser.add_argument('--resume', type=str, default=None, help='path to resume model')
    parser.add_argument('--device', type=str, help='cpu or cuda')

    opt = parser.parse_args()
    if opt.device is None:
        opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs(opt.ckpt_path, exist_ok=True)

    # run the training procedure
    train(data_root=opt.data_root,
          augment=opt.augment,
          in_channels=3,
          num_class=26,
          batch_norm=opt.bn,
          dropout=opt.dropout,
          stn=opt.stn,
          n_epochs=opt.epoch,
          batch_size=opt.batchsize,
          lr=opt.lr,
          momentum=opt.momentum,
          weight_decay=opt.weight_decay,
          optim_type=opt.optim_type,
          ckpt_path=opt.ckpt_path,
          max_ckpt_save_num=opt.max_ckpt_save_num,
          ckpt_save_interval=1,
          val_interval=opt.val_interval,
          resume=opt.resume,
          device=opt.device)
