# ========================================================
#             Media and Cognition
#             Homework 2 Convolutional Neural Network
#             test.py - test our model for character classification
#             Student ID:
#             Name:
#             Tsinghua University
#             (C) Copyright 2023
# ========================================================
import argparse
import os
import string

import cv2
import torch

from datasets import get_data_loader
from networks import Classifier


def test(data_root, ckpt_path, epoch, save_results, device='cpu'):
    """
    The main testing procedure
    ----------------------------
    :param data_root: path to the root directory of dataset
    :param ckpt_path: path to load checkpoints
    :param epoch: epoch of checkpoint you want to load
    :param save_results: whether to save results
    :param device: 'cpu' or 'cuda', we can use 'cpu' for our homework if GPU with cuda support is not available
    """

    if save_results:
        save_dir = os.path.join(ckpt_path, 'results')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
    # construct testing data loader
    test_loader = get_data_loader(data_root, 'test', image_size=(32, 32), batch_size=1)

    print('[Info] loading checkpoint from %s ...' % os.path.join(ckpt_path, 'ckpt_epoch_%d.pth' % epoch))
    checkpoint = torch.load(os.path.join(ckpt_path, 'ckpt_epoch_%d.pth' % epoch))
    configs = checkpoint['configs']
    model = Classifier(configs['in_channels'], configs['num_class'], configs['use_batch_norm'],
                       configs['use_stn'], configs['dropout_prob'])
    # load model parameters (checkpoint['model_state']) we saved in model_path using model.load_state_dict()
    model.load_state_dict(checkpoint['model_state'])
    # put the model on CPU or GPU
    model = model.to(device)

    # enter the evaluation mode
    model.eval()
    correct = 0
    n = 0
    letters = string.ascii_letters[-26:]
    for input, label in test_loader:
        # set data type and device
        input, label = input.type(torch.float).to(device), label.type(torch.long).to(device)
        # get the prediction result
        pred = model(input)
        pred = torch.argmax(pred, dim=-1)
        label = label.squeeze(dim=0)

        # set the name of saved images to 'idx_correct/wrong_label_pred.jpg'
        if pred == label:
            correct += 1
            save_name = '%04d_correct_%s_%s.jpg' % (n, letters[int(label)], letters[int(pred)])
        else:
            save_name = '%04d_wrong_%s_%s.jpg' % (n, letters[int(label)], letters[int(pred)])

        if save_results:
            img = 255 * (input * 0.5 + 0.5).squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(save_dir, save_name), img)

        n += 1
    # calculate accuracy
    accuracy = float(correct) / float(len(test_loader))
    print('accuracy on the test set: %.3f' % accuracy)

    if save_results:
        print('results saved to %s' % save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # set configurations of the testing process
    parser.add_argument('--path', type=str, default='data', help='path to data file')
    parser.add_argument('--epoch', type=int, default=15, help='epoch of checkpoint you want to load')
    parser.add_argument('--ckpt_path', type=str, default='ckpt', help='path to load checkpoints')
    parser.add_argument('--save', action='store_true', default=False, help='whether to save results')
    parser.add_argument('--device', type=str, help='cpu or cuda')

    opt = parser.parse_args()
    if opt.device is None:
        opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # run the testing procedure
    test(data_root=opt.path,
         ckpt_path=opt.ckpt_path,
         epoch=opt.epoch,
         save_results=opt.save,
         device=opt.device)
