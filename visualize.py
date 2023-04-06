# ========================================================
#             Media and Cognition
#             Homework 2 Convolutional Neural Network
#             visual.py - visualization
#             Student ID:
#             Name:
#             Tsinghua University
#             (C) Copyright 2023
# ========================================================

import argparse
import copy
import os
import string

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
from torch.autograd import Variable
from torch.nn import MaxPool2d
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid

from datasets import get_data_loader
from networks import Classifier


class ConvFilterVisualization:

    def __init__(self, model, save_dir):
        self.model = model
        self.model.eval()
        self.save_dir = save_dir
        self.conv_output = None

    def hook_layer(self, layer_idx, filter_idx):
        def hook_function(module, input, output):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = output[0, filter_idx]

        # Hook the selected layer
        self.hook = self.model[layer_idx][2].register_forward_hook(hook_function)

    def visualize(self, conv_layer_indices, layer_idx, filter_idx, opt_steps, upscaling_steps=4, upscaling_factor=1.2,
                  blur=None):
        # Hook the selected layer
        self.hook_layer(conv_layer_indices[layer_idx], filter_idx)
        im_size = 32
        x = torch.rand(1, 3, im_size, im_size, requires_grad=True) * 2 - 1
        for _ in range(upscaling_steps):
            x = Variable(x, requires_grad=True)

            optimizer = torch.optim.Adam([x], lr=0.1, weight_decay=1e-6)
            for n in range(opt_steps):
                optimizer.zero_grad()
                self.model(x)
                loss = - self.conv_output.mean()
                loss.backward()
                optimizer.step()
            image = 255 * (x * 0.5 + 0.5).squeeze(0).permute(1, 2, 0).detach().numpy()
            im_size = int(upscaling_factor * im_size)  # calculate new image size
            x = cv2.resize(image, (im_size, im_size), interpolation=cv2.INTER_CUBIC)  # scale image up
            x = np.clip((x / 255 - 0.5) * 2, -1, 1)
            x = torch.from_numpy(x)
            x.requires_grad = True
            x = x.view(1, 3, im_size, im_size)
        if blur is not None:
            image = cv2.blur(image, (blur, blur))
        save_dir = os.path.join(self.save_dir, 'layer_%d' % layer_idx)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        cv2.imwrite(os.path.join(save_dir, 'filter_%d.jpg' % filter_idx), np.clip(image, 0, 255))
        self.hook.remove()
        return image / 255


class ConvFeatureVisualization:

    def __init__(self, model, save_dir):
        self.model = model
        self.model.eval()
        self.save_dir = save_dir
        self.conv_output = None

    def hook_layer(self, layer_idx):
        def hook_function(module, input, output):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = output[0]

        # Hook the selected layer
        self.hook = self.model[layer_idx][2].register_forward_hook(hook_function)

    def visualize(self, conv_layer_indices, layer_idx, image):
        self.hook_layer(conv_layer_indices[layer_idx])
        self.model(image)
        save_dir = os.path.join(self.save_dir, 'layer_%d' % layer_idx)
        w = 16
        h = int(self.conv_output.shape[0] / w)
        fig, axes = plt.subplots(h, w, figsize=(w / 1.6, h))
        plt.suptitle('output feature map of layer %d' % layer_idx)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for i in range(self.conv_output.shape[0]):
            x = self.conv_output[i].detach().numpy()
            x = ((x - x.min()) / (x.max() - x.min())) if x.max() > x.min() else (x - x.min())
            x = cv2.resize(x, (32, 32), interpolation=cv2.INTER_CUBIC)
            axes[i // w, i % w].imshow(x, cmap='rainbow')
            axes[i // w, i % w].set_title(str(i), fontsize='small')
            axes[i // w, i % w].axis('off')
            cv2.imwrite(os.path.join(save_dir, 'channel_%d.jpg' % i), 255 * x)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'feature_map.jpg'), dpi=200)
        plt.show()
        print('Results are saved as {}'.format(os.path.join(save_dir, 'feature_map.jpg')))
        self.hook.remove()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # set configurations of the visualization process
    parser.add_argument('--path', type=str, default='data', help='path to data file')
    parser.add_argument('--epoch', type=int, default=15, help='epoch of checkpoint you want to load')
    parser.add_argument('--ckpt_path', type=str, default='ckpt', help='path to load checkpoints')
    parser.add_argument('--type', type=str, default='filter', choices=['filter', 'feature', 'tsne', 'stn'],
                        help='type of visualized data, can be filter, feature and tsne')
    parser.add_argument('--layer_idx', type=int, default=0, help='index of convolutional layer for visualizing filter and feature')
    parser.add_argument('--image_idx', type=int, default=128, help='index of images for visualizing feature')
    parser.add_argument('--save_dir', type=str, default='visualized/', help='directory to save visualization results')

    opt = parser.parse_args()
    if not os.path.exists(opt.save_dir):
        os.mkdir(opt.save_dir)
    print('[Info] loading checkpoint from %s ...' % os.path.join(opt.ckpt_path, 'ckpt_epoch_%d.pth' % opt.epoch))
    checkpoint = torch.load(os.path.join(opt.ckpt_path, 'ckpt_epoch_%d.pth' % opt.epoch))
    configs = checkpoint['configs']
    model = Classifier(configs['in_channels'], configs['num_class'], configs['use_batch_norm'],
                       configs['use_stn'], configs['dropout_prob'])
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    stn = model.stn
    conv_net = torch.nn.ModuleList()
    for name, m in model.named_children():
        if name != 'fc_net' and name != 'stn':
            conv_net.append(m)
    conv_net = torch.nn.Sequential(*conv_net)
    fc_net = model.fc_net

    if opt.type == 'filter':
        filter_dir = os.path.join(opt.save_dir, 'filter')
        if not os.path.exists(filter_dir):
            os.mkdir(filter_dir)

        conv_layer_indices = []
        filter_nums = []
        for i, m in enumerate(conv_net.children()):
            if not isinstance(m, MaxPool2d):
                conv_layer_indices.append(i)
                filter_nums.append(m[0].out_channels)

        visual = ConvFilterVisualization(conv_net, filter_dir)
        w = 16
        h = int(filter_nums[opt.layer_idx] / w)
        fig, axes = plt.subplots(h, w, figsize=(w / 1.6, h))
        plt.suptitle('conv filters of layer %d' % opt.layer_idx)
        for i in range(filter_nums[opt.layer_idx]):
            x = visual.visualize(conv_layer_indices, opt.layer_idx, i, 30, blur=None)
            axes[i // w, i % w].imshow(x[:, :, 0], cmap='rainbow')
            axes[i // w, i % w].set_title(str(i), fontsize='small')
            axes[i // w, i % w].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(opt.save_dir, 'filter', 'filter_layer_%d.jpg' % opt.layer_idx), dpi=200)
        plt.show()
        print('Results are saved as {}'.format(
            os.path.join(opt.save_dir, 'filter', 'filter_layer_%d.jpg' % opt.layer_idx)))

    elif opt.type == 'feature':
        feature_dir = os.path.join(opt.save_dir, 'feature')
        if not os.path.exists(feature_dir):
            os.mkdir(feature_dir)

        conv_layer_indices = []
        for i, m in enumerate(conv_net.children()):
            if not isinstance(m, MaxPool2d):
                conv_layer_indices.append(i)

        visual = ConvFeatureVisualization(conv_net, feature_dir)
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])

        dataset = ImageFolder(os.path.join(opt.path, 'train'), transform=transform)
        img_idx = opt.image_idx
        image, _ = dataset[img_idx]
        image_out = 255 * (image / 2 + 0.5).permute(1, 2, 0).detach().numpy()
        image_out = cv2.cvtColor(image_out, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(feature_dir, 'image.jpg'), image_out)
        # print(image.shape)
        visual.visualize(conv_layer_indices, opt.layer_idx, image.unsqueeze(0))

    elif opt.type == 'tsne':
        tsne_dir = os.path.join(opt.save_dir, 'tsne')
        if not os.path.exists(tsne_dir):
            os.mkdir(tsne_dir)

        data_loader = get_data_loader(opt.path, 'train', image_size=(32, 32), batch_size=8)
        labels = []
        features = []
        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.float(), y.long()
                x = stn(x)
                x = conv_net(x)
                x = x.contiguous().view(x.shape[0], -1)
                x = fc_net[0](x)
                x = fc_net[1](x)
                features.append(copy.deepcopy(x.detach()))
                labels.append(copy.deepcopy(y))
            features = torch.cat(features, dim=0)
            labels = torch.cat(labels, dim=0)
            Y = TSNE(n_components=2, init='pca', random_state=0, learning_rate='auto').fit_transform(features[:800].numpy())
            labels = labels[:800].numpy()

        letters = list(string.ascii_letters[-26:])
        Y = (Y - Y.min(0)) / (Y.max(0) - Y.min(0))
        for i in range(len(labels)):
            c = plt.cm.rainbow(float(labels[i]) / 26)
            plt.text(Y[i, 0], Y[i, 1], s=letters[labels[i]], color=c)
        plt.savefig(os.path.join(tsne_dir, 'tsne.jpg'), dpi=300)
        plt.show()
        print('Results are saved as {}'.format(os.path.join(tsne_dir, 'tsne.jpg')))

    else:
        stn_dir = os.path.join(opt.save_dir, 'stn')
        if not os.path.exists(stn_dir):
            os.mkdir(stn_dir)

        data_loader = get_data_loader(opt.path, 'train', image_size=(32, 32), batch_size=16)
        labels = []
        features = []
        with torch.no_grad():
            x, y = next(iter(data_loader))
            x_transformed = stn(x)

        img_original = make_grid((x + 1) / 2).cpu().numpy().transpose(1, 2, 0)
        img_transformed = make_grid((x_transformed + 1) / 2).cpu().numpy().transpose(1, 2, 0)
        fig, axes = plt.subplots(2, 1, figsize=(6, 4))
        plt.suptitle('The Effect of the Spatial Transformer Network')
        axes[0].imshow(img_original)
        axes[0].set_title('original')
        axes[0].axis('off')
        axes[1].imshow(img_transformed)
        axes[1].set_title('transformed')
        axes[1].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(opt.save_dir, 'stn', 'stn.jpg'), dpi=200)
        plt.show()
        print('Results are saved as {}'.format(os.path.join(opt.save_dir, 'stn', 'stn.jpg')))
