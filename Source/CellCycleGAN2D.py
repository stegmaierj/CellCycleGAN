#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# CellCycleGAN.
# Copyright (C) 2020 D. Bähr, D. Eschweiler, A. Bhattacharyya, 
# D. Moreno-Andrés, W. Antonin, J. Stegmaier
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the Liceense at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Please refer to the documentation for more information about the software
# as well as for installation instructions.
#
# If you use this application for your work, please cite the repository and one
# of the following publications:
#
# D. Bähr, D. Eschweiler, A. Bhattacharyya, D. Moreno-Andrés, W. Antonin, J. Stegmaier, 
# "CellCycleGAN: Spatiotemporal Microscopy Image Synthesis of Cell
# Populations using Statistical Shape Models and Conditional GANs", arxiv,
# 2020.
#
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
from argparse import ArgumentParser
from collections import OrderedDict
from torch.utils.data import DataLoader
from CellCycleGANDataLoader import CCGH5DataLoader

# specify the generator network
class Generator(nn.Module):
    
    def __init__(self, patch_size, in_channels, out_channels, feat_channels=32):
        super(Generator, self).__init__()
        
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feat_channels = feat_channels
        
        # Define layer instances       
        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels//2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(feat_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels//2, feat_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(feat_channels),
            nn.ReLU(inplace=True)
            )
        self.c1rec = nn.Sequential(
            nn.Conv2d(out_channels, feat_channels//2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(feat_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels//2, feat_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(feat_channels),
            nn.ReLU(inplace=True)
            )
        self.d1 = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(feat_channels),
            nn.ReLU(inplace=True)
            )

        self.c2 = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, feat_channels*2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(feat_channels*2),
            nn.ReLU(inplace=True)
            )
        self.d2 = nn.Sequential(
            nn.Conv2d(feat_channels*2, feat_channels*2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(feat_channels*2),
            nn.ReLU(inplace=True)
            )

        self.c3 = nn.Sequential(
            nn.Conv2d(feat_channels*2, feat_channels*2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(feat_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels*2, feat_channels*4, 3, padding=1),
            nn.InstanceNorm2d(feat_channels*4),
            nn.ReLU(inplace=True)
            )
        self.d3 = nn.Sequential(
            nn.Conv2d(feat_channels*4, feat_channels*4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(feat_channels*4),
            nn.ReLU(inplace=True)
            )

        self.c4 = nn.Sequential(
            nn.Conv2d(feat_channels*4, feat_channels*4, kernel_size=3, padding=1),
            nn.InstanceNorm2d(feat_channels*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels*4, feat_channels*8, kernel_size=3, padding=1),
            nn.InstanceNorm2d(feat_channels*8),
            nn.ReLU(inplace=True)
            )

        self.u1 = nn.Sequential(
            nn.ConvTranspose2d(feat_channels*8, feat_channels*8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(feat_channels*8),
            nn.ReLU(inplace=True)
            )
        self.c5 = nn.Sequential(
            nn.Conv2d(feat_channels*12, feat_channels*4, kernel_size=3, padding=1),
            nn.InstanceNorm2d(feat_channels*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels*4, feat_channels*4, kernel_size=3, padding=1),
            nn.InstanceNorm2d(feat_channels*4),
            nn.ReLU(inplace=True)
            )

        self.u2 = nn.Sequential(
            nn.ConvTranspose2d(feat_channels*4, feat_channels*4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(feat_channels*4),
            nn.ReLU(inplace=True)
            )
        self.c6 = nn.Sequential(
            nn.Conv2d(feat_channels*6, feat_channels*2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(feat_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels*2, feat_channels*2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(feat_channels*2),
            nn.ReLU(inplace=True)
            )

        self.u3 = nn.Sequential(
            nn.ConvTranspose2d(feat_channels*2, feat_channels*2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(feat_channels*2),
            nn.ReLU(inplace=True)
            )
        self.c7 = nn.Sequential(
            nn.Conv2d(feat_channels*3, feat_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(feat_channels),
            nn.ReLU(inplace=True)
            )
        
        self.out = nn.Sequential(
            nn.Conv2d(feat_channels, out_channels, kernel_size=1), 
            nn.Sigmoid()
            )
        self.out2 = nn.Sequential(
            nn.Conv2d(feat_channels, 2, kernel_size=1), 
            nn.Sigmoid()
            )

    def forward(self, mask):
        
        c1 = self.c1(mask)
        d1 = self.d1(c1)
        
        c2 = self.c2(d1)
        d2 = self.d2(c2)
        
        c3 = self.c3(d2)
        d3 = self.d3(c3)
        
        c4 = self.c4(d3)
        
        u1 = self.u1(c4)
        c5 = self.c5(torch.cat((u1,c3),1))
        
        u2 = self.u2(c5)
        c6 = self.c6(torch.cat((u2,c2),1))
        
        u3 = self.u3(c6)
        c7 = self.c7(torch.cat((u3,c1),1))
        
        out1 = self.out(c7)

        ## second output path for label reconstruction

        c1 = self.c1rec(out1)
        d1 = self.d1(c1)
        
        c2 = self.c2(d1)
        d2 = self.d2(c2)
        
        c3 = self.c3(d2)
        d3 = self.d3(c3)
        
        c4 = self.c4(d3)
        
        u1 = self.u1(c4)
        c5 = self.c5(torch.cat((u1,c3),1))
        
        u2 = self.u2(c5)
        c6 = self.c6(torch.cat((u2,c2),1))
        
        u3 = self.u3(c6)
        c7 = self.c7(torch.cat((u3,c1),1))
        
        out2 = self.out2(c7)
        
        return out1, out2
        

# specify the discriminator network        
class Discriminator(nn.Module):
    
    def __init__(self, patch_size, in_channels):
        super(Discriminator, self).__init__()
        
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_size = tuple([int(p/2**4) for p in patch_size])

        # Define layer instances
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1)
        self.leaky1 = nn.LeakyReLU(negative_slope=0.2)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.norm2 = nn.InstanceNorm2d(128)
        self.leaky2 = nn.LeakyReLU(negative_slope=0.2)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.norm3 = nn.InstanceNorm2d(256)
        self.leaky3 = nn.LeakyReLU(negative_slope=0.2)
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.norm4 = nn.InstanceNorm2d(512)
        self.leaky4 = nn.LeakyReLU(negative_slope=0.2)
        
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.norm5 = nn.InstanceNorm2d(512)
        self.leaky5 = nn.LeakyReLU(negative_slope=0.2)
        
        self.conv6 = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)
        self.sig6 = nn.Sigmoid()


    def forward(self, img):
        
        out = self.conv1(img)
        out = self.leaky1(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.leaky2(out)
        
        out = self.conv3(out)
        out = self.norm3(out)
        out = self.leaky3(out)
        
        out = self.conv4(out)
        out = self.norm4(out)
        out = self.leaky4(out)
        
        out = self.conv5(out)
        out = self.norm5(out)
        out = self.leaky5(out)
        
        out = self.conv6(out)
        out = self.sig6(out)

        return out


# Define the lightning module for the GAN
class GAN2D(pl.LightningModule):

    def __init__(self, hparams):
        super(GAN2D, self).__init__()
        self.hparams = hparams

        # networks
        self.generator = Generator(patch_size=hparams.patch_size, in_channels=hparams.in_channels, out_channels=hparams.out_channels)
        self.discriminator = Discriminator(patch_size=hparams.patch_size, in_channels=hparams.in_channels)

        # cache for generated images
        self.generated_imgs = None
        self.reconstructed_labels = None
        self.last_imgs = None
        self.last_masks = None

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        
        # Get image ans mask of current batch
        imgs, masks = batch['raw_image'], batch['target_image']
        current_batch_size = imgs.shape[0]

        # train generator
        if optimizer_idx == 0:
            self.last_imgs = imgs
            self.last_masks = masks

            # generate images
            self.generated_imgs, self.reconstructed_labels = self.forward(masks)

            sample_raw_imgs_input = 255*imgs[:,0,:,:]
            sample_raw_imgs_input = sample_raw_imgs_input.unsqueeze(1)
            sample_raw_imgs_input = sample_raw_imgs_input.type(torch.uint8)
            grid3 = torchvision.utils.make_grid(sample_raw_imgs_input)
            self.logger.experiment.add_image('input/raw_real', grid3, 0)

            sample_imgs_input = 255*masks[:,0,:,:]
            sample_imgs_input = sample_imgs_input.unsqueeze(1)
            sample_imgs_input = sample_imgs_input.type(torch.uint8)
            grid2 = torchvision.utils.make_grid(sample_imgs_input)
            self.logger.experiment.add_image('input/stateMasks_real', grid2, 0)

            sample_imgs_input = 255*masks[:,1,:,:]
            sample_imgs_input = sample_imgs_input.unsqueeze(1)
            sample_imgs_input = sample_imgs_input.type(torch.uint8)
            grid2 = torchvision.utils.make_grid(sample_imgs_input)
            self.logger.experiment.add_image('input/intensityMasks_real', grid2, 0)

            # log sampled images
            sample_imgs = 255*self.generated_imgs
            sample_imgs = sample_imgs.type(torch.uint8)
            grid = torchvision.utils.make_grid(sample_imgs)
            self.logger.experiment.add_image('output/raw_generated', grid, 0)

            # log sampled images
            sample_imgs = 255*self.reconstructed_labels[:,0,:,:]
            sample_imgs = sample_imgs.unsqueeze(1)
            sample_imgs = sample_imgs.type(torch.uint8)
            grid = torchvision.utils.make_grid(sample_imgs)
            self.logger.experiment.add_image('output/stateMasks_generated', grid, 0)

            # log sampled images
            sample_imgs = 255*self.reconstructed_labels[:,1,:,:]
            sample_imgs = sample_imgs.unsqueeze(1)
            sample_imgs = sample_imgs.type(torch.uint8)
            grid = torchvision.utils.make_grid(sample_imgs)
            self.logger.experiment.add_image('output/intensityMasks_generated', grid, 0)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones((current_batch_size,)+(1,)+self.discriminator.out_size, dtype=torch.float32)
            if self.on_gpu:
                valid = valid.cuda(imgs.device.index)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(self.discriminator(torch.cat((self.generated_imgs, masks[:,0:2,:,:]), axis=1)), valid)
            g_loss += torch.mean(torch.abs(masks[:,0:2,:,:] - self.reconstructed_labels))

            tqdm_dict = {'g_loss': g_loss, 'epoch': self.current_epoch}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            # how well can it label as real?
            valid = torch.ones((current_batch_size,)+(1,)+self.discriminator.out_size, dtype=torch.float32)
            if self.on_gpu:
                valid = valid.cuda(imgs.device.index)

            real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

            # how well can it label as fake?
            fake = torch.zeros((current_batch_size,)+(1,)+self.discriminator.out_size, dtype=torch.float32)
            if self.on_gpu:
                fake = fake.cuda(imgs.device.index)

            fake_images = torch.cat((self.generated_imgs.detach(), self.last_masks[:,0:2,:,:]), axis=1)
            fake_loss = self.adversarial_loss(self.discriminator(fake_images), fake)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output
        
    def test_step(self, batch, batch_idx):
        x, y = batch['image'], batch['mask']
        x_hat = self.forward(y)
        return {'test_loss': F.kl_div(x_hat, x)} 

    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        lr = self.hparams.learning_rate

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        return [opt_g, opt_d], []

    @pl.data_loader
    def train_dataloader(self):
        dataset = CCGH5DataLoader(image_dirs=self.hparams.input_paths, patch_size=self.hparams.patch_size, input_filter=self.hparams.input_filter, normalization_mode='data_range')
        return DataLoader(dataset, batch_size=self.hparams.batch_size)
    
    @pl.data_loader
    def test_dataloader(self):
        dataset = CCGH5DataLoader(image_dirs=self.hparams.input_paths, patch_size=self.hparams.patch_size, input_filter=self.hparams.input_filter, normalization_mode='data_range')
        return DataLoader(dataset, batch_size=self.hparams.batch_size)
    
    @pl.data_loader
    def val_dataloader(self):
        dataset = CCGH5DataLoader(image_dirs=self.hparams.input_paths, patch_size=self.hparams.patch_size, input_filter=self.hparams.input_filter, normalization_mode='data_range')
        return DataLoader(dataset, batch_size=self.hparams.batch_size)

    def on_epoch_end(self):
        return
        
    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no cover
        """
        Parameters you define here will be available to your model through self.hparams
        """
        parser = ArgumentParser(parents=[parent_parser])

        # network params
        parser.add_argument('--in_channels', default=2, type=int)
        parser.add_argument('--out_channels', default=2, type=int)
        parser.add_argument('--patch_size', default=(96,96), type=int)
        parser.add_argument('--drop_prob', default=0.2, type=float)
        parser.add_argument('--learning_rate', default=0.001, type=float)

        # data
        parser.add_argument('--input_paths', default="/images/", type=str)
        parser.add_argument('--input_filter', default="*.*", type=str)

        # training params (opt)
        parser.add_argument('--batch_size', default=2, type=int)
        
        return parser