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
import numpy as np
import torch
from skimage import io
from argparse import ArgumentParser
from scipy.ndimage import distance_transform_edt
from torch.autograd import Variable
from os import listdir
from os.path import isfile, join
import glob

SEED = 1337
torch.manual_seed(SEED)
np.random.seed(SEED)

# main function
def main(hparams):
    
    """
    Main testing routine specific for this project
    :param hparams:
    """

    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = network(hparams=hparams)
    model = model.load_from_checkpoint(hparams.ckpt_path)
    model = model.cuda()
    
    # ------------------------
    # 2 INIT DATA TILER
    # ------------------------
    os.makedirs(hparams.output_path, exist_ok=True)
    
    # ------------------------
    # 3 PROCESS EACH IMAGE
    # ------------------------
    batch_size = hparams.batch_size
    input_path = hparams.input_path
    output_path = hparams.output_path

    # identify images that should be processed
    image_files = list()
    for f in sorted(glob.glob(input_path + '*.png')):
        if isfile(join(input_path, f)):
            image_files.append(join(input_path, f))

    # perform GAN-based processing
    for f in range(0, len(image_files), batch_size):
        
        # convert the RGB images to the appropriate float format
        myimage = np.zeros((batch_size, 3, 96, 96), dtype=float)
        for i in range(0, batch_size):

            # load conditioning image
            data = io.imread(image_files[np.min((f+i, len(image_files)-1))])

            myimage[i,0,:,:] = data[:,:,0] / 6     # divide by number of stages
            myimage[i,1,:,:] = data[:,:,1] / 255   # normalize intensity conditioning
            myimage[i,2,:,:] = data[:,:,2] / 255   # normalize random noise

        # confert input image to torch tensor
        data = torch.Tensor(myimage).cuda()

        # predict the current batch and write it back to main memory
        pred_patch, pred_stages = model(data)
        pred_patch = torch.squeeze(pred_patch.cpu()).detach()

        # convert float image back to uint8
        result_image = np.uint8(pred_patch * 255)

        # write the individual images to disk again
        for i in range(0, batch_size):
            output_filename = image_files[np.min((f+i, len(image_files)-1))].replace(input_path, output_path)
            io.imsave(output_filename, result_image[i,...])

if __name__ == '__main__':
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments

    parent_parser = ArgumentParser(add_help=False)

    parent_parser.add_argument(
        '--input_path',
        type=str,
        default='',
        help='output path for test results'
    )

    parent_parser.add_argument(
        '--output_path',
        type=str,
        default='',
        help='output path for test results'
    )
    
    parent_parser.add_argument(
        '--ckpt_path',
        type=str,
        default='',
        help='output path for test results'
    )
    
    parent_parser.add_argument(
        '--distributed_backend',
        type=str,
        default='dp',
        help='supports three options dp, ddp, ddp2'
    )
    
    parent_parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of GPUs to use'
    )
    
    parent_parser.add_argument(
        '--overlap',
        type=int,
        default=40,
        help='overlap of adjacent patches'
    )

    parent_parser.add_argument(
        '--input_batch',
        type=str,
        default='image',
        help='which part of the batch is used as input (image | mask)'
    )
    
    parent_parser.add_argument(
        '--model',
        type=str,
        default='gan2d',
        help='which model to load (gan2d)'
    )
    
    parent_parser.add_argument(
        '--clip',
        type=float,
        default=(-1.0, 1.0),
        help='number of GPUs to use',
        nargs='+'
    )
    
    parent_args = parent_parser.parse_known_args()[0]
    
    # load the desired network architecture
    if parent_args.model.lower() == 'gan2d':
        from CellCycleGAN2D import GAN2D as network
    else:
        raise ValueError('Model {0} unknown.'.format(parent_args.model))
        
    # each LightningModule defines arguments relevant to it
    parser = network.add_model_specific_args(parent_parser)
    hyperparams = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hyperparams)