# include system-related packages
import random
import os
import numpy as np
import argparse
import sys
import h5py
from os import listdir
from os.path import isfile, join
import glob

# import torch-related packages
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from skimage import data, io, filters, util

# class for loading optical flow *.h5 files.
# expects the *.h5 files to contain the following data sets:
# raw_image: original image pair containing the previous and the current time point.
#            Dimension for 2D are: 2 x 1 x H x W, and for 3D: 2 x D x H x W .
# target_image: target values for the current time point with 5 channels (0-2: flowX, flowY, flowZ; 3: seeds, 4: segmentation)
#               Dimensions of the target image for 2D are 5 x 1 x H x W and for 3D: 5 x D x H x W.
# manual_stages: binary image with zeros for parts of the image to ignore and ones for regions with available ground truth. 
#             Dimensions are 1 x H x W and D x H x W for 2D and 3D, respectively.
class CCGH5DataLoader(Dataset):
    """Semantic segmentation dataset."""

    def __init__(self, image_dirs, input_filter='', patch_size=(1, 96, 96), transforms=None, random_subset=1.0, inference_mode=False, normalization_mode='none'):
        """
        Args:
            image_dir (string): Path to h5 files containing the fields raw_image, target_image, centroids and dont_cares.
            transform (dict, optional): Optional transforms to be applied on a sample. Only a limited amount of transforms is available currently, 
                                        as it's not trivial to apply arbitary transformations to the flow fields.
            TODO: inference_mode (boolean): loads the entire image and returns a tiled representation of the image that can be processed separately
            normalization_mode (int): none: no normalization, max: divide by maximum intensity, zero_one: scale [min, max] to [0, 1], zscore: scale to zero mean, unit std. dev., data_range: 8/16 bit range -> [0,1]
            use_centroid_sampling: uses the centroids dict entry of the H5 files to sample regions of the desired size that contain a randomly selecte nucleus at a random location in the image patch
        """

        # specify the image dirs and the transforms to be used.
        self.image_dirs = image_dirs
        self.transforms = transforms
        self.random_subset = random_subset
        self.inference_mode = inference_mode
        self.normalization_mode = normalization_mode

        if (transforms is not None):
            self.transforms = {
                'gaussian_blur' : 0.0,                  # probability of applying gaussian blur. 0: deactivated, 1: always activated.
                'gaussian_blur_max_sigma' : 2.0,        # sigma value will be drawn randomly in the interval [0, gaussian_blur_max_sigma].
                'intensity_jitter' : 0.0,               # probability of applying intensity jitter. 0: deactivated, 1: always activated.
                'intensity_jitter_range' : [0.8, 1.0],  # range in which to jitter the intensiy. 1.0 has no effect, < 1 decreases intensity, > 1 increases intensity.
                'gaussian_noise' : 0.0,                 # probabiltiy of applying gaussian noise. 0: deactivated, 1: always activated.
                'gaussian_noise_mean' : 0.0,            # mean value to be used for the gaussian noise. Default: 0.0.
                'gaussian_noise_scale' : 1.0,           # value drawn from a standard gaussian with sigma=1 will be multiplied by this constant. Default: 1.0.
                'depth_flip': 0.0,                      # flips the image along the z-axis with the provided probability. 0: deactivated, 1: always activated.
                'vertical_flip': 0.0,                   # flips the image along the y-axis with the provided probability. 0: deactivated, 1: always activated.
                'horizontal_flip': 0.0                  # flips the image along the x-axis with the provided probability. 0: deactivated, 1: always activated.
            }
        
            for key in transforms:
                self.transforms[key] = transforms[key]

        # input can be a list of directories that will all be parsed for input images.
        if type(image_dirs) is not list:
            image_dirs = {image_dirs}

        self.image_files = list()
        for image_dir in image_dirs:
            print(image_dir)
            for f in sorted(glob.glob(image_dir + input_filter)):
                if isfile(join(image_dir, f)):
                    self.image_files.append(join(image_dir, f))

        # identify the number of images or select random subset, e.g., for validation
        if (self.random_subset < 1.0):
            self.num_images = np.round(self.random_subset * len(self.image_files)).astype(np.int32)
            self.image_files = random.choices(self.image_files, k=self.num_images)  
        else:
            self.num_images = len(self.image_files)

        # use full image size, if no patch size is provided.
        if (patch_size != None):
            self.patch_size = patch_size
        else:
            if (self.inference_mode == False):
                f_handle = h5py.File(self.image_files[0], 'r')
                raw_image = f_handle['raw_image']
                self.patch_size = raw_image.maxshape[1:]
            else:
                raw_image = io.imread(self.image_files[0])
                self.patch_size = raw_image.shape

        # check if the current input image is a 2D or a 3D image.
        self.image_dimension = len(self.patch_size)

    # function to return the number of images
    def __len__(self):
        return self.num_images

    # function to get the filename of an index
    def __getfilename__(self, idx):
        return self.image_files[idx]

    # function that returns a data set from the list of files.
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # if in inference mode, simply load two sequential tif files and append them as the raw image
        if (self.inference_mode == True):

            # get the current file extension
            _, extension = os.path.splitext(self.image_files[idx])
            extension = extension.lower()

            print('Trying to load file ' + self.image_files[idx])

            # call appropriate loader depending on the file format
            if extension.find('.tif') > -1:
                raw_image = io.imread(self.image_files[idx])

                # convert rgb to gray if three channels are present
                if (len(raw_image.shape) > 2):
                    raw_image = np.mean(raw_image, -1)

            elif extension.find('.h5') > -1:
                f_handle = h5py.File(self.image_files[idx], 'r')
                raw_image = f_handle['raw_image']
                raw_image = raw_image[:]
            else:
                print('Unrecognized file format. Supported formats are tiff and hdf5.')

            # convert image to zero mean, unit standard deviation.
            raw_image = raw_image.astype(np.float32)
            raw_image = self.normalize_intensity(raw_image, self.normalization_mode)
            target_image = None
            dont_cares = None

            # compose and return the current sample
            sample = {'raw_image': raw_image, 'target_image': target_image, 'dont_cares': dont_cares}
            return sample

        # open the file handle and handles to the images
        f_handle = h5py.File(self.image_files[idx], 'r')
        raw_image = f_handle['raw_image']
        target_image = f_handle['target_image']
        manual_stages = f_handle['manual_stages']
        
        if (raw_image.ndim == 2):
            image_size = (1, raw_image.maxshape[0], raw_image.maxshape[1])
        elif (raw_image.ndim == 3):
            image_size = raw_image.maxshape
        else:
            image_size = raw_image.maxshape[1:]

        # load the raw image. In case we only have one channel, extend the first two dimensions
        # to consistently have 4 dimensional images.
        selected_frame = random.randint(0, raw_image.maxshape[0]-1)
        raw_image = raw_image[selected_frame, :, :]
        raw_image = raw_image[np.newaxis, :]
                
        # load the target image. In case we only have one channel, extend the first two dimensions
        # to consistently have 4 dimensional images.
        target_image = target_image[selected_frame, :, :]
        target_image = target_image[np.newaxis, :]
        dont_cares = np.zeros_like(target_image)
        
        # convert image to float
        raw_image = raw_image.astype(np.float32)
        target_image = target_image.astype(np.float32)
        dont_cares = dont_cares.astype(np.float32)

        # convert image to zero mean, unit standard deviation or scale [min, max] to [0, 1]
        raw_image = self.normalize_intensity(raw_image, self.normalization_mode)

        # apply selected augmentations
        if (self.transforms is not None):
            sample = self.apply_transformations(raw_image, target_image, dont_cares, self.transforms)
            raw_image = sample['raw_image'].astype(np.float32)
            target_image = sample['target_image'].astype(np.float32)
            dont_cares = sample['dont_cares'].astype(np.float32)

        # suppress pixels outside of the mask
        current_stage = manual_stages[selected_frame]
        target_image = current_stage * target_image

        raw_image[target_image <= 0] = 0
        raw_image = raw_image * np.random.uniform(0.5, 1.0)

        stage_image = current_stage * target_image / 6 # divide by number of stages to avoid over pronouncing the stage vs. the intensity
        mean_intensity_image = np.mean(raw_image[target_image > 0]) * target_image
        raw_image = np.concatenate((raw_image, stage_image, mean_intensity_image), axis=0)

        image_shape = stage_image.shape
        target_image = np.concatenate((stage_image, mean_intensity_image, np.float32(np.random.rand(image_shape[0], image_shape[1], image_shape[2]))), axis=0)

        # compose and return the current sample 
        sample = {'raw_image': raw_image, 'target_image': target_image, 'dont_cares': dont_cares}
        return sample

    # function to apply random transformations to the raw image.
    # due to the limitations of applicable transformations, only transformations 
    # affecting the raw intensities are considered (gaussian noise, intensity jitter, gaussian blur)
    def apply_transformations(self, raw_image, target_image, dont_cares, transformations=None):

        # draw three random variables that determine to enable or disable the respective transforms
        gaussian_noise_probability = np.random.random_sample()
        gaussian_blur_probability = np.random.random_sample()
        intensity_jitter_probability = np.random.random_sample()
        vertical_flip_probability = np.random.random_sample()
        horizontal_flip_probability = np.random.random_sample()
        depth_flip_probability = np.random.random_sample()

        # apply gaussian noise
        if (gaussian_noise_probability <= self.transforms['gaussian_noise']):
            scale_factor = self.transforms['gaussian_noise_scale'] * (np.max(raw_image[:]) - np.min(raw_image[:]))
            gaussian_noise = scale_factor * np.random.normal(loc=0.0, scale=1.0, size=raw_image.shape)
            raw_image = raw_image + gaussian_noise

        # apply gaussian blur
        if (gaussian_blur_probability <= self.transforms['gaussian_blur']):
            blur_level = np.random.random_sample() * self.transforms['gaussian_blur_max_sigma']
            num_channels = raw_image.shape[0]
            for i in range(0, num_channels):
                raw_image[i, ...] = filters.gaussian(raw_image[i, ...], sigma=blur_level)

        # apply intensity jitter
        if (intensity_jitter_probability <= self.transforms['intensity_jitter']):
            jitter_range = self.transforms['intensity_jitter_range']
            raw_image = (jitter_range[0] + (jitter_range[1] - jitter_range[0]) * np.random.random_sample()) * raw_image

        # apply depth flip
        if (depth_flip_probability <= self.transforms['depth_flip']):
            raw_image = np.flip(raw_image, 1)
            target_image = np.flip(target_image, 1)
            dont_cares = np.flip(dont_cares, 1)

        # apply vertical flip
        if (vertical_flip_probability <= self.transforms['vertical_flip']):
            raw_image = np.flip(raw_image, 2)
            target_image = np.flip(target_image, 2)
            dont_cares = np.flip(dont_cares, 2)

        # apply horizontal flip
        if (horizontal_flip_probability <= self.transforms['horizontal_flip']):
            raw_image = np.flip(raw_image, 3)
            target_image = np.flip(target_image, 3)
            dont_cares = np.flip(dont_cares, 3)

        # ensure that the intensity values are still in the valid range of [0,1]
        raw_image = np.maximum(0, np.minimum(1.0, raw_image))

        # return the result image
        return {'raw_image': raw_image, 'target_image': target_image, 'dont_cares': dont_cares}

    # normalize image intensity according to provided normalization mode
    # none: no normalization, max: divide by maximum intensity, zero_one: scale [min, max] to [0, 1], zscore: scale to zero mean, unit std. dev., data_range
    def normalize_intensity(self, raw_image, normalization_mode):
        
        if (np.max(raw_image[:]) == 0) or (np.max(raw_image[:]) - np.min(raw_image[:])) == 0:
            return raw_image
        
        if (normalization_mode == 'max'):
            raw_image = raw_image / np.max(raw_image[:])
        elif (normalization_mode == 'zero_one'):
            raw_image = (raw_image - np.min(raw_image[:])) / (np.max(raw_image[:]) - np.min(raw_image[:]))                
        elif (normalization_mode == 'zscore'):
            raw_image = (raw_image - np.mean(raw_image[:])) / np.std(raw_image[:])
        elif (normalization_mode == 'data_range'):
            if (np.max(raw_image[:]) <= 255):
                raw_image = raw_image / 255
            else:
                raw_image = raw_image / 65535
        
        return raw_image