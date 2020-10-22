%%
% CellCycleGAN.
% Copyright (C) 2020 D. Bähr, D. Eschweiler, A. Bhattacharyya, 
% D. Moreno-Andrés, W. Antonin, J. Stegmaier
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the Liceense at
% 
%     http://www.apache.org/licenses/LICENSE-2.0
% 
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.
%
% Please refer to the documentation for more information about the software
% as well as for installation instructions.
%
% If you use this application for your work, please cite the repository and one
% of the following publications:
%
% D. Bähr, D. Eschweiler, A. Bhattacharyya, D. Moreno-Andrés, W. Antonin, J. Stegmaier, 
% "CellCycleGAN: Spatiotemporal Microscopy Image Synthesis of Cell
% Populations using Statistical Shape Models and Conditional GANs", arxiv,
% 2020.
%
%%

%% set random seed to get reproducible results
clear settings;
settings.randomSeed = 50108; %% change the random number generator initialization to generate a set of different images

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% input and output directories
settings.sourceDir = [pwd filesep]; %% if you don't start the script from the current folder, make sure to provide an absolute path to the script directory
settings.inputFolder = '/images/BiomedicalImageAnalysis/CellSimulation_ISBI2021/cells_7domains/'; %% path to the *.h5 files extracted from the SI of Zhong et al., 2012, NMeth
settings.outputFolder = '/work/scratch/stegmaier/Projects/2020/CellSimulation_ISBI2021/CellCycleGAN/Data/'; %% output folder for temporary and generated image data
settings.shapeModelPath = [settings.sourceDir 'shapeModel.mat']; %% shape model path
settings.pretrainedModelPath = '/work/scratch/stegmaier/Projects/2020/CellSimulation_ISBI2021/Results/_ckpt_epoch_11998.ckpt';

%% enable if you want to recompute the shape models
settings.createNewShapeModel = false;

%% create ouput folders if non-existent yet
settings.outputFolderTempLabel = [settings.outputFolder 'TempLabel/'];
settings.outputFolderTempRaw = [settings.outputFolder 'TempRaw/'];
settings.outputFolderLabel = [settings.outputFolder 'Label/'];
settings.outputFolderRaw = [settings.outputFolder 'Raw/'];
if (~isfolder(settings.outputFolderTempLabel)); mkdir(settings.outputFolderTempLabel); end
if (~isfolder(settings.outputFolderTempRaw)); mkdir(settings.outputFolderTempRaw); end
if (~isfolder(settings.outputFolderLabel)); mkdir(settings.outputFolderLabel); end
if (~isfolder(settings.outputFolderRaw)); mkdir(settings.outputFolderRaw); end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% settings for the image generation
settings.imageSize = [1024, 1024];                      %% image size of the generated images
settings.numObjects = 20;                               %% initial number of objects
settings.numStages = 6;                                 %% number of stages (only needs to be changed if other ground truth is used)
settings.numSamplingPoints = 60;                        %% number of sampling points used to sample the shape borders (default=60, recompute shape models if changed!)
settings.numFrames = 50;                                %% number of frames to be simulated. Note that a large number combined with a small image size may result in very crowded scenes
settings.borderPadding = 100;                           %% safety border that should not be occupied by cells. If simulation is too long, cells may move towards the boundary and are stuck at this padding border
settings.debugFigures = false;                          %% if enabled a few debug figures are visualized
settings.numRefinementSteps = 1;                        %% regularization for the initial cell positions. Higher numbers yield very homogeneously distributed cells
settings.patchSize = 96;                                %% image patch size used for generation and training. Note that the number should be often divisible by 2, due to the downsampling performed by the GAN

settings.randomMotionMultiplier = 2.0;                  %% random displacements applied relative to the predecessor sampled from N(0,1) and multiplied with the selected randomMotionMultiplier
settings.rotationAngleMultiplier = 1.0;                 %% random angles applied relative to the predecessor sampled from N(0,1) and multiplied with the selected rotationAngleMultiplier
settings.anaPhaseDisplacementMultiplier = 1.0;          %% displacement of the two chromatin masses after division relative to the minor axis of the metaphase.
settings.shapeRandomVariablesMultiplier = 1.0;          %% value is multiplied with a random number sampled from N(0, 1) and used to scale the eigenvectors of the shape model
settings.randomizationWithinStageWeight = 0.1;          %% same as before, to add small random fluctuations between shapes of the same stage. Increase if differences should be more pronounced.

%% post processing
settings.postProcessingGaussianBlurSigma = 2.0;         %% standard deviation used for the Gaussian-based PSF simulation
settings.postProcessingGaussianNoiseSigma = 0.00005;    %% variance of the zero-mean additive Gaussian noise
settings.postProcessingPoissonNoiseWeight = 0.5;        %% weighted average between original image and Poisson disrupted image. 0.5, e.g., weights both images similarly
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% simulation parameters estimated from the ground truth data

%% measured transition probabilities from the training data
%% 0.9601    0.2995    0.5394    0.8020    0.5442    0.7542 %% probability of self-cycle
%% 0.0399    0.7005    0.4606    0.1980    0.4558    0.2458 %% probability of stage transition

%% Define transition matrix for a graphical model with 6 states 
% (interphase, prophase, prometaphase, metaphase, anaphase, telophase, interphase).
%% Diagonal elements represent self-cycles, off diagonal elements stage transitions
settings.transitionMatrix = [0.96, 0.04, 0.0, 0.0, 0.0, 0.0; ...     %% interphase
                             0.0, 0.3, 0.7, 0.0, 0.0, 0.0; ...       %% prometaphase
                             0.0, 0.0, 0.46, 0.54, 0.0, 0.0; ...     %% prophase
                             0.0, 0.0, 0.0, 0.8, 0.2, 0.0; ...       %% metaphase
                             0.0, 0.0, 0.0, 0.0, 0.54, 0.46; ...     %% anaphase
                             0.25, 0.0, 0.0, 0.0, 0.0, 0.75];        %% telophase

%% hard constraints for the graphical model
settings.minStageLengths = [7     1     1     1     1     1];   %% minimum length of each stage
settings.maxStageLengths = [31    3    30    29     6    20];   %% maximum length of each stage

%% measured mean intensities and standard deviation of mean intensity values per stage
settings.meanIntensities = [38.6889 43.1369  65.5235 95.0452 73.6037  54.3290]; %% mean intensity averaged of snippets belonging to the same stage
settings.stdIntensities = [11.2874 12.3908 19.0159 24.9552 18.9785 13.4924];    %% mean intensity standard deviation of snippets belonging to the same stage

%% stage indices used for consistent referencing
settings.interPhaseIndex = 1;                           %% index for the interphase
settings.proPhaseIndex = 2;                             %% index for the prophase
settings.prometaPhaseIndex = 3;                         %% index for the prometaphase
settings.metaPhaseIndex = 4;                            %% index for the metaphase
settings.anaPhaseIndex = 5;                             %% index for the anaphase
settings.teloPhaseIndex = 6;                            %% index for the telophase

%% extract the statistical shape model for the current set of input images or load it
if (settings.createNewShapeModel == true || ~isfile(settings.shapeModelPath))
    shapeModel = GenerateShapeModels(settings.inputFolder, settings);
    save(settings.shapeModelPath, 'shapeModel');
else
    load(settings.shapeModelPath);
end
settings.shapeModel = shapeModel;
settings.numEigenVectors = size(settings.shapeModel.eigenVectors{1}, 1);

%% perform the benchmark generation
rng(settings.randomSeed);
settings.saveResultImages = false;
PerformMaskGeneration;

%% call Python script for the GAN translation of the image patches
system(['python ' settings.sourceDir 'CellCycleGAN.py --input_path ' settings.outputFolderTempLabel ' --output_path ' settings.outputFolderTempRaw ' --ckpt_path ' settings.pretrainedModelPath]);

%% call the mask generation script again in to assemble the final result image
%% using the newly created synthetic image snippets of the GAN
rng(settings.randomSeed);
settings.saveResultImages = true;
PerformMaskGeneration;