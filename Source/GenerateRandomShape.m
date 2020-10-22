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
function [currentShape, randomWeights, maskImage] = GenerateRandomShape(stage, settings, randomWeights)

    %% draw random vector if none is provided
    if (nargin <= 2)
        randomWeights = randn(settings.numEigenVectors, 1);
    end
    
    %% get the mean shape 
    currentShape = settings.shapeModel.meanShapes{stage};
    
    %% create the random shape as a linear combination of the eigenvectors and the mean shape
    for j=1:settings.numEigenVectors
        currentShape = currentShape(:) + randomWeights(j) * settings.shapeModel.eigenVectors{stage}(:,j) * sqrt(settings.shapeModel.eigenValues{stage}(j,j));
    end
    
    %% convert the linearized shape representation to a list of 2D vectors
    currentShape = reshape(currentShape, [settings.numSamplingPoints, 2]);
    
    %% convert boundary points to a shape mask
    maskImage = stage * poly2mask(currentShape(:,2) + settings.patchSize/2, currentShape(:,1) + settings.patchSize/2, settings.patchSize, settings.patchSize);                
end