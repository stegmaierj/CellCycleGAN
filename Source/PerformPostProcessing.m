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
function resultImage = PerformPostProcessing(rawImage, settings)

    %% perform post processing
    resultImage = double(rawImage);
    resultImage = max(imgaussfilt(resultImage, settings.postProcessingGaussianBlurSigma), resultImage);        
    resultImage = uint8(settings.postProcessingPoissonNoiseWeight * imnoise(resultImage, 'poisson') + (settings.postProcessingPoissonNoiseWeight) * resultImage);
    resultImage = max(0, imnoise(uint8(resultImage), 'gaussian', 0, settings.postProcessingGaussianNoiseSigma));
end