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

%% function to sample a set of evenly distributed boundary points based on a fixed angular step
function [samplingPoints] = ExtractSamplingPoints(maskImage, numSamplingPoints)

    %% find the centroid to center the boundary
    regionProps = regionprops(maskImage > 0, 'Orientation');
    
    %% rotate the mask such that the major axis alings with the y axis
    maskImage = imrotate(maskImage, -regionProps(1).Orientation + 90);
    regionProps = regionprops(maskImage > 0, 'Centroid');
        
    %% identify the boundary pixels of the current mask
    currentBoundary = bwboundaries(maskImage > 0);
    currentBoundary = currentBoundary{1,1};
    currentBoundary(:,1) = currentBoundary(:,1) - regionProps(1).Centroid(2);
    currentBoundary(:,2) = currentBoundary(:,2) - regionProps(1).Centroid(1);
   
    %% create sampling angles based on the number of sampling points
    stepSize = 2*pi / (numSamplingPoints);
    samplingRange = -pi:stepSize:pi;
    
    %% initialize the sampling points and compute the angles of the current boundary pixels
    samplingPoints = zeros(numSamplingPoints, 2);
    boundaryAngles = atan2(currentBoundary(:,2), currentBoundary(:,1));
    
    %% identify the closest boundary points to the sampling points
    for i=1:numSamplingPoints
        currentAngle = samplingRange(i);
        [~, minIndex] = min(abs(boundaryAngles - currentAngle));
        samplingPoints(i,:) = currentBoundary(minIndex, :);        
    end
    
    %% plot debug figures if enabled
    debugFigures = false;
    if (debugFigures == true)
        figure(2); hold on;
        scatter(samplingPoints(:,2), samplingPoints(:,1), 10, 1:numSamplingPoints, 'filled');
        axis equal;
        colormap hsv;
    end
end