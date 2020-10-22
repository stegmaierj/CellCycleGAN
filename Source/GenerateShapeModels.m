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
function shapeModel = GenerateShapeModels(inputFolder, settings)

    %% find valid input files
    inputFiles = dir([inputFolder '*.h5']);
    numStages = settings.numStages;
    numSamplingPoints = settings.numSamplingPoints;
      
    %% initialize the mean shape and the shape covariance matrix
    meanShapes = cell(numStages, 1);
    shapeCovMatrices = cell(numStages, 1);
    for i=1:numStages
        meanShapes{i} = zeros(numSamplingPoints, 2);
        shapeCovMatrices{i} = zeros(numSamplingPoints*2, numSamplingPoints*2);
    end
    numShapes = zeros(numStages, 1);

    %% Compute the mean shape
    for f=1:length(inputFiles)

        %% load input image and state labels
        maskImage = h5read([inputFolder inputFiles(f).name], '/target_image');
        stateLabels = h5read([inputFolder inputFiles(f).name], '/manual_stages');

        %% get the number of frames in the current image
        numFrames = size(maskImage, 3);

        %% process all frames
        for i=1:numFrames

            %% get the current image and stage label
            currentImage = squeeze(maskImage(:,:,i));
            currentStage = stateLabels(i);
            
            %% extract the boundary points for the current image
            currentSamplingPoints = ExtractSamplingPoints(currentImage, numSamplingPoints);

            %% compute the mean shape and increase number of summands for later normalization
            meanShapes{currentStage} = meanShapes{currentStage} + currentSamplingPoints;
            numShapes(currentStage) = numShapes(currentStage) + 1;
        end
    end

    %% scaele the mean shapes with the number of contributing shapes
    for i=1:numStages
        meanShapes{i} = meanShapes{i} / numShapes(i);
    end

    figure(2); clf; hold on;
    for i=1:numStages
        subplot(1,numStages,i);
        plot(meanShapes{i}(:,2), meanShapes{i}(:,1), '*r');
        axis equal;
        axis([-30, 30, -30, 30]);
    end


    %% Compute the shape covariance matrix
    numShapes = zeros(numStages, 1);

    %% Compute the covariance matrix
    for f=1:length(inputFiles)
        
        %% load input image and state labels
        maskImage = h5read([inputFolder inputFiles(f).name], '/target_image');
        stateLabels = h5read([inputFolder inputFiles(f).name], '/manual_stages');

        %% get the number of frames in the current image
        numFrames = size(maskImage, 3);

        %% process all frames
        for i=1:numFrames

            %% get the current image and stage label
            currentImage = squeeze(maskImage(:,:,i));
            currentStage = stateLabels(i);
            
            %% extract the boundary points for the current image
            currentSamplingPoints = ExtractSamplingPoints(currentImage, numSamplingPoints);
            demeanedSamplingPoints = currentSamplingPoints - meanShapes{currentStage};

            %% compute the mean shape and increase number of summands for later normalization
            shapeCovMatrices{currentStage} = shapeCovMatrices{currentStage} + (demeanedSamplingPoints(:) * demeanedSamplingPoints(:)');
            numShapes(currentStage) = numShapes(currentStage) + 1;
        end
    end

    %% normalize the covariance matrix result
    for i=1:numStages
        shapeCovMatrices{i} = shapeCovMatrices{i} / (numShapes(i) - 1);
    end

    %% perform the eigenvalue decomposition of the covariance matrix
    eigenVectors = cell(numStages,1);
    eigenValues = cell(numStages, 1);
    
    for i=1:numStages
        [V,D] = eig(shapeCovMatrices{i});
        eigenVectors{i} = V;
        eigenValues{i} = D;
    end
        
    %% assemble the output variable
    shapeModel.meanShapes = meanShapes;
    shapeModel.shapeCovMatrices = shapeCovMatrices;
    shapeModel.eigenVectors = eigenVectors; %%%%%%%%%% --> convert to two dimensional vectors!
    shapeModel.eigenValues = eigenValues;
    
    %% plot a few random shapes
    if (settings.debugFigures == true)
        
        %% create 100 random shapes for all stages
        for i=1:100
            
            %% create new figure
            figure(3); clf; hold off;

            %% loop through all stages
            for s=1:numStages
                
                %% open subplot for the current shape
                subplot(1,numStages,s);

                %% get the current mean shape and modify it using random contributions of the eigenvectors
                currentShape = meanShapes{s}(:);
                for j=1:size(eigenValues,1)
                    currentShape = currentShape + randn * eigenVectors{s}(:,j) * sqrt(eigenValues{s}(j,j));
                end
                
                %% convert linearized coordinates to a list of 2D vectors            
                currentShape = reshape(currentShape, [numSamplingPoints, 2]);

                %% plot the current shape
                plot(currentShape(:,2), currentShape(:,1), '-r'); hold on;
                axis equal;
                axis([-40, 40, -40, 40]);
                drawnow;

                title(['Shape model ' num2str(s)]);
            end
            
            %% pause to check the current shapes
            pause(1);
        end
    end
end