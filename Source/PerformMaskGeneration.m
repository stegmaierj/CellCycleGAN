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
 
%% initialize the objects in the first frame
clear objects;
objects = struct();
for i=1:settings.numObjects
    objects(i,1).id = i;
    objects(i,1).stages = randperm(settings.numStages,1);
    objects(i,1).rotation = rand * 360;
    objects(i,1).intensityStdDev = randn * settings.stdIntensities(objects(i,1).stages);
    objects(i,1).predecessor = -1;
    objects(i,1).rawImage = [];
end

%% specify the next object id, that is used when introducing daughter cells
nextObjectId = settings.numObjects+1;

%% generate simulated cell cycles for the entire series
for t=2:settings.numFrames

    %% process all objects
    for o=1:size(objects,1)

        %% get the previous object and continue if it does not exist
        prevObject = objects(o,t-1);
        if (isempty(prevObject.id))
            continue;
        end

        %% sample the next stage from the graphical model
        nextStage = GetNextStage(prevObject.stages, settings);

        %% perform division event if stage transition from 
        if (prevObject.stages(end) == settings.metaPhaseIndex && nextStage == settings.anaPhaseIndex)

            %% initialize daughters with their mother cell
            daughter1 = prevObject;
            daughter2 = prevObject;
            daughter1.stages = nextStage;
            daughter2.stages = nextStage;

            %% set the next object ids and the predecessors
            daughter1.id = nextObjectId;
            daughter2.id = nextObjectId+1;
            daughter1.predecessor = prevObject.id;
            daughter2.predecessor = prevObject.id;

            %% save new daughters to the objects structure
            objects(nextObjectId, t) = daughter1;
            objects(nextObjectId+1, t) = daughter2;
            objects(o,t-1).successors = [nextObjectId, nextObjectId+1];
            nextObjectId = nextObjectId + 2;
        else

            %% simply add object to current frame without division
            objects(o,t) = prevObject;
            objects(o,t).stages = [objects(o,t).stages, nextStage];
            objects(o,t).rotation = mod(prevObject.rotation + settings.rotationAngleMultiplier * randn, 360);
            objects(o,t).predecessor = -1;
        end
    end
end

%% perform interpolation between the different stage models
currentOutIndex = 1;
for o=1:size(objects,1)

    %% find the temporal extent of the current object
    minIndex = inf;
    maxIndex = 0;
    for t=1:settings.numFrames
        if (isempty(objects(o,t).id))
            continue;
        end

        if (t < minIndex)
            minIndex = t;
        end
        if (t >= maxIndex)
            maxIndex = t;
        end
    end

    %% get the stage sequence of the current object, identify the number of frames and find the stage changes
    stageSequence = objects(o,maxIndex).stages;
    numObjectFrames = length(stageSequence);
    stageChanges = [1, find(diff(stageSequence)) + 1, numObjectFrames];

    %% specify a shape model for each stage transition
    shapeModelStages = stageSequence(stageChanges);
    numShapeModels = length(stageChanges);

    %% create random vector for each shape model
    shapeRandomVariables = settings.shapeRandomVariablesMultiplier * randn(settings.numEigenVectors, numShapeModels);

    %% create the shape models 
    shapeTemplates = cell(numShapeModels,1);
    weightKernels = zeros(numShapeModels, numObjectFrames);
    for i=1:numShapeModels
       shapeTemplates{i} = GenerateRandomShape(shapeModelStages(i), settings, shapeRandomVariables(:,i)); 
       weightKernels(i,:) = normpdf(1:numObjectFrames, stageChanges(i), max(1, sum(stageSequence == shapeModelStages(i))));
    end

    %% create weighting function that is used to interpolate between shape models
    summedWeights = sum(weightKernels, 1);
    for i=1:numShapeModels
        weightKernels(i,:) = weightKernels(i,:) ./ summedWeights;
    end

    %% initialize the first shape model with the predecessor if available to ensure a smooth transition
    predecessorIndex = objects(o,minIndex).predecessor;
    if (predecessorIndex > 0)
        shapeTemplates{1} = objects(predecessorIndex,minIndex-1).currentShape;
    end

    %% perform the shape simulation based on the shape templates and the weighting functions
    for s=minIndex:maxIndex

        %% get the relative frame index and the current stage
        relativeFrameIndex = s - minIndex + 1;
        currentStage = stageSequence(relativeFrameIndex);

        %% create current shape
        currentShape = zeros(settings.numEigenVectors,1);
        for i=1:numShapeModels
            currentShape = currentShape + weightKernels(i,relativeFrameIndex) * shapeTemplates{i}(:);

            %% create the random shape as a linear combination of the eigenvectors and the mean shape
            for j=1:settings.numEigenVectors
                currentShape = currentShape + weightKernels(i,relativeFrameIndex) * settings.randomizationWithinStageWeight * randn * settings.shapeModel.eigenVectors{shapeModelStages(i)}(:,j) * sqrt(settings.shapeModel.eigenValues{shapeModelStages(i)}(j,j));
            end
        end

        %% convert current linearized version of the shape back to a mask
        currentShape = reshape(currentShape, [settings.numSamplingPoints, 2]);
        maskImage = currentStage * poly2mask(currentShape(:,2) + settings.patchSize/2, currentShape(:,1) + settings.patchSize/2, settings.patchSize, settings.patchSize);
        
        %% create the mask image with a special generation step for early ana-phase
        if (s == maxIndex && currentStage == settings.metaPhaseIndex)
            
            %% find properties of the mask for repulsive computations
            regionProps = regionprops(maskImage>0, 'BoundingBox', 'Centroid', 'EquivDiameter', 'MinorAxisLength', 'MajorAxisLength', 'Orientation');

            %% apply cell division orthogonal to the major axis
            currentOrientation = deg2rad(regionProps(1).Orientation + 90);
            directionVector = [cos(currentOrientation), sin(currentOrientation)];
            
            maskImage = max(imtranslate(maskImage, 0.2 * regionProps(1).MinorAxisLength * directionVector, 'nearest'), imtranslate(maskImage, -0.2 * regionProps(1).MinorAxisLength * directionVector, 'nearest'));
            maskImage(maskImage > 0) = settings.anaPhaseIndex;
        end

        %% create the current GAN-conditioning image
        currentIntensity = max(0, min(255, round(settings.meanIntensities(currentStage) + objects(o,s).intensityStdDev)));
        outputImage = cat(3, maskImage, double(maskImage > 0) * currentIntensity, 255 * rand(96,96));
        
        %% write the GAN conditioning image of the current cell to the temporary folder        
        outputFileNameTempLabel = sprintf('%s%04d.png', settings.outputFolderTempLabel, currentOutIndex);        
        imwrite(uint8(outputImage), outputFileNameTempLabel);
        
        %% perform GAN-based image synthesis
        outputFileNameTempRaw = sprintf('%s%04d.png', settings.outputFolderTempRaw, currentOutIndex);
        if (isfile(outputFileNameTempRaw))
            rawImage = imread(outputFileNameTempRaw);
            objects(o,s).rawImage = rawImage;
        end

        %% set the current shape
        objects(o,s).maskImage = maskImage;
        objects(o,s).currentShape = currentShape;
        
        %% increment 
        currentOutIndex = currentOutIndex+1;        
    end

    disp(['Finished creating shapes for ' num2str(o) ' / ' num2str(size(objects,1)) ' objects ...']);
end

%% stop processing here as no raw images exist yet
if (settings.saveResultImages == false)
    return;
end

%% sample some random starting positions
initialPositions = rand(settings.numObjects, 2);
initialPositions(:,1) = settings.borderPadding + min(round(initialPositions(:,1) * (settings.imageSize(1)-2*settings.borderPadding)), settings.imageSize(1)-2*settings.borderPadding);
initialPositions(:,2) = settings.borderPadding + min(round(initialPositions(:,2) * (settings.imageSize(2)-2*settings.borderPadding)), settings.imageSize(2)-2*settings.borderPadding);

%% optionally redistribute seed points with more homogeneous distribution
for r=1:settings.numRefinementSteps
    distanceMap = zeros(settings.imageSize);
    for i=1:settings.numObjects
        distanceMap(initialPositions(i,1), initialPositions(i,2)) = 1;
    end

    distanceMap = bwdist(distanceMap);
    watershedImage = watershed(distanceMap);
    currentRegionProps = regionprops(watershedImage, 'Centroid');

    for i=1:length(currentRegionProps)
       initialPositions(i,:) = round(currentRegionProps(i).Centroid); 
    end
end   

%% assemble result images frame by frame
for t=1:settings.numFrames

    %% initialize the current result image        
    resultImage = zeros(settings.imageSize);
    resultImageStates = zeros(settings.imageSize);
    resultImageIntensity = zeros(settings.imageSize);
    resultImageRaw = uint8(2*ones(settings.imageSize));

    %% add all objects currently present to the result frame
    for o=1:size(objects,1)

        %% continue if object does not exist
        if (isempty(objects(o,t).id))
            continue;
        end

        %% initialize object positions with their predecessors
        if (t==1)
           objects(o,t).position = initialPositions(o,:);
        else
           if (objects(o,t).predecessor > 0)
              objects(o,t).position = objects(objects(o,t).predecessor,t-1).position;
           else
              objects(o,t).position = objects(o,t-1).position;
           end
        end

        %% get the current mask image and rotate it randomly
        maskImage = objects(o,t).maskImage;
        maskImage = imrotate(maskImage, objects(o,t).rotation);
        
        if (~isempty(objects(i,1).rawImage))
            rawImage = imrotate(objects(o,t).rawImage, objects(o,t).rotation, 'bilinear');
        end

        %% find properties of the mask for repulsive computations
        regionProps = regionprops(maskImage>0, 'BoundingBox', 'Centroid', 'EquivDiameter', 'MinorAxisLength', 'MajorAxisLength', 'Orientation');

        %% apply cell division orthogonal to the major axis
        if (objects(o,t).predecessor > 0)

            %% compute the orientation vector
            currentOrientation = deg2rad(regionProps(1).Orientation);
            directionVector = [cos(currentOrientation), sin(currentOrientation)];

            %% split objects
            objects(o,t).position = objects(o,t).position + (-1)^(mod(o,2)) * settings.anaPhaseDisplacementMultiplier * regionProps(1).MinorAxisLength * directionVector;
        end

        %% compute repulsive forces based on the axes lengths
        Rm = regionProps(1).MajorAxisLength * 5;
        Rn = regionProps(1).MinorAxisLength * 5;
        repulsiveForce = zeros(1,2);

        %% check all neighbors if they intersect with the current object
        currentPosition = objects(o,t).position;
        for k=1:size(objects,1)

            %% continue if no object present
            if (t <= 1 || k==o || isempty(objects(k,t-1).id))
                continue;
            end

            %% get the current 
            neighborPosition = objects(k,t-1).position;

            %% compute the distance vector between the current object and its neighbor
            currentDistanceVector = neighborPosition - currentPosition;
            currentDistance = norm(currentDistanceVector);

            %% continue if distance is zero
            if (currentDistance <= 0)
                continue;
            end

            %% normalize the current direction
            currentDirection = (currentDistanceVector / currentDistance);

            %% compute repulsive force according to the paper by Macklin et al., 2012.
            if (currentDistance <= Rn)
                repulsiveForce = repulsiveForce - (((1-Rn/Rm)^2 - 1) * (currentDistance / Rn) + 1) * currentDirection;
            elseif (currentDistance > Rn && currentDistance <= Rm)             
                repulsiveForce = repulsiveForce - (1 - currentDistance / Rm)^2 * currentDirection;
            end
        end

        %% update the object position respecting the image boundaries and border padding
        objects(o,t).position = max(settings.borderPadding, min(round(objects(o,t).position + repulsiveForce + settings.randomMotionMultiplier*randn(1,2)), settings.imageSize(1)-settings.borderPadding));

        %% compute ranges of the input patch that correspond to the global result image coordinates
        rangeX = max(1, min(round(regionProps(1).BoundingBox(2)):(regionProps(1).BoundingBox(2)+regionProps(1).BoundingBox(4)), size(maskImage,1)));
        rangeY = max(1, min(round(regionProps(1).BoundingBox(1)):(regionProps(1).BoundingBox(1)+regionProps(1).BoundingBox(3)), size(maskImage,2)));
        rangeXGlobal = (1:length(rangeX)) + objects(o,t).position(1) - round(regionProps(1).BoundingBox(3)/2);
        rangeYGlobal = (1:length(rangeY)) + objects(o,t).position(2) - round(regionProps(1).BoundingBox(4)/2);

        %% add the current mask to the result image           
        resultImage(rangeXGlobal, rangeYGlobal) = max(resultImage(rangeXGlobal, rangeYGlobal), o * (maskImage(rangeX, rangeY) > 0));
        
        
        currentStage = objects(o,t).stages(end);
        currentIntensity = round(settings.meanIntensities(currentStage) + objects(o,t).intensityStdDev);
                
        resultImageIntensity(rangeXGlobal, rangeYGlobal) = max(resultImageIntensity(rangeXGlobal, rangeYGlobal), currentIntensity * (maskImage(rangeX, rangeY) > 0));
        resultImageStates(rangeXGlobal, rangeYGlobal) = max(resultImageStates(rangeXGlobal, rangeYGlobal), currentStage * (maskImage(rangeX, rangeY) > 0));
        
        if (~isempty(objects(o,t).rawImage))
            resultImageRaw(rangeXGlobal, rangeYGlobal) = max(resultImageRaw(rangeXGlobal, rangeYGlobal), rawImage(rangeX, rangeY));
        end
    end
    
    %% apply post processing to the generated raw image
    resultImageRaw = PerformPostProcessing(resultImageRaw, settings);
            
    %% write the final result images
    imwrite(uint16(resultImage), sprintf('%s/man_track%03d.tif', settings.outputFolderLabel, t-1));
    imwrite(uint16(65535*double(resultImageRaw)/255), sprintf('%s/t%03d.tif', settings.outputFolderRaw, t-1));
    
    disp(['Finished writing result images ' num2str(t) ' / ' num2str(settings.numFrames)]);
end

%% create the tracking information
fileId = fopen([settings.outputFolderLabel 'man_track.txt'], 'wb');
trackingResult = [];
for o=1:size(objects,1)

    %% find the temporal extent of the current object
    minIndex = inf;
    maxIndex = 0;
    for t=1:settings.numFrames
        if (isempty(objects(o,t).id))
            continue;
        end

        if (t < minIndex)
            minIndex = t;
        end
        if (t >= maxIndex)
            maxIndex = t;
        end
    end
    
    fprintf(fileId, '%i %i %i %i\n', o, minIndex-1, maxIndex-1, max(0, objects(o,minIndex).predecessor));
end

%% close the file id
fclose(fileId);

disp('------');
disp(['Generated raw image data set written to: ' settings.outputFolderRaw]);
disp(['Generated label image data set written to: ' settings.outputFolderLabel]);
disp(['Tracking information in CTC format written to : ' settings.outputFolderLabel '/man_track.txt']);

