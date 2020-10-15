%% initialize the objects in the first frame
clear objects;
objects = struct();
for i=1:settings.numObjects
    objects(i,1).id = i;
    objects(i,1).stages = randperm(settings.numStages,1);
    objects(i,1).rotation = rand * 360;
    objects(i,1).predecessor = -1;
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
            objects(o,t).rotation = mod(prevObject.rotation + randn, 360);
            objects(o,t).predecessor = -1;
        end
    end
end

%% perform interpolation between the different stage models
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
    shapeRandomVariables = 0.75 * randn(settings.numSamplingPoints, numShapeModels);

    %% create the shape models 
    shapeTemplates = cell(numShapeModels,1);
    weightKernels = zeros(numShapeModels, numObjectFrames);
    for i=1:numShapeModels
       shapeTemplates{i} = GenerateRandomShape(shapeModelStages(i), settings, shapeRandomVariables(:,i)); 
       weightKernels(i,:) = normpdf(1:numObjectFrames, stageChanges(i), 1);
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
        currentShape = zeros(settings.numSamplingPoints,2);
        for i=1:numShapeModels
            currentShape = currentShape + weightKernels(i,relativeFrameIndex) * shapeTemplates{i};

            %% create the random shape as a linear combination of the eigenvectors and the mean shape
            for j=1:settings.numSamplingPoints
                currentShape = currentShape + weightKernels(i,relativeFrameIndex) * settings.randomizationWithinStageWeight * randn * settings.shapeModel.eigenVectors{shapeModelStages(i)}(:,j) * sqrt(settings.shapeModel.eigenValues{shapeModelStages(i)}(j,j));
            end

        end

        %% create the mask image            
        maskImage = currentStage * poly2mask(currentShape(:,2) + settings.patchSize/2, currentShape(:,1) + settings.patchSize/2, settings.patchSize, settings.patchSize);

        objects(o,s).maskImage = maskImage;
        objects(o,s).currentShape = currentShape;
    end

    disp(['Finished creating shapes for ' num2str(o) ' / ' num2str(size(objects,1)) ' objects ...']);
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

        %% find properties of the mask for repulsive computations
        regionProps = regionprops(maskImage>0, 'BoundingBox', 'Centroid', 'EquivDiameter', 'MinorAxisLength', 'MajorAxisLength', 'Orientation');

        %% apply cell division orthogonal to the major axis
        if (objects(o,t).predecessor > 0)

            %% compute the orientation vector
            currentOrientation = deg2rad(regionProps(1).Orientation);
            directionVector = [cos(currentOrientation), sin(currentOrientation)];

            %% split objects
            objects(o,t).position = objects(o,t).position + (-1)^(mod(o,2)) * 0.4 * regionProps(1).MinorAxisLength * directionVector;
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
        objects(o,t).position = max(settings.borderPadding, min(round(objects(o,t).position + repulsiveForce + 2*randn(1,2)), settings.imageSize(1)-settings.borderPadding));

        %% compute ranges of the input patch that correspond to the global result image coordinates
        rangeX = max(1, min(round(regionProps(1).BoundingBox(2)):(regionProps(1).BoundingBox(2)+regionProps(1).BoundingBox(4)), size(maskImage,1)));
        rangeY = max(1, min(round(regionProps(1).BoundingBox(1)):(regionProps(1).BoundingBox(1)+regionProps(1).BoundingBox(3)), size(maskImage,2)));
        rangeXGlobal = (1:length(rangeX)) + objects(o,t).position(1) - round(regionProps(1).BoundingBox(3)/2);
        rangeYGlobal = (1:length(rangeY)) + objects(o,t).position(2) - round(regionProps(1).BoundingBox(4)/2);

        %% add the current mask to the result image           
        resultImage(rangeXGlobal, rangeYGlobal) = max(resultImage(rangeXGlobal, rangeYGlobal), o * (maskImage(rangeX, rangeY) > 0));
    end

    %% plot final result images if enabled
    %if (settings.debugFigures == true)
        figure(2);
        imagesc(max(resultImage, max(resultImage(:))*(imgradient(resultImage) > 0)));
        axis equal;
        pause(0.5);
   % end
end