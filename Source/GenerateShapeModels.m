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
        shapeCovMatrices{i} = zeros(numSamplingPoints, numSamplingPoints);
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
    numShapes = zeros(numStages, 3);

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
            shapeCovMatrices{currentStage} = shapeCovMatrices{currentStage} + (demeanedSamplingPoints * demeanedSamplingPoints');
            numShapes(currentStage) = numShapes(currentStage) + 1;
        end
    end

    %% normalize the covariance matrix result
    for i=1:numStages
        shapeCovMatrices{i} = shapeCovMatrices{i} / numShapes(i);
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
    shapeModel.eigenVectors = eigenVectors;
    shapeModel.eigenValues = eigenValues;
    
    %% plot a few random shapes
    if (settings.debugFigures == true)
        for i=1:100
            figure(3); clf; hold off;

            for s=1:numStages
                subplot(1,numStages,s);

                currentShape = meanShapes{s};
                for j=1:numSamplingPoints
                    currentShape = currentShape + randn * eigenVectors{s}(:,j) * sqrt(eigenValues{s}(j,j));

                end

                plot(currentShape(:,2), currentShape(:,1), '*r'); hold on;

                currentBoundary = boundary(currentShape(:,2), currentShape(:,1), 0.5);
                plot(currentShape(currentBoundary,2), currentShape(currentBoundary,1));
                axis equal;
                axis([-30, 30, -30, 30]);
                drawnow;

                title(['Shape model ' num2str(s)]);
            end

            pause(1);
        end
    end
end