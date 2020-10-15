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