% This script is to show constrained unsupervised learning methods: 
% TC3, TC3+GMM, TC3+GMM+DHMM & TC3+GMM+CHMM, for H2B data.
% 
% Seven sample image sequences reported in the paper. 
% Same image sequences (with more info) are available at www.cellcognition.org.
%
% Qing Zhong, Gerlich Lab, IBC, ETHZ, 2011.08.03, updated 2012.01.10

clear all
addpath(genpath('HMMall')) % using http://www.cs.ubc.ca/~murphyk/Software/HMM/hmm.html
addpath(genpath('helper'))
addpath(genpath('functions'))

%%%%%%%%%%% MODIFIED CODE BELOW %%%%%%%%%%%%%%
outputFolder = [uigetdir filesep];
%%%%%%%%%%% MODIFIED CODE ABOVE %%%%%%%%%%%%%%

% number of classes
k=6;
movieInd = [37,38,39,40,42,45,46]; % seven image sequences
lg = length(movieInd);
defaultStream = RandStream('mt19937ar','seed',0);
savedState = defaultStream.State;
defaultStream.State = savedState;

% choose an image sequence
for s=5:lg
    path = strcat('../data/exp911_3/analyzed/00',num2str(movieInd(s)));

    % feature data and user annotation
    data = readData(path);
    image = readImage(strcat(path,'/_cutter/primary'),'B01.png');
    ua = load(strcat('../data/exp911_3/labels/labels_00', num2str(movieInd(s)),'.txt'));

    %%%%%%%%%%% MODIFIED CODE BELOW %%%%%%%%%%%%%%
    %% compute the number of cells from the montage width
    numCells = size(image,1)/100;
    
    %% extract the individual cells
    for i=1:numCells

        %% initialize result images
        rawImage = zeros(96,96,40);
        maskImage = zeros(96,96,40);
        manualLabels = zeros(1,40);

        %% extract all fourty frames and save them to 3D result images
        for j=1:40

            %% specify the current extraction range
            rangeX = ((i-1)*100+1):(i*100);
            rangeY = ((j-1)*100+1):(j*100);

            %% resize the image to the desired format used for CNN training
            currentImage = imresize(rgb2gray(image(rangeX, rangeY, :)), [96, 96], 'bilinear');

            %% perform segmentation of the centered nucleus
            currentMaskImage = SegmentCenterNucleus(imgaussfilt(currentImage, 2));
            currentMaskImage = imdilate(currentMaskImage, strel('disk', 3));

            %% extract the current label
            currentLabel = ua(i,j);

            %% add current frame to the result images
            rawImage(:,:,j) = currentImage;
            maskImage(:,:,j) = currentMaskImage;
            manualLabels(j) = currentLabel;
        end

        %% assemble the result image file name
        outputFileName = sprintf('%scell_exp=%02d_id=%03d.h5', outputFolder, movieInd(s), i);

        %% save result images in the h5 format
        hdf5write(outputFileName, '/raw_image', uint8(rawImage));
        hdf5write(outputFileName, '/target_image', uint8(maskImage), 'WriteMode', 'append');
        hdf5write(outputFileName, '/manual_stages', uint8(manualLabels), 'WriteMode', 'append');
    end
    %%%%%%%%%%% MODIFIED CODE ABOVE %%%%%%%%%%%%%%
end