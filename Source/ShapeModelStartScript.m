

%% approximation from the data
%% 0.9601    0.2995    0.5394    0.8020    0.5442    0.7542 %% probability of self-cycle
%% 0.0399    0.7005    0.4606    0.1980    0.4558    0.2458 %% probability of stage transition


%% Define the model with 6 states (interphase, prophase, prometaphase, metaphase, anaphase, telophase, interphase).
transitionMatrix = [0.96, 0.04, 0.0, 0.0, 0.0, 0.0; ...     %% interphase
                    0.0, 0.3, 0.7, 0.0, 0.0, 0.0; ...       %% prometaphase
                    0.0, 0.0, 0.46, 0.54, 0.0, 0.0; ...       %% prophase
                    0.0, 0.0, 0.0, 0.8, 0.2, 0.0; ...       %% metaphase
                    0.0, 0.0, 0.0, 0.0, 0.54, 0.46; ...       %% anaphase
                    0.25, 0.0, 0.0, 0.0, 0.0, 0.75];          %% telophase
%                 
%                 
% emissionMatrix = eye(6);

% %% Define the model with 6 states (interphase, prophase, prometaphase, metaphase, anaphase, telophase, interphase).
% transitionMatrix = [0.97, 0.03, 0.0; ...     %% interphase
%                     0.0, 0.9, 0.1; ...       %% prometaphase
%                     0.1, 0.0, 0.9];          %% telophase
               
emissionMatrix = eye(3);

createNewShapeModel = false;

clear settings;
settings.transitionMatrix = transitionMatrix;
settings.emissionMatrix = emissionMatrix;
settings.imageSize = [1024, 1024];
settings.numObjects = 20;
settings.numStages = 6;
settings.numSamplingPoints = 50;
settings.numFrames = 100;
settings.borderPadding = 100;
settings.debugFigures = false;
settings.numRefinementSteps = 1;
settings.interPhaseIndex = 1;
settings.proPhaseIndex = 2;
settings.prometaPhaseIndex = 3;
settings.metaPhaseIndex = 4;
settings.anaPhaseIndex = 5;
settings.teloPhaseIndex = 6;
settings.patchSize = 128;
settings.randomizationWithinStageWeight = 0.1;
settings.minStageLengths = [7     1     1     1     1     1];
settings.maxStageLengths = [31    3    30    29     6    20];

inputFolder = 'D:\Projects\2020\CellSimulation_ISBI2021\Data\cells_7domains\';
shapeModelPath = 'shapeModel.mat';

%% extract the statistical shape model for the current set of input images
if (createNewShapeModel == true || ~isfile(shapeModelPath))
    shapeModel = GenerateShapeModels(inputFolder, settings);
    save(shapeModelPath, 'shapeModel');
else
    load(shapeModelPath);
end
settings.shapeModel = shapeModel;

PerformMaskGeneration

