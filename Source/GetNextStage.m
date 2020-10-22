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
function nextStage = GetNextStage(previousStages, settings)

    %% initialize to stage 1 if no history exists
    if (isempty(previousStages))
        previousStages = 1;
    end
    
    %% identify stage transitions
    stageTransitions = find(diff(previousStages));
    
    %% identify the current stage and the duration
    currentStage = previousStages(end);
    if (~isempty(stageTransitions))
        lastStageLength = length(previousStages) - stageTransitions(end);
    else
        lastStageLength = length(previousStages);
    end
    
    %% apply minimum length rule
    if (lastStageLength < settings.minStageLengths(currentStage))
        nextStage = currentStage;
        return;
    end
    
    %% apply maximum length rule
    if (lastStageLength >= settings.maxStageLengths(currentStage))
        nextStage = mod(currentStage, settings.numStages) + 1;
        return;
    end
    
    %% if none of the rules was applied, randomly walk through the markov model
    stageChangeProbability = rand;
    for i=1:settings.numStages
        if (stageChangeProbability >= sum(settings.transitionMatrix(currentStage, 1:(i-1))) && ...
            stageChangeProbability < sum(settings.transitionMatrix(currentStage, 1:i)))
            nextStage = i;
            return;
        end
    end
end