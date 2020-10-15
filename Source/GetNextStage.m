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