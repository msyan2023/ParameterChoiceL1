function ShowAccuracyClassification(pred, Labels, TrainingOrTesting)

NumPoints = length(Labels); 

fprintf('The Correct rate is <strong>%0.2f%%</strong> over %s samples\n',...
    sum((pred - Labels)==0)/NumPoints*100, TrainingOrTesting)

end
