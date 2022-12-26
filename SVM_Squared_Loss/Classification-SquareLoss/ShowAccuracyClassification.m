function accuracy = ShowAccuracyClassification(pred, Labels, TrainingOrTesting)

NumPoints = length(Labels); 

accuracy = sum((pred - Labels)==0)/NumPoints*100;

fprintf('The Correct rate is <strong>%0.2f%%</strong> over %s samples\n',...
    accuracy, TrainingOrTesting)

end
