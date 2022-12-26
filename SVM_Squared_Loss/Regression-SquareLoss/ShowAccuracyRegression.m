function MSE = ShowAccuracyRegression(pred, Labels, TrainingOrTesting)

NumPoints = length(Labels); 

MSE = sum((pred - Labels).^2)/NumPoints;    % mean square error

fprintf('MSE is %f over %s samples.\n', MSE , TrainingOrTesting)

end
