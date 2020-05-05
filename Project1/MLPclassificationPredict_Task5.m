function [py1] = MLPclassificationPredict_Task5(w,b,X,nHidden,nLabels)
[nInstances,nVars] = size(X);

% Form Weights
offsetw = nVars*nHidden(1);
offsetb = nHidden(1);
Bias1 = b(1:offsetb)';
Weights1 = reshape(w(1:offsetw),nVars,nHidden(1));
hiddenWeights{1} = [Weights1;Bias1];
ip{1} = [X, ones(nInstances,1)] * hiddenWeights{1};
fp{1} = tanh(ip{1});
for h = 2:length(nHidden)
    % forward method
    matrix_size = nHidden(k-1) * nHidden(k);
    HiddenWeights = reshape(w(offsetw+1:offsetw+matrix_size), nHidden(k-1), nHidden(k));
    HiddenBias = b(offsetb+1:offsetb+nHidden(k))';
    HiddenWeights = [HiddenWeights;HiddenBias];
    ip{h} = [fp{h-1},ones(nInstances,1)] * HiddenWeights;
    fp{h} = tanh(ip{h});
    offsetw = offsetw + matrix_size;
    offsetb = offsetb + nHidden(k);
end
% Compute Output
outputWeights = reshape(w(offsetw+1:offsetw+ nHidden(end)*nLabels) ,nHidden(end),nLabels);
outputBias  = b(offsetb+1:end)';
outputWeights = [outputWeights; outputBias];
yhat = [fp{end}, ones(nInstances,1)] * outputWeights;
[v,py1] = max(yhat,[],2);

%y = binary2LinearInd(y);
