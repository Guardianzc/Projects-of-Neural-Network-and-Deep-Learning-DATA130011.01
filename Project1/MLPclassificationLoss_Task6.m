function [gradientW, gradientB] = MLPclassificationLoss_Task6(w,b,X,y,nHidden,nLabels)

[nInstances,nVars] = size(X);

% Form Weights
l_Hidden = length(nHidden);
offsetw = nVars*nHidden(1);
offsetb = nHidden(1);
Bias1 = b(1:offsetb)';
Weights1 = reshape(w(1:offsetw),nVars,nHidden(1));
hiddenWeights{1} = [Weights1;Bias1];
outputWeights = reshape(w(offsetw+1:offsetw+ nHidden(end)*nLabels) ,nHidden(end),nLabels);
outputBias  = b(offsetb+1:end)';
outputWeights = [outputWeights; outputBias];
% store the gradient
gradientW = zeros(size(w));
gradientB = zeros(size(b));

% Compute Output
% Inoput weight
ip{1} = [X, ones(nInstances,1)] * hiddenWeights{1};
fp{1} = tanh(ip{1});
% Hidden layer
for h = 2:length(nHidden)
    matrix_size = nHidden(k-1) * nHidden(k);
    HiddenWeights = reshape(w(offsetw+1:offsetw+matrix_size), nHidden(k-1), nHidden(k));
    HiddenBias = b(offsetb+1:offsetb+nHidden(k))';
    HiddenWeights = [HiddenWeights;HiddenBias];
    ip{h} = [fp{h-1},ones(nInstances,1)] * HiddenWeights;
    fp{h} = tanh(ip{h});
    offsetw = offsetw + matrix_size;
    offsetb = offsetb + nHidden(k);
end
yhat = [fp{end}, ones(nInstances,1)] * outputWeights;

relativeErr = yhat - y;
if nargout > 1
    pyhat = exp(yhat) / sum(exp(yhat));
    True = ( y == 1);
    err = pyhat - True;
    % Output Weights
    gOutput = [fp{end}, ones(nInstances,1)]'* err;
    len = nHidden(l_Hidden) * nLabels;
    gradientW(offsetw+1:offsetw+nHidden(end)*nLabels) = reshape(gOutput(1:end-1,:), len,1);
    gradientB(offsetb+1:end) = gOutput(end,:)';
    % Hidden layers
    if length(nHidden) > 1
        % Last Layer of Hidden Weights
        clear backprop
        h = len(Hidden);
        len = nHidden(h) * nHidden(h-1);
        backprop = err * (sech(ip{end}).^2 .* outputWeights(1:end-1,:)');
        gHidden = [fp{end-1}, ones(nInstances,1)]'* backprop;
        offsetw = offsetw - len;
        offsetb = offsetb - len;
        % store the gradient
        gradientW(offsetw+1:offsetw+len) = reshape(gHidden(1:end-1,:), len,1);
        gradientB(offsetb+1:offsetb+len) = gHidden(end,:)';
        % backprop = sum(backprop,1);

        % Other Hidden Layers
        for h = length(nHidden)-1:-1:2
            backprop = (backprop*hiddenWeights{h+1}(1:end-1,:)').*sech(ip{h+1}).^2;
            gHidden = [fp{h}, ones(nInstances,1)]'*backprop;
            offsetw = offsetw - len;
            offsetb = offsetb - len;
            gradientW(offsetw+1:offsetw+len) = reshape(gHidden(1:end-1,:), len,1);
            gradientB(offsetb+1:offsetb+len) = gHidden(end,:)';
        end

        % Input Weights
        backprop = (backprop*hiddenWeights{2}(1:end-1,:)').* sech(ip{1}).^2;
        len = nHidden(1) * nVars;
        gInput = [X, ones(nInstances,1)]'*backprop;
        offsetw = offsetw - len;
        offsetb = offsetb - len;
        gradientW(offsetw+1:offsetw+len) = reshape(gInput(1:end-1,:), len,1);
        gradientB(offsetb+1:offsetb+len) = gInput(end,:)';
    else
       % Input Weights
        backprop = err * (outputWeights(1:end-1,:)').* sech(ip{1}).^2;
        len = nHidden(1) * nVars;
        gInput = [X, ones(nInstances,1)]'*backprop;
        gradientW(1:offsetw) = reshape(gInput(1:end-1,:), len,1);
        gradientB(1:offsetb) = gInput(end,:)';
    end  
end
