function [gradientW, gradientB, fpn_store] = CNNclassificationLoss_Task10(w,b,X,y,filter,padding, nLabels)

[nInstances,nVars] = size(X);

% Form Weights
offsetw = filter^2;
offsetb = 0;
CNN = reshape(w(1:offsetw),filter,filter);
outputWeights = reshape( w(offsetw+1:end) ,[] ,nLabels);
outputBias  = b(offsetb+1:end)';
outputWeights = [outputWeights; outputBias];
% store the gradient
gradientW = zeros(size(w));
gradientB = zeros(size(b));
% store the gradient
gOutput = zeros(size(outputWeights,1), size(outputWeights,2));
gCNN = zeros(filter,filter);
for i = 1:nInstances
    % Forward
    % Input weight
    pic = reshape(X(i,:), 16, 16);
    % padding
    pad_pic = padarray(pic, [padding, padding], 0, 'both');
    ip = conv2(pad_pic, CNN, 'valid');
    fp = tanh(ip);
    % reshape fps
    fp_s = reshape(fp, 144, 1);
    % Hidden layer 
    fpn = [fp_s; 1]';
    % store all the fpn to do the finetuning
    fpn_store(i,:) = fpn;
    yhat = fpn * outputWeights;
    % Softmamx
    pyhat = exp(yhat) / sum(exp(yhat));
    True = ( y(i,:) == 1);
    err = pyhat - True;
    % Backward
    % Output Weights
    gOutput = gOutput + reshape(fpn' * err, size(gOutput,1), size(gOutput,2));
    clear backprop
    % gradientW(offsetw+1:offsetw+nHidden(end)*nLabels) = reshape(gOutput(1:end-1,:), len,1);
    % gradientB(offsetb+1:end) = gOutput(end,:)';
    % Input Weights
    % gradient of fp
    backprop = reshape(err * (outputWeights(1:end-1,:)'), size(ip,1), size(ip,2));
    backprop = backprop .* sech(ip).^2;
    % gradient of CNN
    reverseX = reshape(X(i,end:-1:1), 16, 16); 
    gCNN = gCNN + conv2(reverseX, backprop, 'valid');
end 
gOutput = gOutput / nInstances;
gCNN = gCNN / nInstances;
offsetw = filter ^ 2;
gradientW(1:offsetw) = reshape(gCNN, offsetw ,1);
gradientW(offsetw+1:end) = reshape(gOutput(1:end-1,:),length(gradientW) - offsetw,1);
gradientB(1:end) = gOutput(end,:)';
end
