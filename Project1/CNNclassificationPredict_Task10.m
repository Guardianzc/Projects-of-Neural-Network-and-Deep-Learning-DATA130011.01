function [py1] = CNNclassificationPredict_Task10(w,b,X,filter,padding, nLabels)
[nInstances,nVars] = size(X);

% Form Weights
offsetw = filter^2;
offsetb = 0;
CNN = reshape(w(1:offsetw),filter,filter);
outputWeights = reshape( w(offsetw+1:end) ,[] ,nLabels);
outputBias  = b(offsetb+1:end)';
outputWeights = [outputWeights; outputBias];

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
    yhat = fpn * outputWeights;
    % Softmamx
    pyhat(i,:) = exp(yhat) / sum(exp(yhat));
 end
[~,py1] = max(pyhat,[],2);

%y = binary2LinearInd(y);
