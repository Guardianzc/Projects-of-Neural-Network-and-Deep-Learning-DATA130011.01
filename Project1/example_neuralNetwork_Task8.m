clc;clear;
load digits.mat
[n,d] = size(X);
nLabels = max(y);
yExpanded = linearInd2Binary(y,nLabels);
t = size(Xvalid,1);
t2 = size(Xtest,1);

% Standardize columns and add bias
[X,mu,sigma] = standardizeCols(X);
X = [ones(n,1) X];
d = d + 1;

% Make sure to apply the same transformation to the validation/test data
Xvalid = standardizeCols(Xvalid,mu,sigma);
Xvalid = [ones(t,1) Xvalid];
Xtest = standardizeCols(Xtest,mu,sigma);
Xtest = [ones(t2,1) Xtest];

% Choose network structure
nHidden = [193];

% Count number of parameters and initialize weights 'w'
nParams = d*nHidden(1);
nBias = nHidden(1);
for h = 2:length(nHidden)
    nParams = nParams+ nHidden(h-1) * nHidden(h);
    nBias = nBias + nHidden(1);
end
nParams = nParams + nHidden(end)*nLabels;
nBias = nBias + nLabels;
w = randn(nParams,1);
b = randn(nBias,1);

% Train with stochastic gradient
maxIter = 100000;
stepSize = 1e-3;
funObj = @(w,b,i)MLPclassificationLoss_Task8(w,b,X(i,:),yExpanded(i,:),nHidden,nLabels);
%funObj = @(w,i)MLPclassificationLoss(w,X(i,:),yExpanded(i,:),nHidden,nLabels);
Trainingerror = [];
Validationerror = [];
Trainingiteration = [];
w_pre = 0;
b_pre = 0;
beta = 0.9;
lambda = 10^(-1);
validation_error = 1;
for iter = 1:maxIter
    if mod(iter-1,round(maxIter/20)) == 0
        % fine tuning
        [~, ~, fpn] = funObj(w,b,1:5000);
        gOutput = (2* (fpn'*fpn) + lambda * eye(size(fpn,2))) \ (2*fpn'*yExpanded);
        w(end - nHidden(end)*nLabels+1: end) = reshape(gOutput(1:end-1,:), nHidden(end)*nLabels,1);
        b(end-nLabels+1:end) = gOutput(end,:)';  
        
        yhat = MLPclassificationPredict_Task6(w,b,Xvalid,nHidden,nLabels);
        ytrain = MLPclassificationPredict_Task6(w,b,X,nHidden,nLabels);
        fprintf('Training iteration = %d, training error = %f\n,validation error = %f\n',iter-1,sum(ytrain~=y)/t,sum(yhat~=yvalid)/t);
        Trainingerror =  [Trainingerror,sum(ytrain~=y)/t];
        Validationerror = [Validationerror,sum(yhat~=yvalid)/t];
        Trainingiteration = [Trainingiteration, iter];    
    end
    i = ceil(rand*n);
   [gradientW, gradientB, fpn] = funObj(w,b,i);
    w = w - stepSize * (gradientW + lambda * w); %+ beta * (w - w_pre); % Update
    b = b - stepSize * gradientB; %+ beta * (b - b_pre); 
end

% fine tuning
[~, ~, fpn] = funObj(w,b,1:5000);
gOutput = (2* (fpn'*fpn) + lambda * eye(size(fpn,2))) \ (2*fpn'*yExpanded);
w(end - nHidden(end)*nLabels+1: end) = reshape(gOutput(1:end-1,:), nHidden(end)*nLabels,1);
b(end-nLabels+1:end) = gOutput(end,:)';  

plot(Trainingiteration, [Trainingerror', Validationerror'])
legend('Training', 'Validation')
xlabel('Iteration','FontSize',12);
ylabel('Error','FontSize',12);

% Evaluate test error
yhat = MLPclassificationPredict_Task6(w,b,Xtest,nHidden,nLabels);
fprintf('Test error with final model = %f\n',sum(yhat~=ytest)/t2);