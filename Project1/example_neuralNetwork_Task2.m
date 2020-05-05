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
for h = 2:length(nHidden)
    nParams = nParams+nHidden(h-1)*nHidden(h);
end
nParams = nParams+nHidden(end)*nLabels;
w = randn(nParams,1);

% Train with stochastic gradient
maxIter = 100000;
stepSize = 1e-4;
funObj = @(w,i)MLPclassificationLoss_Task3(w,X(i,:),yExpanded(i,:),nHidden,nLabels);
%funObj = @(w,i)MLPclassificationLoss(w,X(i,:),yExpanded(i,:),nHidden,nLabels);
Trainingerror = [];
Validationerror = [];
Trainingiteration = [];
w_pre= 0;
beta = 0.9;
for iter = 1:maxIter
    if mod(iter-1,round(maxIter/20)) == 0
        yhat = MLPclassificationPredict(w,Xvalid,nHidden,nLabels);
        ytrain = MLPclassificationPredict(w,X,nHidden,nLabels);
        fprintf('Training iteration = %d, training error = %f\n,validation error = %f\n',iter-1,sum(ytrain~=y)/t,sum(yhat~=yvalid)/t);
    end
    i = ceil(rand*n);
    [f,g] = funObj(w,i);
    w_temp = w; % Store the weight of the last iteration
    w = w - stepSize * g + beta * (w - w_pre); % Update
    w_pre = w_temp;
end
Trainingerror =  [Trainingerror,sum(ytrain~=y)/t];
Validationerror = [Validationerror,sum(yhat~=yvalid)/t];
Trainingiteration = [Trainingiteration, stepSize];

semilogx(Trainingiteration, [Trainingerror', Validationerror'])
legend('Training', 'Validation')
xlabel('Stepsize','FontSize',12);
ylabel('Error','FontSize',12);

% Evaluate test error
yhat = MLPclassificationPredict(w,Xtest,nHidden,nLabels);
fprintf('Test error with final model = %f\n',sum(yhat~=ytest)/t2);