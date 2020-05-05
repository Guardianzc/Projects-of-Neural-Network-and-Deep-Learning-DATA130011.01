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
Hidden = [];
Validationerror = [];
Trainingerror = [];
for hid = 1:200
    % Choose network structure
    nHidden = [hid];
    
    % Count number of parameters and initialize weights 'w'
    nParams = d*nHidden(1);
    for h = 2:length(nHidden)
        nParams = nParams+nHidden(h-1)*nHidden(h);
    end
    nParams = nParams+nHidden(end)*nLabels;
    w = randn(nParams,1);
    % Train with stochastic gradient
    maxIter = 100000;
    stepSize = 1e-3;
    funObj = @(w,i)MLPclassificationLoss(w,X(i,:),yExpanded(i,:),nHidden,nLabels);
    for iter = 1:maxIter
        if mod(iter-1,round(maxIter/20)) == 0
            yhat = MLPclassificationPredict(w,Xvalid,nHidden,nLabels);
            ytrain = MLPclassificationPredict(w,X,nHidden,nLabels);
        end
        i = ceil(rand*n);
        [f,g] = funObj(w,i);
        w = w - stepSize*g;
    end
    Trainingerror =  [Trainingerror, sum(ytrain~=y)/t];
    Validationerror = [Validationerror, sum(yhat~=yvalid)/t];
    Hidden = [Hidden, hid];
    % Evaluate test error
    yhat = MLPclassificationPredict(w,Xtest,nHidden,nLabels);
    fprintf('Test error with final model = %f, with hidden = %d\n',sum(yhat~=ytest)/t2, hid);
end
plot(Hidden, [Trainingerror', Validationerror'])
legend('Training', 'Validation')
xlabel('Hidden layer','FontSize',12);
ylabel('Error','FontSize',12);
[min_of_Trainingerror,i_Trainingerror]=min(Trainingerror);
[min_of_Validationerror,i_Validationerror]=min(Validationerror);
fprintf('The best number of hidden layer is %d, with training error %f\n, validation error %f\n ',i_Validationerror, min_of_Trainingerror,min_of_Validationerror);