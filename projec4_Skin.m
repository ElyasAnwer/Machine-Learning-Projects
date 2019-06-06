%%%%This program solves two class , three features classification problem with
% losistic regression using 2nd order Newton's method 
% Skin Segmentation Date Set from UCI, https://archive.ics.uci.edu/ml/datasets/Skin+Segmentation#
% created by Elyas , TTU, 03/25/2018
% Modified by Lily, 03/26/2018
% Modified by Elyas, 03/28/2018
clear all; close all
tic;
M = csvread('skin.csv');
M = M(randperm(size(M, 1)), :); % shuffle the data
n = 1E4;
X = M(1:n,1:3);
y = M(1:n,4);

n_folds = 5; % k = 10 
for ii = 1 : n_folds
    %% train set
    jj = ii-1;
    test_index(ii,:) =  (jj*(size(X,1)/n_folds)+1:ii*(size(X,1)/n_folds));
    train_index(ii,:) = setdiff(1:size(X,1),test_index(ii,:));

    Y_in_train = y(train_index(ii,:));
    X_in_train = X(train_index(ii,:),:);
    
    train_skin_ind = find (Y_in_train == 1);
    nc1 = length(train_skin_ind );
    train_non_skin_ind = find (Y_in_train == 0);
    nc2 = length(train_non_skin_ind);
    n = nc1 + nc2;
    y_train = [ones(nc1,1); zeros(nc2,1)];

    train_skin_features = X_in_train(train_skin_ind,:);
    train_non_skin_features = X_in_train(train_non_skin_ind,:);

    xsort_train = [train_skin_features;train_non_skin_features];
    x_train = [ones(n,1) xsort_train];

    weight_vec = zeros(1,size(X,2)+1);
    max_ite_newton = 20;
    [weight_vec,cost_val,iter,Y_pre] = newtonGradient(weight_vec,x_train,y_train,max_ite_newton);
    fprintf('Iterations took = %d \n',iter );
    
    figure
    scatter3(train_skin_features(:,1),train_skin_features(:,2),train_skin_features(:,3),'MarkerFaceColor',[0 .75 .75]); hold on
    scatter3(train_non_skin_features(:,1),train_non_skin_features(:,2),train_non_skin_features(:,3),'MarkerFaceColor',[1 .5 .5])
    [x1p,x2p] = meshgrid(0:5:300, 0:5:300);
    x3 = -weight_vec(1)/weight_vec(4) -(weight_vec(2)/weight_vec(4))*x1p - (weight_vec(3)/weight_vec(4))*x2p;
    surf(x1p,x2p,x3)
    xlabel('Intensity of Blue'); ylabel('Intesity of Green'); zlabel('Intesity of Red');
    legend('Skin','Non-skin','Decision plane (Newtons method)')
    title (['Train set ' num2str(ii) '( Number of training samples = ' num2str(n) ')' ])


   %% test set
    Y_in_test = y(test_index(ii,:));
    X_in_test = X(test_index(ii,:),:);

    test_skin_ind = find (Y_in_test == 1);
    test_non_skin_ind = find (Y_in_test == 0);
    nc1_test = length(test_skin_ind ) ;
    nc2_test = length(test_non_skin_ind);
    n_test = nc1_test + nc2_test;
    y_test = [ones(nc1_test,1); zeros(nc2_test,1)];

    test_skin_features = X_in_test(test_skin_ind,:);
    test_non_skin_features = X_in_test(test_non_skin_ind,:);

    figure
    scatter3(test_skin_features(:,1),test_skin_features(:,2),test_skin_features(:,3),'MarkerFaceColor',[0 .75 .75]); hold on
    scatter3(test_non_skin_features(:,1),test_non_skin_features(:,2),test_non_skin_features(:,3),'MarkerFaceColor',[1 .5 .5])
    surf(x1p,x2p,x3)
    xlabel('Intensity of Blue'); ylabel('Intesity of Green'); zlabel('Intesity of Red');
    
    xsort_test = [test_skin_features;test_non_skin_features];
    x_test = [ones(n_test,1) xsort_test];
    
    Y_pre_test = 1./(1 + exp(-weight_vec*x_test'));

    [accuracy,index_mis,mis] = misclass(y_test, Y_pre_test,n_test);
    scatter3(x_test(index_mis,2),x_test(index_mis,3),x_test(index_mis,4),'kx')
    legend('Skin','Non-skin','Decision plane (Newtons method)','misclassified data')
    title (['Test fold ' num2str(ii) ' with accuracy = ' num2str(accuracy) '% ( Number of testing samples = ' num2str(n_test) ')'  ])
    accu_all(ii) = accuracy;
end
Avg_accuracy = mean (accu_all);
fprintf('Average accuracy of classifier = %f \n',Avg_accuracy);
elapsedTime = toc
%% Newtons-method to solve gradient descent
function [w,J,iter,h_x] = newtonGradient(w_ini,x,y,max_iter)
w = w_ini;
% J = zeros(length(y),1);
for i = 1 : max_iter
    %sigmoid
    h_x = 1./(1 + exp(-w*x'));
    %cost function
    J(i) = -(1/size(x,1))*sum(y.*log(h_x)' + (1-y).*log(1-h_x)');
    %gradient
    grad = (1/size(x,1))*((h_x - y')*x);
    if norm(grad) <= 1E-6
        break
    end
    %hessian
    H = (1/size(x,1))*(x'*diag(h_x)*diag(1-h_x)*x);
    %update weight vectos
    w = w - (pinv(H)*grad')';
end
iter = length(J);
end

%% Compute # of misclassification
% sample in class 1 has value 1 and 0 if it's in class 2
function [accuracy,index_mis,mis] = misclass(y, Y_pre,n)
mis = 0;
ii = 0;
for i = 1:n
    if y(i) == 1 && Y_pre(i) <= 0.5
        ii = ii +1;
        mis = mis + 1;
        index_mis(ii) = i;
    elseif y(i) == 0 && Y_pre(i) >= 0.5
        ii = ii +1;
        mis = mis + 1;
        index_mis(ii) = i;
    end
end
fprintf('Number of misclassified data = %d \n',mis );
accuracy = 100*(size(y,1)- mis)/size(y,1);
fprintf('Accuracy of classifier = %f \n',accuracy);
end
