%%%%This program solves two class , two features classification problem with
% losistic regression using 2nd order Newton's method
% Matlab built in patients data set (smoker data set)
% created by Elyas , TTU, 03/25/2018
% Modified by Lily, 03/26/2018
% Modified by Elyas, 03/28/2018
clear all; close all
tic;
load patients
rng(33)
%% K fold cross validation 
rand_index = randperm(size(Smoker,1));
n_folds = 5; % k = 5 
for ii = 1 : n_folds
    %% train set
    jj = ii-1;
    
    test_index(ii,:) = rand_index (jj*(size(Smoker,1)/n_folds)+1:ii*(size(Smoker,1)/n_folds));
    
    train_index(ii,:) = setdiff(rand_index,test_index(ii,:));
    
    Y_in_train = Smoker(train_index(ii,:));
    
    X1_in_train = Diastolic(train_index(ii,:));
    
    X2_in_train = Systolic(train_index(ii,:));
    
    train_smoker_ind = find (Y_in_train == 1);
    nc1 = length(train_smoker_ind );
    train_non_smoker_ind = find (Y_in_train == 0);
    nc2 = length(train_non_smoker_ind);
    n = nc1 + nc2;
    y_train = [ones(nc1,1); zeros(nc2,1)];
    
    train_smoker_features=[X1_in_train(train_smoker_ind) X2_in_train(train_smoker_ind)];
    train_non_smoker_features=[X1_in_train(train_non_smoker_ind) X2_in_train(train_non_smoker_ind)];
    
    phi_train = [train_smoker_features;train_non_smoker_features];
    padones = ones(size (phi_train,1),1);
    x_train = [padones phi_train];
    
    weight_vec = [0 0 0];
    max_ite_newton = 20;
    [weight_vec,cost_val,iter,Y_pre] = newtonGradient(weight_vec,x_train,y_train,max_ite_newton);
    fprintf('Iterations took = %d \n',iter );
    
    figure
    plot(train_smoker_features(:,1),train_smoker_features(:,2),'r*');hold on
    plot(train_non_smoker_features(:,1),train_non_smoker_features(:,2),'o');
    x_plot=linspace(min(x_train(:,2)),max(x_train(:,2)),10);
    x2 = -weight_vec(1)/weight_vec(3) -(weight_vec(2)/weight_vec(3))*x_plot;
    plot(x_plot,x2,'r')
    xlabel('Diastolic'); ylabel('Systolic');
    legend('Smoker','Non-smoker','Decision boundary (Newtons method)')
    title (['Train set ' num2str(ii) '(Number of training samples = ' num2str(n) ')' ])
%     axis tight
    
    %% test set
    Y_in_test = Smoker(test_index(ii,:));
    
    X1_in_test = Diastolic(test_index(ii,:));
    
    X2_in_test = Systolic(test_index(ii,:));
    
    test_smoker_ind = find (Y_in_test == 1);
    test_non_smoker_ind = find (Y_in_test == 0);
    nc1_test = length(test_smoker_ind ) ;
    nc2_test = length(test_non_smoker_ind);
    n_test = nc1_test + nc2_test;
    y_test = [ones(nc1_test,1); zeros(nc2_test,1)];
    
    test_smoker_features=[X1_in_test(test_smoker_ind) X2_in_test(test_smoker_ind)];
    test_non_smoker_features=[X1_in_test(test_non_smoker_ind) X2_in_test(test_non_smoker_ind)];
    
    figure
    plot(test_smoker_features(:,1),test_smoker_features(:,2),'r*');hold on
    plot(test_non_smoker_features(:,1),test_non_smoker_features(:,2),'o');
    plot(x_plot,x2,'r')
    xlabel('Diastolic'); ylabel('Systolic');
    
    phi_test = [test_smoker_features;test_non_smoker_features];
    padones = ones(size (phi_test,1),1);
    x_test = [padones phi_test];
    
    Y_pre_test = 1./(1 + exp(-weight_vec*x_test'));
    
    [accuracy,index_mis,mis] = misclass(y_test, Y_pre_test,n_test);
    plot(x_test(index_mis,2),x_test(index_mis,3),'ks','MarkerSize',10)
    legend('Smoker','Non-smoker','Decision boundary (Newtons method)','misclassified data')
    title (['Test fold ' num2str(ii) ' with accuracy = ' num2str(accuracy) '% (Number of testing samples = ' num2str(n_test) ')' ])
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




