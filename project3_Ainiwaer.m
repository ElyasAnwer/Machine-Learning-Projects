% This program plot variance vs bias tradeoff graph using examples from
% Bishop's book Ch.3,P151%% Elyas, 02/28/2018

clear all;close all;clc
% set up initial parameters Nt: number of train samples s: spacial scale(with
% of gaussian) for gassian basis functions L: number of data sets
% N_test:number of test samples noise_sigma: gussian noise parameter
Nt = 25; s = 0.1; L = 100; N_test = 1000; noise_sigma = 0.3;
lambda = linspace(0.1,7,20);
 rng(181)
%% construct gaussian basis fucntion phi for train set
% cu_se=rng
x_train = rand(Nt,1);
x_train = sort(x_train);
phi = zeros(Nt,Nt);
y_train = sin(2*pi*x_train) + noise_sigma*randn(Nt,L);
for i = 1 : length(x_train)
    for j = 1 : length(x_train)
        phi (i,j) = exp(-(x_train(i)-x_train(j)).^2/(2*s.^2));
    end
end
Phi =[ones(Nt,1), phi];
%% construct gaussian basis function for test set
%cu_se=rng
% rng(cu_se)
x_test = rand(N_test,1);
x_test = sort(x_test);
phitest = zeros(Nt,N_test);
y_test =sin(2*pi*x_test) + noise_sigma*randn(N_test,1);
for i = 1 : length(x_train)
    for j = 1 : length(x_test)
        phitest (i,j) = exp(-(x_train(i)-x_test(j)).^2/(2*s.^2));
    end
end
Phitest = [ones(1,N_test);phitest];
%% test for different regulerization parameters within 100 data sets
bia_sq = zeros(length(lambda),1);
variance = zeros(length(lambda),1);
bias_variance = zeros(length(lambda),1);
Test_avg_err = zeros(length(lambda),1);
for iii = 1 : length(lambda)
    y_pre = zeros(Nt,L);
    Test_error = zeros(1,L);
    for ii = 1 : L
%                 y_train = sin(2*pi*x_train) + noise_sigma*randn(Nt,1);
%                 figure(1)
%                 plot(x_train,y_train)
%                 hold on
        % solve for weight vectors using RLS
        W = pinv(Phi'*Phi + lambda(iii)*eye(Nt+1))*Phi'*y_train(:,ii);
        y_pre(:,ii) = Phi * W;
        %                 figure(3)
        %                 plot(x_train,y_pre)
        %compute test error with regularization
        Test_error (:,ii) = sum((Phitest'*W-y_test).^2)/N_test + (lambda(iii)*W'*W)/N_test;
    end
    Test_avg_err(iii) = mean(Test_error);
    y_avg = mean(y_pre,2);
    bia_sq(iii) = mean((y_avg - sin(2*pi*x_train)).^2);
%             figure(6)
%         plot(x_train,y_avg,'r','linewidth',2);hold on;plot(x_train,sin(2*pi*x_train))
    for jjj = 1 : L
       dif_val(jjj) = sum((y_avg - y_pre(:,L)).^2)/Nt;
    end
    variance(iii) = mean(dif_val);
    bias_variance(iii) = bia_sq(iii) + variance(iii);
end
%% plot variance vs bias trade-off graph
figure(9)
plot(log(lambda),bia_sq,'b','linewidth',1.5)
hold on
plot(log(lambda),variance,'r','linewidth',1.5)
plot(log(lambda),bias_variance,'m','linewidth',1.5)
plot(log(lambda),Test_avg_err,'k','linewidth',1.5)
legend('(bias)^2','variance','(bias)^2 + variance','test error','Location','northwest')
xlabel('ln(\lambda)');legend boxoff; axis([-3 2 0 0.16])





