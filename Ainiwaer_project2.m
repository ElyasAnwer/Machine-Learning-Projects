%  clear all
close all; clc;
N = 50; % number of training samples
x_train = rand(N,1); % generate uniformaly distributed samples
noise = normrnd(0,0.3,N,1); % generate normaly distributed noise with mean: 0 and std:0.3
t_train = sin(2*pi*x_train) + noise;

figure(1);
plot(x_train,t_train,'bo');
hold on
xlabel('x');ylabel('t')
axis([-0.1 1 -1.5 1.5])
plot([0:0.01:1],sin(2*pi*[0:0.01:1]),'g','linewidth',2)
title(['Number of training data: N= ' num2str(N)])
%%
Nt = 100; % number of testing samples
x_test = rand(Nt,1); % generate uniformaly distributed samples
noise = normrnd(0,0.3,Nt,1); % generate normaly distributed noise with mean: 0 and std:0.3
t_test = sin(2*pi*x_test) + noise;

% figure
% plot(x_train,t_train,'bo')

%% ridge rigression added, adjust lamda to make it work, if lamda = 0 becomes traditional
% LS with no regularization
lamda = 0.1;
max_poly_deg = 9;
J_train = zeros(max_poly_deg,1);
E_train_rms = zeros(max_poly_deg,1);
J_test = zeros(max_poly_deg,1);
E_test_rms = zeros(max_poly_deg,1);

for M = 0 : max_poly_deg
    % M =  0; % degree of polynomial fit the data
    
    % construt desing matrix training set
    Q_train = zeros(N,M+1);
    Q_train(:,1) = 1;
    for i = 1 : M
        Q_train(:,i+1) = x_train.^i;
    end
    % do back slah a\b or pinv(a), both perform relatively stable inverse
    % of a matrix
    W = (Q_train'*Q_train +lamda*eye(M+1))\Q_train'*t_train
    plot(x_train,Q_train*W,'r*','linewidth',3)
    
    % construct design matirx for plotting ONLY
    xx = linspace(0,1,50)';
    Qplot = zeros(50,M+1);
    Qplot(:,1) = 1;
    for j = 1 : M
        Qplot(:,j+1) = xx.^j;
    end
    plot(xx,Qplot*W,'k')
    
    J_train(M+1) = 0.5*(Q_train*W - t_train)'*(Q_train*W-t_train); % calculate cost function
    E_train_rms(M+1) = sqrt(J_train(M+1)/N);   % calculate RMS error
%     pause

    Q_test = zeros(Nt,M+1);
    Q_test(:,1) = 1;
    for i = 1 : M
        Q_test(:,i+1) = x_test.^i;
    end
    J_test(M+1) = 0.5*(Q_test*W - t_test)'*(Q_test*W-t_test); % calculate cost function
    E_test_rms(M+1) = sqrt(J_test(M+1)/Nt);   % calculate RMS error
end

figure(3)
hold on
plot(0:1:max_poly_deg,E_train_rms,'bo')
plot(0:1:max_poly_deg,E_test_rms,'ro')
plot(0:1:max_poly_deg,E_train_rms, 'b', 'linewidth',2)
plot(0:1:max_poly_deg,E_test_rms, 'r', 'linewidth',2)
axis([-1 10 0 1])
xlabel('M');ylabel('E-RMS')
legend('Training','Test')
title(['Number of training data: N= ' num2str(N)])



