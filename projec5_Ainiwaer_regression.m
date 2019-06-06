% Project 5 ML: Feed-forward neural network and back propagation to do regression,
% Hidden layer activation : tanh
% Output layer activation : Identity
% Input is 1-D so 1 input node
% Output is 1-D so 1 output node
% Hidden node is 3 as required , can be increased to fit complex model
% Elyas Anwer, 04/19/2018

clear all; close all;
% LR = 0.01; beta = 0.8; max_ite = 5000 ;
% input_nodes = 1 ; hidden_nodes = 3; output_nodes = 1;

NN(0.01,0.8,5000,1,3,1,1)
NN(0.01,0.8,5000,1,20,1,2)
% T = X.^2; % calculate target
% T = abs(X); % calculate target
% T = sign(X); % calculate target
function NN(LR,beta,max_ite,input_nodes,hidden_nodes,output_nodes,fig_num)
rng(100)
X=2*rand(1,50)-1;
T=sin(2*pi*X)+0.3*randn(1,50);
N = length(T);

figure(fig_num)
plot(X,T,'o');hold on

%%%% Xavier weight initizilation %%%%
% W_1 = randn(hidden_nodes,input_nodes)*sqrt(1/input_nodes);
% W_2 = randn(output_nodes,hidden_nodes)*sqrt(1/hidden_nodes);

%%%% Normal weight initizilation %%%%
W_1 = randn(hidden_nodes,input_nodes);
W_2 = randn(output_nodes,hidden_nodes);
B_1 = zeros(hidden_nodes,1);
B_2 = zeros(output_nodes,1);

Vdw_1 = zeros(hidden_nodes,input_nodes);
Vdw_2 = zeros(output_nodes,hidden_nodes);
Vdb_1 = zeros(hidden_nodes,1);
Vdb_2 = zeros(output_nodes,1);

Cost = zeros(1,max_ite);
for i = 1 : max_ite
    % forward propogation
    A_1 = W_1 * X + repmat(B_1,1,N);
    Z_1 = (exp(A_1) - exp(-A_1))./(exp(A_1) + exp(-A_1));
    
    A_2 = W_2 * Z_1 + repmat(B_2,1,N);
    Z_2 = A_2;
    %back propogation
    del_2 = Z_2 - T;
    del_1 = (W_2'*del_2).*(1-Z_1.^2);
    
    %gradient
    dw_2 = del_2*Z_1';
    dw_1 = del_1*X';
    db_2 = sum(del_2,2);
    db_1 = sum(del_1,2);
    
    %batch gradient with momentum
    Vdw_2 = beta*Vdw_2 + (1-beta)*dw_2;
    Vdw_1 = beta*Vdw_1 + (1-beta)*dw_1;
    Vdb_2 = beta*Vdb_2 + (1-beta)*db_2;
    Vdb_1 = beta*Vdb_1 + (1-beta)*db_1;
    
    W_2 = W_2 - LR*Vdw_2;
    W_1 = W_1 - LR*Vdw_1;
    B_2 = B_2 - LR*Vdb_2;
    B_1 = B_1 - LR*Vdb_1;
    
    %error calculation
    Cost(i) = 0.5*sum((del_2).^2)/N;
%     figure(3)
%     plot(i,Cost(i),'r.');hold on
end

%% plot model
x_pre =linspace(-1,1,100);
y_pre =forwardNN_reg(W_1,W_2,B_1,B_2,x_pre);

figure(fig_num)
plot(x_pre,y_pre,'r')
xlabel('Input');ylabel('Output')
title (['Training error=' num2str(Cost(end)) ', Hidden layers=' num2str(hidden_nodes)])
end
%% calculate forward-pass
function Predic = forwardNN_reg(W_1,W_2,B_1,B_2,X)
A_1 = W_1 * X + repmat(B_1,1,1);
Z_1 = (exp(A_1) - exp(-A_1))./(exp(A_1) + exp(-A_1));

A_2 = W_2 * Z_1 + repmat(B_2,1,1);
Predic = A_2;
end


