% Project 5 ML: Feed-forward neural network and back propagation to do classification for XOR
% Hidden layer activation : Relu 
% Output layer activation : Sigmoid
% Input is 2-D so 2 input nodes
% Output is 1-D so 1 output nodes
% Hidden node is 2 as required 
% Elyas Anwer, 04/19/2018

clear all;close all
rng(100)
LR = 0.01 ; beta = 0.8; max_ite = 5000;
X = [0 0 1 1; 0 1 0 1];
Y = [ 0 1 1 0];
N = length(Y);

input_nodes = 2 ; hidden_nodes = 2; output_nodes = 1;

%%%% Xavier weight initizilation %%%% 
% W_1 = randn(hidden_nodes,input_nodes)*sqrt(2/input_nodes);
% W_2 = randn(output_nodes,hidden_nodes)*sqrt(2/hidden_nodes);

%%%% Normal weight initizilation %%%%
W_1 = rand(hidden_nodes,input_nodes);
W_2 = rand(output_nodes,hidden_nodes);
B_1 = zeros(hidden_nodes,1);
B_2 = zeros(output_nodes,1);

Vdw_1 = zeros(hidden_nodes,input_nodes);
Vdw_2 = zeros(output_nodes,hidden_nodes);
Vdb_1 = zeros(hidden_nodes,1);
Vdb_2 = zeros(output_nodes,1);

Cost = zeros(1,max_ite);
for i = 1 : max_ite
    
    A_1 = W_1 * X + repmat(B_1,1,N);
    %     Z_1 = 1 ./(1 + exp(-A_1));Sigmoid activation hidden layer
    Z_1 = A_1;
    Z_1 (Z_1<0) = 0; % Relu activation hidden layer
    
    A_2 = W_2 * Z_1 + repmat(B_2,1,N);
    Z_2 = 1 ./(1 + exp(-A_2));% Sigmoid activation output layer
    
    % back_propogation
    del_2 = Z_2 - Y;
    %     de_2_acti = Z_1.*(1-Z_1); sigmoid derivative
    de_2_acti = A_1;
    de_2_acti (de_2_acti>0) = 1;  % Relu derivative
    de_2_acti (de_2_acti<=0) = 0; % Relu derivative
    del_1 = (W_2'*del_2).*de_2_acti;
    
    dw_2 = del_2*Z_1';
    dw_1 = del_1*X';
    db_2 = sum(del_2,2);
    db_1 = sum(del_1,2);
    
    Vdw_2 = beta*Vdw_2 + (1-beta)*dw_2;
    Vdw_1 = beta*Vdw_1 + (1-beta)*dw_1;
    Vdb_2 = beta*Vdb_2 + (1-beta)*db_2;
    Vdb_1 = beta*Vdb_1 + (1-beta)*db_1;

    W_2 = W_2 - LR*Vdw_2;
    W_1 = W_1 - LR*Vdw_1;
    B_2 = B_2 - LR*Vdb_2;
    B_1 = B_1 - LR*Vdb_1;
    
    Cost(i) = 0.5*sum((del_2).^2);
%     figure(1)
%     plot(i,Cost(i),'r.');hold on
end
% Predic_Y = Z_2
%% plot decision boundary
x1 = -2:0.05:2;
x2 = -2:0.05:2;
matrix_pre = zeros(length(x1),length(x2));

for i = 1 : length(x1)
    for j = 1 : length(x2)
        Predic = forwardNN_RElu(W_1,W_2,B_1,B_2,[x1(i);x2(j)]);
        matrix_pre (i,j) = Predic;
    end
end

figure(3)
pcolor(x1,x2,matrix_pre);hold on; colormap jet; colorbar;
plot(0,0,'rx','MarkerFaceColor',[.5 1 .5],'MarkerSize',10)
plot(1,1,'rx','MarkerFaceColor',[.5 1 .5],'MarkerSize',10)
plot(0,1,'bo','MarkerFaceColor',[0 1 0],'MarkerSize',10)
plot(1,0,'bo','MarkerFaceColor',[0 1 0],'MarkerSize',10)
xlabel('X1');ylabel('X2')
%% calculate forward-pass
function Predic = forwardNN_RElu(W_1,W_2,B_1,B_2,X)
A_1 = W_1 * X + repmat(B_1,1,1);
Z_1 = A_1;
Z_1 (Z_1<0) = 0;

A_2 = W_2 * Z_1 + repmat(B_2,1,1);
Predic = 1 ./(1 + exp(-A_2));
end




