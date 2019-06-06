% This program use closed from and gradient descent method to fit linear model
% to Matlab's carbig data. Elyas,TTU,02/02/2018
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load matlab built in data
close all; 
clear all;
load carbig;
%% eliminate Nan's
[index,b]=find ( isnan(Horsepower) == 1);
Horsepower(index)=[];
Weight(index)=[];

figure(1)
p1 = plot(Weight,Horsepower,'rx','MarkerSize',5);
xlabel('Weight');ylabel('Horsepower')
% normalize data (otherwise there is numerical error,ill-conditioned data)
x1 = Weight/max(Weight);
x0 = ones(size(Weight));
x  = [x0,x1];
t  = Horsepower/max(Horsepower);
%%
% gradient descent implementation
w=[0;0] ; lr = 0.5; misfit = 10;

for i = 1 : 5000
    
    thetax = x * w;
    %cost function gradient
    grad = (1/length(t))*(x'*(thetax-t));
    %weight vector update
    w = w - lr*grad;
    
    %calculate cost function
    J(i) = (x*w-t)'*(x*w-t);
        figure(99)
        plot(i,J(i))
        hold on
  
    if length(J) >1
        misfit = abs(J(i) - J(i-1));
    end
    
    if misfit < 0.00001
        break
    end
    
end

figure(1)
hold on
% map back weight vector
wb(1) =  max(Horsepower)*w(1);
wb(2) = (max(Horsepower)*w(2))/(max(Weight));
yy = wb(1) + wb(2)*Weight;

p2 = plot(Weight,yy,'b','linewidth',2);
legend(p2,'Closed Form')
title('Matlab''s carbig dataset')
%%
% closed form implementation
wc  = pinv(x'*x)*x'*t;
% map back weight vector
wc(1) =  max(Horsepower)*wc(1);
wc(2) = (max(Horsepower)*wc(2))/(max(Weight));
yyc = wc(1) + wc(2)*Weight;

figure(2)
p1 = plot(Weight,Horsepower,'rx','MarkerSize',5);
xlabel('Weight');ylabel('Horsepower')
hold on
p2 = plot(Weight,yyc,'g','linewidth',2);
legend(p2,'Gradient Descent')
title('Matlab''s carbig dataset')

















