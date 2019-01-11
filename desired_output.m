% Pattern recognition - Homework 3
% Linear classifier - Desired output approach. 

% Generates 1000 points from each of the 2 classes. Classes are linearly
% separable and have normal distribution. 
% A linear classifier based on desired output is calculated. Results are
% plotted. 

% January, 2019
% Savic Jovana 2013/243

close all
clear all

% Generate 500 points from each of two 2-dimensional classes. 
% Probability density functions are defined as:
% f1(X) = N(M1, S1)
% f2(X) = N(M2, S2)

N = 1000;

M1 = [0.8 1.9]';
M2 = [4.7 5.8]';

S1 = [1 0.7; 0.3 0.5]; 
S2 = [1.3 1.5; 0.8 1.1]; 

P1 = 0.5; P2 = 0.5;

% Apply color transform.
[F1, L1] = eig(S1);
[F2, L2] = eig(S2); 

T1 = F1 * L1^(1/2); 
T2 = F2 * L2^(1/2); 

% Generate first class' data points.
for i = 1:N
    X1(:,i) = T1*randn(2,1)+M1;
end

% Generate second class' data points.
for i = 1:N
    X2(:,i) = T2*randn(2,1)+M2;
end

% Plot data points
figure(1)
plot(X1(1,:),X1(2,:),'r*');
title('Data set');
hold on
plot(X2(1,:), X2(2,:), 'bo');
hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Linear classifier - desired output approach. 
% Find linear classifier based on desired output approach. See what happens
% as Gamma matrix changes. Plot classifiers. 

Z1 = (-1).*[ones(1,N); X1];
Z2 = [ones(1,N); X2];
U = [Z1 Z2];
Gama = ones(2*N, 1);

Gama1 = [2*ones(N,1); ones(N,1)];
Gama2 = [ones(N,1); 2*ones(N,1)];

W = (U*U')^(-1)*U*Gama;
W1 = (U*U')^(-1)*U*Gama1;
W2 = (U*U')^(-1)*U*Gama2;

xp = [-4 12];
figure(1)
hold on
plot(xp, -(W(1)+W(2)*xp)/W(3), 'k');
plot(xp, -(W1(1)+W1(2)*xp)/W1(3), 'g');
plot(xp, -(W2(1)+W2(2)*xp)/W2(3), 'm');
legend('class1', 'class2', 'gamma(Z1) = 1, gamma(Z2) = 1', ... 
    'gamma(Z1) = 2, gamma(Z2) = 1', 'gamma(Z1) = 1, gamma(Z2) = 2');
hold off