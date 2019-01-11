% Pattern recognition - Homework 3
% Linear classifier - Dimension reduction approach. 

% Generates 1000 points from each of the 2 classes. Classes are linearly
% separable and have normal distribution. 
% A dimension reduction is performed based on dispersion - from 2 to 1. 
% A vector V is defined as linear transformation that perfoms dimension
% reduction. Scalar v0 is defined as Bayesian minimum error classifier
% treshlod. 

% January, 2019
% Savic Jovana 2013/243

close all
clear all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Generate 1000 points from each of two 2-dimensional classes. 
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
% Dimension reduction based on data points dispersion.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Within class scatter matrix.
Sw = P1*S1 + P2*S2;
% Combined expected value vector.
M0 = P1*M1 + P2*M2;
% Between class scatter matrix.
Sb = P1*(M1-M0)*(M1-M0)';

S1 = Sb; S2 = Sw;
[F, L] = eig(S2^(-1)*S1);

% Take eigenvector for which lambda is maximum.
[l ind] = max(diag(L));
V = F(:, ind);

Y1 = V'*X1;
Y2 = V'*X2;

% Plot histograms to find v0. 
figure, f1 = histogram(Y1, 50, 'Normalization',  'probability'); 
hold on, 
f2 = histogram(Y2, 50, 'Normalization',  'probability'); 
legend('class1', 'class2'),
hold off;

% Everything above v0 is first class, and everything below v0 is second
% class. 
v0 = -2.386; 

% Plot classifer. 
x=-4:0.1:12;
y=-2:0.1:10;
h=zeros(length(x),length(y));
for i=1:length(x)
    X=[x(i)*ones(1,length(y));y];
    h(i,:)=V'*X-v0;
end

figure(1);
hold on
contour(x,y,h',[0 0],'black');
hold off
