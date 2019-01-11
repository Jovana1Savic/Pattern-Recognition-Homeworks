% Pattern recognition - Homework 3
% Quadratic classifier - Desired output approach. 

% Generates 1000 points from each of the 2 classes. Classes aren't linearly
% separable. 
% A quadratic classifier based on desired output is calculated. Results are
% plotted. 

% January, 2019
% Savic Jovana 2013/243


close all
clear all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate 1000 points from each of two 2-dimensional classes. 
N = 1000;

% Generate first class' data points. An elipse.
x1 = 4; y1 = 4;
a1 = 2.4; b1 = 3.2;
for i = 1:N
    theta = rand(1,1)*2*pi;
    R = rand(1,1);
    X1(1,i) = x1 + R*a1*cos(theta);
    X1(2,i) = y1 + R*b1*sin(theta);
end

% Generate second class' data points. An elipse with circle hole.
x2 = 4; y1 = 4;
a2 = 7; b2 = 5.1;
r = 3.6;
for i = 1:N
    theta = rand(1,1)*2*pi;
    R = rand(1,1);
    X2(1,i) = x1 + (r+(a2-r)*R)*cos(theta);
    X2(2,i) = y1 + (r+(b2-r)*R)*sin(theta);
end

% Plot data points
figure(1)
p1 = plot(X1(1,:),X1(2,:),'r*');
title('Data set');
hold on
p2 = plot(X2(1,:), X2(2,:), 'bo');
hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Quadratic classifier - desired output approach. 

Z1 = (-1).*[ones(1,N); X1(1,:).^2; X1(1,:).*X1(2,:); X1(2,:).^2; X1];
Z2 = [ones(1,N); X2(1,:).^2; X2(1,:).*X2(2,:); X2(2,:).^2; X2];
U = [Z1 Z2];
 
Gamma = ones(2*N, 1); % All outputs have equal weight.
Gamma1 = [2*ones(N,1); ones(N,1)]; % Punish first type errors. 
Gamma2 = [ones(N,1); 2*ones(N,1)]; % Punish second type errors. 

[xp, yp] = getQuadraticClassifier(U, Gamma);
[xp1, yp1] = getQuadraticClassifier(U, Gamma1);
[xp2, yp2] = getQuadraticClassifier(U, Gamma2);

% Plot results.

figure(1)
hold on
p3 = plot(xp, yp, 'k-' , 'LineWidth', 1.75);
p4 = plot(xp1, yp1 , 'k--','LineWidth', 1.75);
p5 = plot(xp2, yp2, 'k-.', 'LineWidth', 1.75);
legend([p1 p2 p3(1) p4(1) p5(1)],'class1', 'class2', 'gamma(Z1) = 1, gamma(Z2) = 1', ... 
     'gamma(Z1) = 2, gamma(Z2) = 1', 'gamma(Z1) = 1, gamma(Z2) = 2');
hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Helper functions.

function [xp1, yp1] = getQuadraticClassifier(U, Gamma)
%   Finds quadratic classfier for given U and Gamma (desired output approach). 
%   Returns x and y values.

% Find matrix W. 
W = (U*U')^(-1)*U*Gamma;

% Solve quadratic equaton. 
syms xp yp
eqn = W(1)+W(2)*xp^2+W(3)*xp*yp+W(4)*yp^2+W(5)*xp+W(6)*yp==0;
xp1 =-1:0.01:12;
[yp, param, cond] = solve(eqn, yp, 'Real', true, 'ReturnConditions', true);

% Remove xp values that don't satisfy the conditions. 
xp_limit = subs(cond(1), xp, xp1);
xp_limit_mask1 = logical(xp_limit);

xp_limit = subs(cond(2), xp, xp1);
xp_limit_mask2 = logical(xp_limit);

xp1 = xp1(xp_limit_mask1 & xp_limit_mask2);

% Substitute given values of xp into equation. 
yp1 = subs(yp, xp, xp1);

end