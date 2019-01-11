% Pattern recognition - Homework 3
% Linear classifier - First, second and third iterative procedure. 

% Generates 1000 points from each of the 2 classes. Classes are linearly
% separable and have normal distribution. 
% A linear classifier is calculated using first iterative procedure. A
% dependency of error based on parameter s is plotted. Optimal s is found
% and a classifier is calculated and plotted based on it. 
% A linear classifier is calculated using second iterative procedure. A
% dependency of error based on parameter s is plotted. Optimal s is found
% and a classifier is calculated and plotted based on it. 
% A linear classifier is calculated using second iterative procedure. A
% dependency of error based on parameter s is plotted. Optimal s is found
% and a classifier is calculated and plotted based on it. 

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
% First iterative procedure.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

S1 = cov(X1');
S2 = cov(X2');
M1 = mean(X1')';
M2 = mean(X2')';

s = 0:0.001:1;
error = zeros(size(s));
for i=1:length(s)
    % Find vector V.
    V = getV(s(i), M1, M2, S1, S2); 
    % Find other params.
    [var1, var2, v0, eta1, eta2] = findParams(V, M1, M2, S1, S2, s(i));  
    % Find error.
    error(i) = integralError(eta1, sqrt(var1), eta2, sqrt(var2), P1, P2);
end

%Plot error dependency on s.
figure(2),
plot(s, error), title('First iterative procedure : Error dependency'),
xlabel('s'), ylabel('$$\varepsilon$$ (s)', 'Interpreter','latex');

%Find optimal s and parameters that correspond to it.
[e_min ind] = min(error);
s_opt = s(ind);

% Find vector V.
V = getV(s_opt, M1, M2, S1, S2);  
% Find other params.
[var1, var2, v0, eta1, eta2] = findParams(V, M1, M2, S1, S2, s_opt);  

% Plot classifer. 
x=-4:0.1:12;
y=-2:0.1:10;
h=zeros(length(x),length(y));
for i=1:length(x)
    X=[x(i)*ones(1,length(y));y];
    h(i,:)=V'*X+v0;
end

figure(1);
hold on
contour(x,y,h',[0 0],'black');
hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Second iterative procedure.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

s = 0:0.001:1;
error = zeros(size(s));
for i=1:length(s)
    % Find vector V.
    V = getV(s(i), M1, M2, S1, S2); 
    % Find v0 limits.
    [v0_begin, v0_end] = v0Limits(V, X1, X2);
    % Find optimal v0.
    [v0_opt, e_min] = findOptimalv0(v0_begin, v0_end, X1, X2, V);
    error(i) = findErrors(V, v0_opt, X1, X2);
end

%Plot error dependency on s.
figure,
plot(s, error), title('Second iterative procedure : Error dependency');
xlabel('s'), ylabel('$$\varepsilon$$ (s)', 'Interpreter','latex');

%Find optimal s and plot classifier.
[e_min ind] = min(error);
s_opt = s(ind);

V = getV(s_opt, M1, M2, S1, S2); 
var1 = (V')*S1*V;
var2 = (V')*S2*V;
% Find v0.
v0 = -(s*var1*(V'*M2)+(1-s)*var2*(V'*M1))/(s*var1+(1-s)*var2);

% Plot classifer. 
x=-4:0.1:12;
y=-2:0.1:10;
h=zeros(length(x),length(y));
for i=1:length(x)
    X=[x(i)*ones(1,length(y));y];
    h(i,:)=V'*X+v0;
end

figure(1);
hold on
contour(x,y,h',[0 0],'m');
hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Third iterative procedure.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N_training = round(0.8*N); %Take 80% as training set. The rest is test set.
X1_training = X1(:,1:N_training); X1_test = X1(:,N_training+1:end);
X2_training = X2(:,1:N_training); X2_test = X2(:,N_training+1:end);

% Estimate expected values and covariance matrices on traning set.
S1 = cov(X1_training');
S2 = cov(X2_training');
M1 = mean(X1_training')';
M2 = mean(X2_training')';

s = 0:0.001:1;
error = zeros(size(s));
for i=1:length(s)
    % Find vector V.
    V = getV(s(i), M1, M2, S1, S2); 
    % Find v0 limits.
    [v0_begin, v0_end] = v0Limits(V, X1_training, X2_training);
    % Find optimal v0.
    [v0_opt, e_min] = findOptimalv0(v0_begin, v0_end, X1_training, X2_training, V);
    error(i) = findErrors(V, v0_opt, X1_test, X2_test);
end

%Plot error dependency on s.
figure,
plot(s, error), title('Third iterative procedure : Error dependency');
xlabel('s'), ylabel('$$\varepsilon$$ (s)', 'Interpreter','latex');

%Find optimal s and plot classifier.
[e_min ind] = min(error);
s_opt = s(ind);

V = getV(s_opt, M1, M2, S1, S2); 
var1 = (V')*S1*V;
var2 = (V')*S2*V;
% Find v0.
v0 = -(s*var1*(V'*M2)+(1-s)*var2*(V'*M1))/(s*var1+(1-s)*var2);

% Plot classifer. 
x=-4:0.1:12;
y=-2:0.1:10;
h=zeros(length(x),length(y));
for i=1:length(x)
    X=[x(i)*ones(1,length(y));y];
    h(i,:)=V'*X+v0;
end

figure(1);
hold on
contour(x,y,h',[0 0],'green');
legend('class1', 'class2', 'First procedure', 'Second procedure', 'Third procedure');
hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Helper functions - first iterative procedure. 

function V = getV(s, M1, M2, S1, S2)
%   Finds value of vector V for given expected value vectors and covariance
%   matrices as defined in first iterative procedure.

    V = inv(s*S1+(1-s)*S2)*(M2-M1);
end

function [var1, var2, v0, eta1, eta2] = findParams(V, M1, M2, S1, S2, s)
%   Finds sigmas squared, v0 and etas for given vector V, expected value
%   vector M, covariance matrix S and parameter s as defined in first iterative
%   procedure.

    var1 = V'*S1*V;
    var2 = V'*S2*V;
    
    v0 = -(s*var1*(V'*M2)+(1-s)*var2*(V'*M1))/(s*var1+(1-s)*var2);
    
    eta1 = V'*M1+v0;
    eta2 = V'*M2+v0;
   
end

function eps = integralError(eta1, sigma1, eta2, sigma2, P1, P2)
%   Finds error by calculating integral as defined in first iterative
%   procedure.

    eps1 = 1 - normcdf(-eta1/sigma1);
    eps2 = normcdf(-eta2/sigma2);
    
    eps = P1*eps1+P2*eps2;
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Helper functions - second and third iterative procedure.

function [v_begin, v_end] = v0Limits(V, X1, X2)
%   Finds upper and lower limit of v0 for given vector V and sample sets X1
%   and X2. Used for third iterative procedure.

    N = size(X1,2); %Preallocate for speed.
    y1 = zeros([N 1]); 
    y2 = zeros([N 1]);
    
    for i=1:N
        y1(i) = V'*X1(:,i);
        y2(i) = V'*X2(:,i);
    end
    
    v_begin = -max(max(y1), max(y2));
    v_end = - min(min(y1), min(y2));
    
end

function [v0_opt, e_min]= findOptimalv0(v0_begin, v0_end, X1, X2, V)
%   Finds optimal v0 for given test set X1 and X2 and vector V. Checks
%   values between v0_begin and v0_end.  

    N = size(X1,2); %Preallocate for speed.
    y1 = zeros([N 1]); 
    y2 = zeros([N 1]);

    for i=1:N
        y1(i) = V'*X1(:,i);
        y2(i) = V'*X2(:,i);
    end
    
    v0_delta = (v0_end-v0_begin)/1000;
    v0 = v0_begin:v0_delta:v0_end;  
    for i = 1:size(v0, 2)
        error(i) = sum(y1 >= -v0(i)) + sum(y2 <= -v0(i));
    end
    
    [e_min, ind] = min(error);
    v0_opt = v0(ind);
end

function  error = findErrors(V, v0, X1, X2)
%   Finds the number of data points in test sets that have been wrongly
%   classified for given V and v0. Used in third iterative procedure.
    
    N = size(X1,2); %Preallocate for speed.
    y1 = zeros([N 1]); 
    y2 = zeros([N 1]);

    for i=1:N
        y1(i) = V'*X1(:,i);
        y2(i) = V'*X2(:,i);
    end

    error = sum(y1 >= -v0) + sum(y2 <= -v0);
end
