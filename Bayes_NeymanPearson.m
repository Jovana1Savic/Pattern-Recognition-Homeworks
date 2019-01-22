% Pattern recognition - Homework 2
% Bayesian and Neyman-Person's classifier.  

% Generates 500 points from each of the 2 classes. First class has normal
% distribution and the second one bimodal. Plots samples.
% Calculates probability density functions and plots them. Plots constant
% values probability density functions in the sample diagram. 
% Plots Bayesian and Neyman-Pearson's classfiers in the sample diagram. 
% Calculates erros for each classifier. 

close all
clear all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate 500 points from each of two 2-dimensional classes. 
% Probability density functions are defined as:
% f1(X) = N(M1, S1)
% f2(X) = P21 x N(M21, S21) + P22 x N(M22, S22)

P21 = 0.35; %P22 = 1 - P21;

M1 = [7 4.8]';
M21 = [3 4.2]'; M22 = [4.4 1.3]';

S1 = [0.8 0.7; 0.2 0.4]; 
S21 = [1.3 1.5; 0.8 5]; S22 = [0.9 1.3; 0.6 1.2];
    
N = 500;

X1 = getNormalDistributionSamples(N, M1, S1);
X2 = getBimodalDistributionSamples(N, M21, S21, M22, S22, P21);

% Plot data points
figure(1);
plot(X1(1,:),X1(2,:),'r*');
title('Data points');
hold on
plot(X2(1,:), X2(2,:), 'bo');
hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Show probability density functions as well as constant distribution
% function value on samples' diagram. 

x = -2:0.01:11;
y = -4:0.01:11;
f1 = zeros(size(x,2), size(y,2)); % Preallocate for speed.
f2 = zeros(size(x,2), size(y,2)); % Preallocate for speed.

% Find f1 and f2 values. 
for i=1:length(x)
    for j = 1:length(y)         
        X = [x(i); y(j)];      
        f1(i,j) = normalDistribution(M1, S1, X);
        f2(i,j) = bimodalDistribution(M21, S21, M22, S22, P21, X);    
    end
end

% 3D plot of f1 and f2. 
figure(2);
mesh(x,y,f1');
hold on;
mesh(x,y,f2');

% Find points which have the same probability denstity function value and 
% plot them.
figure(1);
hold on
fmax = max(max(f1));
contour(x,y,f1',[fmax*exp(-1/2),fmax*exp(-4/2),fmax*exp(-9/2)],'r');
fmax = max(max(f2));
contour(x,y,f2',[fmax*exp(-1/2),fmax*exp(-4/2),fmax*exp(-9/2)],'b');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot Bayesian and Neyman-Pearson's classifiers. 

% Compute discriminatory function. 
for i = 1:length(x)
    for j = 1:length(y)
        h(i,j) = -log(f1(i,j)) + log(f2(i,j)); 
    end
end

% Find eps0, mu dependency. 
[eps0, mu] = find_mu(M1, S1, M21, S21, M22, S22, P21);

% Plot Bayesian classfier (h(X) = 0), and Neyman Person's classifier for
% h(X) = mu(4) and h(X) = mu(5). 
figure(1);
hold on
contour(x,y,h',[0 0],'k','LineWidth',2); % Plot Bayesian classifier.
contour(x,y,h', [-log(mu(5)) -log(mu(5))], 'g', 'LineWidth', 2); % Plot Neyman-Pearson's classifier. 
contour(x,y,h', [-log(mu(6)) -log(mu(6))], 'm', 'LineWidth', 2); % Plot Neyman-Pearson's classifier. 
legend({'class1', 'class2', 'class1 const', 'class2 const', 'Bayesian', ... 
    ['$\mu = ', num2str(mu(5)), '$'], ...
    ['$\mu = ', num2str(mu(6)), '$']}, 'Interpreter','latex');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate errors for Bayesian and Neyman-Person's classifiers.

% Integral errors.
[eps1, eps2] = findIntegralErrors(x, y, f1, f2, h, 0); % Bayesian.
disp('Bayesian classifer errors');
display(eps1);
display(eps2);

[eps1, eps2] = findIntegralErrors(x, y, f1, f2, h, -log(mu(5))); % Neyman-Pearson.
disp(['Neyman-Person classifer errors mu = ', num2str(mu(5)), ...
    ' , eps0 = ', num2str(eps0(5))]);
display(eps1);
display(eps2);

[eps1, eps2] = findIntegralErrors(x, y, f1, f2, h, -log(mu(6))); % Neyman-Pearson.
disp(['Neyman-Person classifer errors mu = ', num2str(mu(6)), ...
    ' , eps0 = ', num2str(eps0(6))]);
display(eps1);
display(eps2);

% Errors based on discriminatory function.
M = 5000;
[Fh1, h1] = findDiscrCDF(M1, S1, M21, S21, M22, S22, P21, M, 1);
[Fh2, h2] = findDiscrCDF(M1, S1, M21, S21, M22, S22, P21, M, 2);

delta_h1 = h1(2)-h1(1); delta_h2 = h2(2)-h2(1);

eps1 = 1- Fh1(find( h1 > 0, 1 )-1); 
eps2 = Fh2(find( h2 > 0, 1 )-1); 
disp('Bayesian classifer errors based on f(h)');
display(eps1);
display(eps2);

eps1 = 1- Fh1(find( h1 > -log(mu(5)), 1 )-1); 
eps2 = Fh2(find( h2 > -log(mu(5)), 1 )-1);
disp(['Neyman-Person classifer errors mu = ', num2str(mu(5)), ...
    ' , eps0 = ', num2str(eps0(5)), ' based on f(h)']);
display(eps1);
display(eps2);

eps1 = 1- Fh1(find( h1 > -log(mu(6)), 1 )-1); 
eps2 = Fh2(find( h2 > -log(mu(6)), 1 )-1);
disp(['Neyman-Person classifer errors mu = ', num2str(mu(6)), ...
    ' , eps0 = ', num2str(eps0(6)), ' based on f(h)']);
display(eps1);
display(eps2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Helper functions.

function X = getNormalDistributionSamples(N, M, S)
%   Generates N samples that have normal distribution defined by mean vector
%   M and covariance matrix S. 

    % Perform color transform. 
    [F, L] = eig(S); 
    T = F * L^(1/2); 
    
    X = zeros(size(M,1), N); % Preallocate for speed. 
   
    % Generate data points.
    for i = 1:N
        X(:,i) = T*randn(2,1)+M;
    end
    
end

function X = getBimodalDistributionSamples(N, M1, S1, M2, S2, P1)
%   Generates N samples that have bimodal distribution defined by mean
%   vectors M1 and M2, covariance matrices S1 and S2, and probability P1. 

    % Perfom color transform. 
    [F1, L1] = eig(S1); [F2, L2] = eig(S2);
    T1 = F1 * L1^(1/2); T2 = F2 * L2^(1/2);
    
    X = zeros(size(M1,1), N); % Preallocate for speed. 
    
   % Generate data points.
    for i = 1:N
        if rand(1,1) < P1
            X(:,i) = T1*randn(2,1)+M1;
        else
            X(:,i) = T2*randn(2,1)+M2;
        end
    end     
    
end

function f = normalDistribution(M, S, X)
%   Calculates value of normal distribution defined by mean vector M,
%   covariance matrix S for value X. 

    f = 1/(2*pi*(det(S)^0.5)) * exp(-0.5*(X-M)' * inv(S) * (X-M));
    
end

function f = bimodalDistribution(M1, S1, M2, S2, P1, X)
%   Calculates value of function f defined by bimodal distribution with
%   mean vectors M1 and M2, covariance matrices S1 and S2, and probability
%   P1.
    
    f = P1*normalDistribution(M1, S1, X) + ...
        (1-P1)*normalDistribution(M2, S2, X);
end

function h = H(X, M1, S1, M21, S21, M22, S22, P21)
%   Finds discriminatory function for given vector X assuming the above
%   defined probability density functions. 
    
    f1 = normalDistribution(M1, S1, X);
    f2 = P21*normalDistribution(M21, S21, X) + ...
         (1-P21)*normalDistribution(M22, S22, X);
   
    h = -log(f1) + log(f2); 
    
end

function [eps0, mu] = find_mu(M1, S1, M21, S21, M22, S22, P21)
%   Finds eps0 dependency on mu. Generates 5000 points from second class
%   and finds probability density function of variable h when samples
%   belong to second class using histogram. Plots the dependency in log
%   scale. 

    % Generate 5000 second class' data points. 
    N = 5000;
    [Fh, h] = findDiscrCDF(M1, S1, M21, S21, M22, S22, P21, N, 2);

    % Find cumulative sum F(h) = eps0 and h = -log(mu). 
    eps0 = zeros(size(Fh)); % Preallocate for speed. 
    mu = zeros(size(Fh)); 
    for i=1:size(Fh,2)
        eps0(i) = Fh(i);
        mu(i) = exp(-h(i));
    end
    
    % Plot dependency. 
    figure, semilogx(mu, eps0), 
    title('$$ \varepsilon_0(\log(\mu)) $$','interpreter','latex')

end

function [eps1, eps2] = findIntegralErrors(x, y, f1, f2, h, t)
%   Finds first and second type errors by integrating f1 and f2 on areas
%   that correspond to opposite class where t is the descision threshold. 

    class1area = h < t;
    class2area = h > t;
    
    eps1 = trapz(y,trapz(x,f1.*double(class2area),1),2);
    eps2 = trapz(y,trapz(x,f2.*double(class1area),1),2);
end

function [Fh, h] = findDiscrCDF(M1, S1, M21, S21, M22, S22, P21, N, class)
%   Generates N samples in order to find probability density function of 
%   discriminatory function. 

    % Generate N data points. 
    if (class == 1)
        X2 = getNormalDistributionSamples(N, M1, S1);
    else
        X2 = getBimodalDistributionSamples(N, M21, S21, M22, S22, P21);
    end

    % Find h(X).
    h = zeros(N,1); % Preallocate for speed. 
    for i=1:N
        X = [X2(1, i); X2(2, i)];
        h(i) = H(X, M1, S1, M21, S21, M22, S22, P21);
    end

    % Find and plot histogram. 
    figure,
    fh_hist = histogram(h, 'Normalization', 'cdf'); 
    title(['$ F(h/ \omega_', num2str(class), ')$'],'interpreter','latex');
    
    % Find upper and lower values of variable h. 
    Fh = fh_hist.Values;
    h = fh_hist.BinLimits(1)+fh_hist.BinWidth:fh_hist.BinWidth:fh_hist.BinLimits(2);
end