% Pattern recognition - Homework 2 
% Wald's sequential test

% Perofrms Wald's test in order to calculate the average number of samples
% needed to make a descision based on defined errors. Compares the result
% with theoretical one. 

% January, 2018
% Savic Jovana 2013/243

close all
clear all

% Probability density functions are defined as:
% f1(X) = N(M1, S1)
% f2(X) = P21 x N(M21, S21) + P22 x N(M22, S22)
% Check helper functions for details. 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Wald's sequential test. 
% Perform Wald's test for each combination of errors 20 times for each
% class and take note of average number of samples needed as well as errors
% made.

% Range of type 1 and type 2 errors. 
eps = logspace(-10,0, 100); 
N = size(eps,2);
M = 20; % Number of tests per error pair. 

% Keep average number of samples and average number of errors.
first_class_avg = zeros(N);
first_class_error = zeros(N);
second_class_avg = zeros(N);
second_class_error = zeros(N);

for i=1:N
    for j=1:N
        eps1 = eps(i); eps2 = eps(j);
        
        m1 = 0; error1 = 0;
        m2 = 0; error2 = 0;
        
        for k=1:M
            % Test first class.
            [m, error] = performWald(eps1, eps2, 1);
            m1 = m1 + m; 
            error1 = error1 + error;
            %Test second class.
            [m, error] = performWald(eps1, eps2, 2);
            m2 = m2 + m; 
            error2 = error2 + error;
        end
        
        % Take average.
        first_class_avg(i,j) = m1/M;
        first_class_error(i,j) = error1/M;
        
        second_class_avg(i,j) = m2/M;
        second_class_error(i,j) = error2/M;
        
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Plot dependency.
figure (1),
mesh(eps, eps, first_class_avg);
title('Average number of measurements needed to classify first class');

figure(2),
mesh(eps, eps, second_class_avg);
title('Average number of measurements needed to classify second class');

% Plot dependencies in log scale.
first_class = first_class_avg(50, :);
figure (3), semilogx(eps, first_class);
set(gca,'XScale','log','YScale','log');
title('First class - eps1 const.');

second_class = second_class_avg(50,:);
figure(4), semilogx(eps, second_class);
set(gca,'XScale','log','YScale','log');
title('Second class - eps1 const');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Find E(h(X)/omega1) and E(h(X)/omega2).

N = 500;

% Generate first class' data points.
for i = 1:N
    X1(:, i) = getFirstClassData();
end

% Generate second class' data points.
for i = 1:N
    X2(:, i) = getSecondClassData();
end

h1 = zeros([N,1]); % h(X)/omega1
h2 = zeros([N,1]); % h(X)/omega2

% Find h(X)/omega1.
for i=1:N
    X = [X1(1, i); X1(2, i)];   
    h1(i) = H(X); 
end

% Find h(X)/omega2.
for i=1:N
    X = [X2(1, i); X2(2, i)];
    h2(i) = H(X);
end

% Expected values.
eta1 = mean(h1(:));
eta2 = mean(h2(:));
m1 = zeros([size(eps,2), 1]);
m2 = zeros([size(eps,2),1]);

% Find expected number of measurements needed.
for i=1:size(eps,2)
    % Eps1 const for first class.
    eps1 = eps(50); eps2 = eps(i);
    [a b] = WaldsTestLimits(eps1, eps2); 
    m1(i) = (a*(1-eps1)+b*eps1)/eta1;
    
    %Eps2 const for second class. 
    eps1 = eps(i); eps2 = eps(50);
    [a b] = WaldsTestLimits(eps1, eps2); 
    m2(i) = (b*(1-eps2)+a*eps2)/eta2;
end

% Plot dependencies in log scale.
figure (5), semilogx(eps, m1);
title('First class - eps1 const');
hold on
semilogx(eps, first_class_avg(50,:));
legend('Theoretical', 'Actual');
hold off

figure(6), semilogx(eps, m2);
title('Second class - eps2 const');
hold on
semilogx(eps, second_class_avg(:,50));
legend('Theoretical', 'Actual');
hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Helper functions.

function h = H(X)
%   Finds discriminatory function for given vector X assuming the above
%   defined probability density functions. 
    
    M1 = [7 4.8]';
    M21 = [3 4.2]'; M22 = [4.4 1.3]';

    S1 = [0.8 0.7; 0.2 0.4]; 
    S21 = [1.3 1.5; 0.8 5]; S22 = [0.9 1.3; 0.6 1.2];
    
    P21 = 0.35;
    
    f1 = 1/(2*pi*(det(S1)^0.5)) * exp(-0.5*(X-M1)' * inv(S1) * (X-M1));
            
    f21 = 1/(2*pi*(det(S21)^0.5)) * exp(-0.5*(X-M21)' * inv(S21) * (X-M21));
    f22 = 1/(2*pi*(det(S22)^0.5)) * exp(-0.5*(X-M22)' * inv(S22) * (X-M22));
    f2= P21*f21 + (1-P21)*f22;
            
    h = -log(f1) + log(f2);   
end

function [a b] = WaldsTestLimits(eps1, eps2)
%   Finds a,b limits used in Wald's test needed to maintain given error rate.
    A = (1-eps1)/eps2; B = eps1/(1-eps2);
    a = -log(A); b = -log(B);
end

function X = getFirstClassData()
% Generates first class' data point based on the above defined probability
% density function.

    M1 = [7 4.8]'; S1 = [0.8 0.7; 0.2 0.4]; 
    
    [F1, L1] = eig(S1); 
    T1 = F1 * L1^(1/2); 
    
    X = T1*randn(2,1)+M1;
end

function X = getSecondClassData()
% Generates second class' data point based on the above defined probability
% density function.

    M21 = [3 4.2]'; M22 = [4.4 1.3]';
    S21 = [1.3 1.5; 0.8 5]; S22 = [0.9 1.3; 0.6 1.2];
    P21 = 0.35;
    
    [F21, L21] = eig(S21); [F22, L22] = eig(S22);
    T21 = F21 * L21^(1/2); T22 = F22 * L22^(1/2);
    
    if rand(1,1) < P21
        X = T21*randn(2,1)+M21;
    else
        X = T22*randn(2,1)+M22;
    end
end

function [m, error] = performWald(eps1, eps2, class)
% Performs Wald's test for given errors eps1 and eps2. Generates points
% that belong to class "class". Returns the number of data points it needed
% before reaching conclusion. Sets error to 1 if there was an error.

    [a, b] = WaldsTestLimits(eps1, eps2); % get limits.
    sm = 0; m = 0; error = 0; % prepare for test. 
    
    while (1)       
        % Generate sample. 
        if class == 1
            X = getFirstClassData();
        else 
            X = getSecondClassData();
        end
        m = m + 1;
        sm = sm + H(X);
        
        % Have we reached one of the limits?
        if sm >= b 
            if class == 1 % we're putting it into second class when it's in fact first.
                error = 1;
            end
            break; % Test is over.
        end
        if sm <= a
            if class == 2 % we're putting it into first class when it's in fact second.
                error = 1;
            end
            break; % Test over.
        end      
    end
    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     
        