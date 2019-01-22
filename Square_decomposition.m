% Pattern recognition - Homework 4
% Square decomposition method

% Generates 500 points from each of the 2 classes. Classes aren't linearly
% separable. Data set is plotted. 
% Square decomposition method is used to cluster given samples, assuming
% that there are 2 classes. Results are plotted. 
% Algorithm is performed 20 times to find average number of iterations. 

% January, 2019
% Savic Jovana 2013/243

warning off
close all
clear all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate two classes and plot them.

N = 500;

% Generate first class' data points. A circle.
x1 = 4; y1 = 4;
a1 = 2.4; 
for i = 1:N
    theta = rand(1,1)*2*pi;
    R = rand(1,1);
    X1(1,i) = x1 + R*a1*cos(theta);
    X1(2,i) = y1 + R*a1*sin(theta);
end

% Generate second class' data points. A circle with a hole.
x2 = 4; y1 = 4;
a2 = 6; 
r = 3.6;
for i = 1:N
    theta = rand(1,1)*2*pi;
    R = rand(1,1);
    X2(1,i) = x1 + (r+(a2-r)*R)*cos(theta);
    X2(2,i) = y1 + (r+(a2-r)*R)*abs(sin(theta));
end

% Plot data points
X = [X1, X2];
omega = [ones(1,N), 2*ones(1,N)];
p = plotClasses(X, omega, 'Initial data set');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Assuming that there are two classes perform square decomposition and plot
% results.
% Perform the same algorithm 20 times in order to find average number of
% iterations.

% Perform square decomposition.
L = 2; % Two classes. 
[omega, l] = squareDecomposition(X, L);
p = plotClasses(X, omega, 'After square decomposition');

% Perform clustering 20 times to find average number of iterations.
I = 20; % Number of iterations.
l = zeros(1,I);
for i = 1:I
    [omega, l(i)] = squareDecomposition(X, L);
end

l_avg = mean(l);
l_var = var(l);
figure, histogram(l), title('Iterations histogram');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Perform square decomposition clustering assuming that the number of
% classes is unknown. 
% Start with 2 classes and go up to 10 classes. 
% Plot the maximum difference between each class appearing probability in 
% order to try and conclude the original number of classes. 

% See what happens as the number of classes changes.
for L=2:10 % Don't go over 10 classes at worst.
    
    % Try once. 
    [omega, l] = squareDecomposition(X, L);
    P(L) = diffProbabilities(X,L,omega);
    
    % Try again. 
    [omega, l] = squareDecomposition(X, L);
    P(L) = (P(L) + diffProbabilities(X,L,omega))/2;
end

% Plot results. 
figure, plot(P), title('max(P(i)-P(j)) i=1,..,L, j=1,...,L, i ~= j;');
l = 1:L;
figure, plot(P.*l), title('Weighted maximum difference between class probabilities');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Helper functions.

function p = plotClasses(X, omega, t)
%   Plots results of clustering in new figure. X is the set of vectors and
%   omega is their classification. Sets title to t. 
    
    L = max(omega); % Number of classes. 
    figure,
    s = 'class1';
    for i=1:L
        class = X(:,omega==i);
        plot(class(1,:), class(2,:), '*');
        hold on
        if i > 1
            s = [s; strcat('class', int2str(i))];
        end
    end
    
    % Set details.
    legend(s);
    title(t);
    hold off
    
    p = 1;
end

function [omega, l] = squareDecomposition(X, L)
%   Performs square decomposition (square clasterization) on given samples
%   X, assuming that there are L classes. Returns classification along with
%   number of iterations needed to get to it. 

    % Define initial classification.
    omega = randi([1 L],1, size(X,2));
   
%     % Plot initial conditions. 
%     p = plotClasses(X, omega, 'Initial clustering');

    M = zeros([size(X,1), L]);
    S = cell([1, L]);
    P = zeros([1, L]);
    l = 0;
    
    while(1)
        % Find mean vectors, covariance matrices and probabilities for each
        % class for current clasterization.
        for j = 1:L
            M(:,j) = mean((X(:, omega == j))')';
            S{j} =  cov((X(:, omega == j))')';
            P(j) = size(omega(omega == j),2) / size(omega,2);
        end 
        % Reclassify samples.
        reclassified = 0;      
        d = zeros([L,1]); % Preallocate for speed.
        for i=1:size(X,2)
            % Check distance for each class.
            for j=1:L
                d(j) = (X(:,i)-M(:,j))'*inv(S{j})*(X(:,i)-M(:,j)) - ...
                    log(det(inv(S{j}))) + log(P(j));
            end
            [k ind] = min(d); % Find smallest distance. 
            % Reclassify if needed. 
            if (omega(i) ~= ind)
                reclassified = 1;
                omega(i) = ind;
            end
        end
        if reclassified == 0
            break;
        end
        l = l+1;
%       p = plotClasses(X, omega, strcat('After l=', int2str(l)));
    end
end

function J = minimizationFunction(X, L, omega)
%   Finds minimization function value for given clusterization with L
%   classes.

    J = 0;
    M = zeros([size(X,1), L]); % Preallocate for speed.
    S = cell([1, L]);
    P = zeros([1, L]);
    
    % Find mean vectors, covariance matrices and probabilities for each
    % class for current clasterization.
    for j = 1:L
        M(:,j) = mean((X(:, omega == j))')';
        S{j} =  cov((X(:, omega == j))')';
        P(j) = size(omega(omega == j),2) / size(omega,2);
    end 
    for i=1:size(X,2)
        j = omega(i); % X(i) class. 
        J = J + (X(:,i)-M(:,j))'*inv(S{j})*(X(:,i)-M(:,j)) - ...
                    log(det(inv(S{j}))) + log(P(j));
    end
end

function max_diff = diffProbabilities(X,L,omega)
%   Finds maximum difference in probabilities of each class for given vector
%   of samples X, number of classes L and classification omega.

    P = zeros([1, L]);  % Preallocate for speed. 
    
    % Find each class probability. 
    for j = 1:L
        P(j) = size(omega(omega == j),2) / size(omega,2);
    end 
    
    % Find maximum difference of probabilities. 
    max_diff = 0;
    for i=1:L
        for j=i+1:L
            diff = abs(P(i)-P(j));
            if max_diff < diff
                max_diff = diff;
            end
        end
    end
    
end