% Pattern recognition - Homework 4
% C-mean vs. Square decomposition method

% Generates 500 points from each of the 3 classes. Classes are linearly
% separable. 
% Both c-mean and square decomposition methods are used to cluster given 
% samples, assuming that there are 3 classes. Results are plotted. 
% Both algorithms are performed 50 times to compare the averaege number of
% iterations as well as sensistivity to initial conditions. 

% January, 2019
% Savic Jovana 2013/243

close all
clear all
warning off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate 500 points from each of three 2-dimensional classes. 

% Probability density functions are defined as:
% f1(X) = N(M1, S1)
% f2(X) = N(M2, S2)
% f3(X) = N(M3, S3);

N = 500;

M1 = [-1 -2]';
S1 = [1 0.5; 0.8 1.1];

M2 = [-2 7]';
S2 = [0.3 0.6; 0.9 0.9];

M3 = [6 1]';
S3 = [0.5 1.2; 1 0.3];

% Apply color transform.
[F1, L1] = eig(S1); [F2, L2] = eig(S2); [F3, L3] = eig(S3); 
T1 = F1 * L1^(1/2); T2 = F2 * L2^(1/2); T3 = F3 * L3^(1/2); 

% Generate data points.
for i = 1:N
    X1(:,i) = T1*randn(2,1)+M1;
    X2(:,i) = T1*randn(2,1)+M2;
    X3(:,i) = T1*randn(2,1)+M3;
end

% Plot data points
X = [X1, X2, X3];
omega = [ones(1,N), 2*ones(1,N), 3*ones(1,N)];
p = plotClasses(X, omega, 'Initial data set');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Perform clustering with both algorithms and plot results.

% Perform square decomposition.
L = 3; % Three classes. 
[omega, l] = squareDecomposition(X, L);
p = plotClasses(X, omega, 'After square decomposition');

% Perform c mean clustering.
[omega, l] = cmean(X, L);
p = plotClasses(X, omega, 'After c-mean');

% Perform both algorithms 50 times to see how number of iterations changes.
I = 50; % Number of iterations.
l_cmean = zeros(1,I); % Preallocate for speed.
l_sqdec = zeros(1,I);

for i=1:I
    [omega, l_cmean(i)] = cmean(X, L);
    [omega, l_sqdec(i)] = squareDecomposition(X, L);
end

figure, histogram(l_cmean), title('C-mean iterations');
figure, histogram(l_sqdec), title('Square decomposition iterations');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Helper functions.

function [omega, l] = cmean(X, L)
%   Performs C-mean clustering for given samples X, assuming that L is the
%   number of classes. Returns probability of each class, minimum distance
%   between any of cluster centers, number of iterations and the
%   classification vector. 

    % Define initial classification.
    omega = randi([1 L],1, size(X,2));
    M = zeros([size(X,1), L]);
    l = 0;
    
    % Start iterations.
    while(1)
        %Find expected values.
        for j = 1:L
            M(:,j) = mean((X(:, omega == j))')';
        end
        % Reclassify samples.
        reclassified = 0;      
        d = zeros([L,1]); % Preallocate for speed.
        for i=1:size(X,2)
            % Check distance for each class.
            for j=1:L
                d(j) = norm(X(:,i)-M(:,j));
            end
            [k ind] = min(d); % Find closest M vector. 
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
    end
end

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
    end
end