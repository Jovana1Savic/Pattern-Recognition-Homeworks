% Pattern recognition - Homework 4
% C - mean clustering 

% Generates 500 points from each of the 4 classes. All classes have normal
% distribution and are linearly separable. 
% Performs c-mean clustering algorithm with multisample reclassification
% (therefore gets stuck in local min, run again if the results seem bad).
% Assumes that the number of classes is 4. 
% Performs c-mean clustering algorithm a few more times to find the average
% number of iterations. 
% Perfoms the same algorithm with unknown number of classes on the same
% data set and tries to estimate around how many classes there are. 

% January, 2019
% Savic Jovana 2013/243

close all
clear all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate 500 points from each of two 2-dimensional classes. 

% Probability density functions are defined as:
% f1(X) = N(M1, S1)
% f2(X) = N(M2, S2)
% f3(X) = N(M3, S3);
% f4(X) = N(M4, S4);

N = 500;

M1 = [-1 -2]';
S1 = [1 0.5; 0.8 1.1];

M2 = [-2 7]';
S2 = [0.3 0.6; 0.9 0.9];

M3 = [6 1]';
S3 = [0.5 1.2; 1 0.3];

M4 = [6.5 8]';
S4 = [1 1; 0.4 0.4];

% Apply color transform.
[F1, L1] = eig(S1); [F2, L2] = eig(S2);
[F3, L3] = eig(S3); [F4, L4] = eig(S4); 

T1 = F1 * L1^(1/2); T2 = F2 * L2^(1/2); 
T3 = F3 * L3^(1/2); T4 = F4 * L4^(1/2); 

% Generate data points.
for i = 1:N
    X1(:,i) = T1*randn(2,1)+M1;
    X2(:,i) = T1*randn(2,1)+M2;
    X3(:,i) = T1*randn(2,1)+M3;
    X4(:,i) = T1*randn(2,1)+M4;
end

% Plot data points
X = [X1, X2, X3, X4];
omega = [ones(1,N), 2*ones(1,N), 3*ones(1,N), 4*ones(1,N)];
p = plotClasses(X, omega, 'Initial data set');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Perform c-mean clustering for 4 classes and plot results. Repeat
% clustering 20 times to find average number of iterations. 

% Perform c-mean clustering for 4 classes. 
L =4;
[P, M_min, l, omega, M] = cmean(X, L);
p = plotClasses(X, omega, 'C-mean clustering, L=4');

% Perform clustering 20 times to find mean number of iterations.
l_avg = 0;
for i = 1:20
    [P, M_min, l, omega, M] = cmean(X, L);
    l_avg = l_avg + l;
end

l_avg = l_avg/20

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Perform c-mean clustering assuming that the number of classes is unknown.
% Start with 2 classes and go up until the algorithm gets stuck classifing
% smaller number of classes. 
% Algorithm is used to estimate possible number of classes.

% See what happens as the number of classes changes.
for L=2:10 % Don't go over 10 classes at worst.
    
    % Try once. 
    [P, M_min1, l, omega, M] = cmean(X, L);
    p = plotClasses(X, omega, strcat('C-mean clustering L=', int2str(L)));
    % How many have we actually found?
    numOfClasses1 = size(unique(omega),2);
    
    % Try again. 
    [P, M_min2, l, omega, M] = cmean(X, L);
    p = plotClasses(X, omega, strcat('C-mean clustering L=', int2str(L)));
    % How many this time? 
    numOfClasses2 = size(unique(omega),2);
    
    % If we get stuck twice with smaller number of classes than assumed, 
    % assume it's because the number of classes was wrong. 
    if (numOfClasses1 < L) && (numOfClasses2 < L)
        display(numOfClasses1);
        display(numOfClasses2);
        break;  
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Helper functions.

function [P, M_min, l, omega, M] = cmean(X, L)
%   Performs C-mean clustering for given samples X, assuming that L is the
%   number of classes. Returns probability of each class, minimum distance
%   between any of cluster centers, number of iterations and the
%   classification vector. 

    % Define initial classification.
    omega = randi([1 L],1, size(X,2));
    
%     % Plot initial conditions. 
%     p = plotClasses(X, omega, 'Initial clustering');
    
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
                d(j) = (X(:,i)-M(:,j))'*(X(:,i)-M(:,j));
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
        
%         p = plotClasses(X, omega, strcat('After l=', int2str(l)));
    end
    
    % Find probability of each class.
    P = zeros([1, L]); % Preallocate for speed.
    for i=1:L
        P(i) = size(omega(omega == i),2) / size(omega,2);
    end
        
    % Find minimal distance between M1...ML vectors.
    M_min = norm(M(:,1)-M(:,2));
    for i=1:L
        for j=i+1:L
            d = norm(M(:,i)-M(:,j));
            if d < M_min
                M_min = d;
            end
        end
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

