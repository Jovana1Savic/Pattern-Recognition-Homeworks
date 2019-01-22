% Pattern recognition - Homework 1
% Digit classificaton

% This program reads 360 images from database baza2018 that show handwritten
% zeros, sixes and eights (120 for each class). Firstly, the images are
% reduced to 5x5 images containg digits without white margins. Then a
% dimension reduction is perfomed in order to find features.
% Finally images are classified based on Bayesian minimum error classifier
% with the assumption of normal distribution. 

% January, 2019
% Jovana Savic

close all
clear all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% N is the number of images in each class - zeros, sixes, and eights. 
% No is the number of images in each class that make learning set. 
N = 120;
No = 100; 
folder = 'baza2018\';

% Read images and their features.
for i = 1:N
    % Read zeros. 
    name = ['baza0' num2str(i,'%03d')]; 
    x = imread(strcat(folder,name), 'bmp');
    X0(:,i) = features(x, name); 
    
    % Read sixes. 
    name = ['baza6' num2str(i,'%03d')]; 
    x = imread(strcat(folder,name), 'bmp');
    X6(:,i) = features(x, name); 
    
    % Read eights. 
    name = ['baza8' num2str(i,'%03d')];
    x = imread(strcat(folder,name), 'bmp');
    X8(:,i) = features(x, name);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Perfom Linear Discriminant Analysis in order to reduce dimensions.

% O_i - training set for digit i; T_i - test set for digit i.
O0 = X0(:, 1:No); T0 = X0(:,No+1:N);
O6 = X6(:, 1:No); T6 = X6(:,No+1:N);
O8 = X8(:, 1:No); T8 = X8(:,No+1:N);

% Mean value and variance matrix of each class.
M1 = mean(O0')'; S1 = cov(O0')';
M2 = mean(O6')'; S2 = cov(O6')';
M3 = mean(O8')'; S3 = cov(O8')';

% Prepare matrices needed for LDA. 
Sw = 1/3*(S1 + S2 + S3); % Within class scatter matrix.
M0 = 1/3*(M1 + M2 + M3); % Combined expected value vector.
Sb = 1/3*((M1-M0)*(M1-M0)'+(M2-M0)*(M2-M0)'+(M3-M0)*(M3-M0)'); % Between class scatter matrix.
Sm = Sw + Sb; % Mixed matrix.

% Take eigenvalues. 
S1 = Sb; S2 = Sw;
S = S2^(-1)*S1;
[F L] = eig(S);

% Take 2 eigenvectors for which lambda is maximum.
[l, ind] = sort(diag(L), 'descend');
A = [F(:, ind(1)), F(:, ind(2))];

% Plot the information perserved by taking first n dimensions.
l_sum = sum(l);
I = zeros(size(l));
m = zeros(size(l));
for i=1:length(I)
    I(i) = sum(l(1:i))/l_sum;
    m(i) = i;
end

figure(3),
plot(m, I*100), title('I(m) %'),
xticks(m');

% Transform measurements to reduce dimension.
for i=1:N
    Y0(:,i) = A'*X0(:,i);
    Y6(:,i) = A'*X6(:,i);
    Y8(:,i) = A'*X8(:,i);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% After reducing to 2 dimensions visualize results and classify samples. 

% Define tranining and test sets.
O0 = Y0(:, 1:No); T0 = Y0(:,No+1:N);
O6 = Y6(:, 1:No); T6 = Y6(:,No+1:N);
O8 = Y8(:, 1:No); T8 = Y8(:,No+1:N);

% Plot training set. 
figure(1),
plot(O0(1,:), O0(2,:), 'sb', O6(1,:), O6(2,:), 'hg', O8(1,:), O8(2,:), 'ro');
title('Training set');

% Plot test set. 
figure(2),
plot(T0(1,:), T0(2,:), 'sb', T6(1,:), T6(2,:), 'hg', T8(1,:), T8(2,:), 'ro');
title('Test set');

% Mean value and variance matrix of each class.
M1 = mean(O0')'; S1 = cov(O0')';
M2 = mean(O6')'; S2 = cov(O6')';
M3 = mean(O8')'; S3 = cov(O8')';

% Use QQ plot to test how well the test data fits the assumed distribution
% (normal distribution). 
figure, qqplot(O0'), title('First class - zeros qq plot');
figure, qqplot(O6'), title('Second class - sixes qq plot');
figure, qqplot(O8'), title('Third class - eights qq plot');

% Classify digits using minimum distance from mean vectors.
% MK = classify(M1, M2, M3, T0, T6, T8)

% Plot classifier lines.
% syms xp yp
% eq1 = (xp-M1(1))^2+(yp-M1(2))^2 == (xp-M2(1))^2+(yp-M2(2))^2;
% eq2 = (xp-M1(1))^2+(yp-M1(2))^2 == (xp-M3(1))^2+(yp-M3(2))^2;
% eq3 = (xp-M2(1))^2+(yp-M2(2))^2 == (xp-M3(1))^2+(yp-M3(2))^2;
% 
% xp = 100:0.1:250;
% yp = -180:0.1:-60;
% for i = 1:length(xp)
%     for j=1:length(yp)
%         x = xp(i); y = yp(j);
%         Z1(i,j) = (norm([x;y]-M1) < norm([x;y]-M2)) & (norm([x;y]-M1) < norm([x;y]-M3));
%         Z3(i,j) = (norm([x;y]-M3) < norm([x;y]-M1)) & (norm([x;y]-M3) < norm([x;y]-M2));
%     end
% end
% 
% figure(1)
% hold on
% contour(xp, yp, Z1', [1 1]);
% contour(xp, yp, Z3', [1 1]);
% hold off
% 
% figure(2)
% hold on
% contour(xp, yp, Z1', [1 1]);
% contour(xp, yp, Z3', [1 1]);
% hold off

MK = classifyGauss(M1, M2, M3, S1, S2, S3, T0, T6, T8)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Helper functions.

function P = features(X, name)
%   Takes image X and removes white margines. Resizes picture to 20x20 and
%   takes pixel values as features. Writes processed image to folder. 
% Remove scanning lines and binarize image.

    [nr,nc] = size(X);
    X = X(round(0.05*nr):round(0.95*nr), round(0.05*nc):round(0.95*nc));
    [nr, nc] = size(X);
    x = imbinarize(X);

    % Remove white margins.
    poc = 1;
    while (poc<nr) && ((sum(x(poc,1:nc))/nc>0.98)... % pretezno belo 
            || (sum(x(poc,1:nc))/nc <= 0.98 && sum(x(poc+1,1:nc))/nc>0.98)) 

            poc = poc + 1;    
    end

    kraj = nr;
    while (kraj>poc) && ((sum(x(kraj,1:nc))/nc>0.98)... % pretezno belo 
            || (sum(x(kraj,1:nc))/nc <= 0.98 && sum(x(kraj-1,1:nc))/nc>0.98)) 

            kraj = kraj - 1;    
    end

    levo = 1;
    while (levo < nc) && ((sum(x(1:nr,levo))/nr >0.98)... % pretezno belo 
            || (sum(x(1:nr,levo))/nr <= 0.98 && sum(x(1:nc,levo+1))/nc>0.98)) 

            levo = levo + 1;   
    end

    desno = nc;
    while (desno > levo) && ((sum(x(1:nr,desno))/nr >0.98)... % pretezno belo 
            || (sum(x(1:nr,desno))/nr <= 0.98 && sum(x(1:nc,desno-1))/nc>0.98))

            desno = desno - 1;   
    end

    X = X(poc:kraj, levo:desno);
    % Write result to file.
%     imwrite(X, strcat('Obradjene slike\', name, '.bmp'));
    
    M = 5;
    X = imresize(X, [M M]);
    
    P = double(X(:)); % Take all pixels as starting features. 
end

function MK = classify(M1, M2, M3, T1, T2, T3)
%   Classifies samples based on their distance from expected value vectors.

MK = zeros(3,3); % confusion matrix
    for k1 = 1:3
        if k1 == 1
            T = T1;
        elseif k1 == 2
            T = T2;
        else
            T = T3;
        end
        for i = 1: length(T(1,:))
            % Find the smallest Eucledian distance from mean vectors. 
            f1 = norm(T(:,i) - M1);
            f2 = norm(T(:,i) - M2);
            f3 = norm(T(:,i) - M3);
            f = [f1, f2, f3];
            % Classify vector. 
            [~, class] = min(f);
            MK(k1, class) = MK(k1, class) + 1;
        end
    end
end

function MK = classifyGauss(M1, M2, M3, S1, S2, S3, T1, T2, T3)
%   Classifies samples assuming normal distribution of samples. Calculates
%   value of probability density function at given measurement and assings
%   it to class for which this value is the greatest. 

MK = zeros(3,3); % confusion matrix
    for k1 = 1:3
        if k1 == 1
            T = T1;
        elseif k1 == 2
            T = T2;
        else
            T = T3;
        end
        for i = 1: length(T(1,:))
            % Find maximum f. 
            f1 = exp(-0.5*(T(:,i)-M1)'*inv(S1)*(T(:,i)-M1))/(sqrt((2*pi)^3)*sqrt(det(S1)));
            f2 = exp(-0.5*(T(:,i)-M2)'*inv(S2)*(T(:,i)-M2))/(sqrt((2*pi)^3)*sqrt(det(S2)));
            f3 = exp(-0.5*(T(:,i)-M3)'*inv(S3)*(T(:,i)-M3))/(sqrt((2*pi)^3)*sqrt(det(S3)));
            f = [f1, f2, f3];
            % Classify vector. 
            [~, class] = max(f);
            MK(k1, class) = MK(k1, class) + 1;
            if (k1 ~= class) 
                disp(['Pogresno klasifikovana ', int2str(i), '-ta ', ...
                    int2str(k1), ' klasa kao klasa ', int2str(class)]);
            end
        end
    end
end