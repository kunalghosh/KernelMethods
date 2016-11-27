% Application of Parzen window classifier on a synthetic dataset

% Load data
load data_all.mat

% 1. Computation and plot of the accuracy

val_sigma = logspace(-2, 1, 9); % returns 9 values between 10^-2 and 10^1 for the parameter sigma

acc_test = zeros(length(val_sigma),1);
acc_train = zeros(length(val_sigma),1);
for i = 1:length(val_sigma)
    
    % Build the Gram matrices
    Kx_train = gaussian_kernel(X_train, X_train, val_sigma(i));
    %Kx_train
    Kx_train_test = gaussian_kernel(X_train, X_test, val_sigma(i));
    % size(Kx_train_test)
    % Prediction on the test examples and computation of the test accuracy
    %size(y_test)
    y_pred_test = parzen_classify(Kx_train, Kx_train_test, y_train);
    acc_test(i) = sum(y_pred_test == y_test')/length(y_test)*100;
    
    % Prediction on the training examples and computation of the training accuracy
    y_pred_train = parzen_classify(Kx_train, Kx_train, y_train);
    acc_train(i) = sum(y_pred_train == y_train')/length(y_train)*100;
end

figure
grid on
semilogx(val_sigma, acc_test) % semilogx plot data as logarithmic scales for the x-axis
hold on
semilogx(val_sigma, acc_train)
hold off
legend('Test accuracy','Training accuracy')
grid on
ylabel('Accuracy')
xlabel('Sigma')


% 2. Plot of the decision boundary

% Define a grid of points
d = 0.1; % Step size of the grid
[x1Grid,x2Grid] = meshgrid(-3.5:d:4.5,-3.5:d:4.5);
xGrid = [x1Grid(:),x2Grid(:)]; % The grid

figure
grid on
for i = 1:length(val_sigma)
    
    % Build the Gram matrices
    Kx_train = gaussian_kernel(X_train, X_train, val_sigma(i));
    Kx_train_grid = gaussian_kernel(X_train, xGrid, val_sigma(i));
    
    % label prediction for all the points in the grid
    pred = parzen_classify(Kx_train, Kx_train_grid, y_train);
    
    subplot(3,3,i)
    % scatter plot of the data
    gscatter(X(:,1),X(:,2),y,'cr','xo');
    hold on
    % plot the contour separating the points that are predicted to be
    % positive and the ones that are predicted to be negative
    contour(x1Grid,x2Grid,reshape(pred,size(x1Grid)),[0,0],'k','LineWidth',1);
    hold off
    xlim([-3.5 4.5])
    ylim([-3.5 4.5])
    title(['sigma = ', num2str(val_sigma(i))])
end







