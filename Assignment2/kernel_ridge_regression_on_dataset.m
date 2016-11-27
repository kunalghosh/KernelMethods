function [ mset_train,mse_test ] = kernel_ridge_regression_on_dataset( X_train, Y_train, X_test, Y_test, sigma, lambda )
%kernel_ridge_regression_on_dataset Computes the kernel ridge regression
% Computes the kernel ridge regression prediction on a test set and returns
% the mean squared error.
% NOTE: This is not very optimal since we are calculating the kernel everytime
% However, the resulting code is cleaner and the gaussian_kernel function can
% be memoized so as to not compute the kernel if the input is same.

   K_train_train = gaussian_kernel(X_train,X_train,sigma);
   K_train_test = gaussian_kernel(X_train,X_test,sigma);
   
   alpha = kernel_ridge_regression(K_train_train, Y_train, lambda);
   mse_test = get_prediction_mse(alpha, K_train_test, Y_test);
   mset_train = get_prediction_mse(alpha, K_train_train, Y_train);

end

