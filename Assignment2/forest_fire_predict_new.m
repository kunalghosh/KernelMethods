% ========================
% Predicting forest fires. 
% ========================

load data_all.mat

lambdas = 10 .^ (-3:7);
sigmas = 10 .^ (-1:4);

% Matrices to save the value of MSE for train and test data.
sigma_len = size(sigmas,2);
lambda_len = size(lambdas,2);
result_shape = [sigma_len, lambda_len];
mse_test = zeros(result_shape);
mse_train = zeros(result_shape);
mse_cv = zeros(result_shape);

lowest_cv_error = Inf;
best_sigma  = 0;
best_lambda = 0;

K = 5; % K fold cross validation
for sigma_idx = 1:sigma_len
   for lambda_idx = 1:lambda_len
        sigma = sigmas(sigma_idx);
        lambda = lambdas(lambda_idx);
        
        % we keep saving the errors so that later we don't need to recompute them 
        [mse_train(sigma_idx, lambda_idx), mse_test(sigma_idx, lambda_idx)] = ...
            kernel_ridge_regression_on_dataset(X_train, y_train, X_test, y_test, sigma, lambda);        
       
        %------------------------------------------
        % Calculate K-fold cross validation error
        %------------------------------------------
        mse_cv(sigma_idx, lambda_idx) = get_k_fold_cv_error_new(K, X_train, y_train, sigmas(sigma_idx), lambdas(lambda_idx));
        cv_error = mse_cv(sigma_idx, lambda_idx)
        %------------------------------------------
        % check if cross validation error lower than previous lowest
        %------------------------------------------
        if cv_error < lowest_cv_error
            lowest_cv_error = cv_error;
            best_sigma = sigmas(sigma_idx);
            best_lambda = lambdas(lambda_idx);
        end
   end
end

% -- MSE on the full training dataset using the best parameters
best_sigma_idx = find(sigmas == best_sigma);
best_lambda_idx = find(lambdas == best_lambda);

mse_test_for_best_params = mse_test(best_sigma_idx, best_lambda_idx);
display(mse_test_for_best_params);

% -- Best parameter values from the K-fold cross validation
display(best_sigma);
display(best_lambda);

%% MSE Test 3D plot
figure();
meshc(log10(lambdas),log10(sigmas),log10(mse_test))
xlabel('Lambda');
ylabel('Sigma');
zlabel('Mean Squared Error');
%% MSE Train 3D plot
hold on;
meshc(log10(lambdas),log10(sigmas),log10(mse_cv))
% meshc(log10(lambdas),log10(sigmas),log10(mse_train))

%title('MSE of Test data (top plane) and Training data (bottom plane), all values in log10 scale.');