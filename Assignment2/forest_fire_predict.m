% ========================
% Predicting forest fires. 
% ========================

load data_all.mat
% pow10 = @(t){10^t};
% lambdas = cell2mat(arrayfun(pow10,(-3:7)));
% sigmas = cell2mat(arrayfun(pow10, (-1:4)));

lambdas = 10 .^ (-3:7);
sigmas = 10 .^ (-1:4);

% Matrices to save the value of MSE for train and test data.
sigma_len = size(sigmas,2);
lambda_len = size(lambdas,2);
result_shape = [sigma_len, lambda_len];
mse_test = zeros(result_shape);
mse_train = zeros(result_shape);

lowest_cv_error = Inf;
best_sigma  = 0;
best_lambda = 0;



K = 5; % K fold cross validation
figure();
title('Train and Test error for different values of sigma and lambda (error values in log scale)');
for sigma_idx = 1:sigma_len
   K_train_train = gaussian_kernel(X_train,X_train,sigmas(sigma_idx));
   K_train_test = gaussian_kernel(X_train,X_test,sigmas(sigma_idx));
   for lambda_idx = 1:lambda_len

        alpha = kernel_ridge_regression(K_train_train, y_train, lambdas(lambda_idx));
        mse_test(sigma_idx, lambda_idx) = get_prediction_mse(alpha, K_train_test, y_test); 
        mse_train(sigma_idx, lambda_idx) = get_prediction_mse(alpha, K_train_train, y_train);
        
        %----------------------------------------
        % Calculate K-fold cross validation error
        %----------------------------------------
        error = get_k_fold_cv_error(K, X_train, y_train, sigmas(sigma_idx), lambdas(lambda_idx));
        
        %-----------------------------------------------------------
        % check if cross validation error lower than previous lowest
        %-----------------------------------------------------------
        if error < lowest_cv_error
            lowest_cv_error = error;
            best_sigma = sigmas(sigma_idx);
            best_lambda = lambdas(lambda_idx);
        end
   end
   subplot(sigma_len,1,sigma_idx);
   loglog(lambdas, mse_train(sigma_idx,:));
   hold on;
   loglog(lambdas, mse_test(sigma_idx,:));
   title(num2str(['Sigma ' num2str(sigmas(sigma_idx))]));
   legend('Train','Test');
   grid on; 
end

% -- MSE on the full training dataset using the best parameters

best_sigma_idx = find(sigmas == best_sigma);
best_lambda_idx = find(lambdas == best_lambda);

mse_test_for_best_params = mse_test(best_sigma_idx, best_lambda_idx);
display(mse_test_for_best_params);

% -- Best parameter values from the K-fold cross validation
display(best_sigma);
display(best_lambda);

%% -- MSE Train 3D plot.
figure()
mse_train_col = reshape(mse_train, sigma_len*lambda_len,1);
mse_test_col = reshape(mse_test, sigma_len*lambda_len,1);

[p,q] = meshgrid(sigmas, lambdas);
plot3(log(p(:)),log(q(:)),log(mse_train_col),'ro');
grid;
hold on;
% --- Plotting the lowest MSE
[val idx] = min(mse_train);
[min_val col_idx] = min(val);
row_idx = idx(find(min_val == min(val)));
display('Lowest MSE Train = ')
display(min_val);
display('For parameters:')
sigma = p(row_idx,col_idx);
lamda = q(row_idx,col_idx);
display(sigma);
display(lamda);

plot3(log(p(row_idx,col_idx)),log(q(row_idx,col_idx)),log(mse_train(row_idx,col_idx)),'bo');
xlabel('sigmas');
ylabel('lambdas');
zlabel('Error');
title('MSE Train (All values in log scale)');

%% -- MSE Test 3D plot.
figure()
plot3(log(p(:)),log(q(:)),log(mse_test_col),'ro');
grid;
hold on;
% --- Plotting the lowest MSE
[val idx] = min(mse_test);
[min_val col_idx] = min(val);
row_idx = idx(find(min_val == min(val)));

display('Lowest MSE Test = ')
display(min_val);
display('For parameters:')
sigma = p(row_idx,col_idx);
lamda = q(row_idx,col_idx);
display(sigma);
display(lamda);

plot3(log(p(row_idx,col_idx)),log(q(row_idx,col_idx)),log(mse_test(row_idx,col_idx)),'bo');
xlabel('sigmas');
ylabel('lambdas');
zlabel('Error');
title('MSE Test (All values in log scale)');