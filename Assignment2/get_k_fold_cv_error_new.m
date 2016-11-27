function [ error ] = get_k_fold_cv_error_new(K, X_train, y_train, sigma, lamda)
% get_k_fold_cv_error Calculates the K fold cross validation

num_datapoints = size(X_train,1);
shuffle_idxs = randperm(num_datapoints);

% -- datapoints in each subset
partition_size = floor(num_datapoints / K);
start_idx = 1;

% -- initialize an empty list of mean_squared_errors
mserrors = zeros(1,K);

for k = 1:K-1
    % -- Calculations to get the Train and Validation split
    end_idx = min(start_idx+partition_size-1,length(shuffle_idxs)); % ending index of validation set
    validation_set_idxs = (start_idx:end_idx);
    
    valid_set_X = X_train(validation_set_idxs,:);
    valid_set_Y = y_train(validation_set_idxs,:);
    
    train_set_idxs = setdiff(shuffle_idxs, validation_set_idxs);
    train_set_X = X_train(train_set_idxs,:);
    train_set_Y = y_train(train_set_idxs,:);
    %------- Actual Training/Testing happens here.
    [~, mserrors(1,k)] = ...
        kernel_ridge_regression_on_dataset(train_set_X, train_set_Y, valid_set_X, valid_set_Y, sigma, lamda);
    %-------
    start_idx = end_idx+1; % starting index of next validation set
end

% -- Calculating the last Kth-validation set now.
validation_set_idxs = (start_idx:num_datapoints);

train_set_idxs = setdiff(shuffle_idxs, validation_set_idxs);
train_set_X = X_train(train_set_idxs,:);
train_set_Y = y_train(train_set_idxs,:);
 
%------- Actual Training/Testing of the Kth-validation set happens here.
[~, mserrors(1,K)] = ...
    kernel_ridge_regression_on_dataset(train_set_X, train_set_Y, valid_set_X, valid_set_Y, sigma, lamda);

%------- Calculate the Average MSE for the K validation sets.
error = mean(mserrors);
end

