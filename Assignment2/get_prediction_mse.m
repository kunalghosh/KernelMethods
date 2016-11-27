function mse = get_prediction_mse(alpha, K_train_test, y_test)
    % alpha is the dual variable (in this case of the ridge regression kernel).
    % K_train_test is the kernel matrix between the train and the test dataset.
    % y_test is the true labels of the test dataset.
    % ---
    % Ridge regression model's prediction is given by
    % g(x) = y'(K+lambda*I)*k
    % where k is  K(train,test(x))
    % Kernel between training data and the test data
    % for which the prediction g(x) is being calculated.
    y_pred = alpha' * K_train_test;
    mse = mean((y_pred' - y_test).^2);
end
