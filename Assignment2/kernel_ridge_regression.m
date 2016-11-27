function alpha = kernel_ridge_regression(K, y, lambda)
    % K is of size (n,n)
    % y is of size (n,1)
    % lambda is a scalar
    alpha = (lambda * eye(size(K)) + K)\y;
end


