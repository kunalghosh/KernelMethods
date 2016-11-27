function kernel_mat = gaussian_kernel(X, Z, sigma)
% X is (n,d)
% Z is (m,d)
% sigma is scalar
% gaussian kernel = exp( -0.5 * (1/sigma^2)* || phi(x1) - phi(x2) ||^2)
% Assuming, here X and Z are already projected in phi (feature) space.
% So in this case, gaussian kernel = exp( -0.5 * (1/sigma^2)* || x1 - x2 ||^2)
% Dimension of kernel_mat is (n,m)
    [n,d] = size(X);
    [m,d] = size(Z);
    kernel_mat = zeros(n,m);
    for row = 1:n
        kernel_mat(row,:) = sum(bsxfun(@minus,Z,X(row,:)).^2,2)';
    end
    kernel_mat = exp(kernel_mat * (-0.5 /(sigma^2)));
end