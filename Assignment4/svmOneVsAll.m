function [ytest_pred, ytrain_pred] = svmOneVsAll(X_train, X_test, y_train, kernel, C)
    
    [N,d] = size(X_train);
    [M,dt] = size(X_test);
    
    % K = set of classes
    K = unique(y_train);
    K_size = length(K);
    
    scores_train = zeros(N, K_size);
    scores_test = zeros(M, K_size);
    
    for i = 1:K_size
        k = K(i);
        SVMModel = fitcsvm(X_train,y_train == k,'KernelFunction',kernel,'BoxConstraint',C);
        [label_train,score_train] = predict(SVMModel, X_train);
        [label_test ,score_test] = predict(SVMModel, X_test);
        
        % scores_train(:,i) = max(score_train,[],2);
        % scores_test(:,i) = max(score_test,[],2);
        
        scores_train(:,i) = score_train(:,2);
        scores_test(:,i)  = score_test(:,2);
    end
    [~,ytest_pred] = max(scores_test, [], 2);
    [~,ytrain_pred] = max(scores_train, [], 2);
end
