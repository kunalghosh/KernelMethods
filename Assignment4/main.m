%% Load Data
X_train = load('./data/X_train.txt');
X_test  = load('./data/X_test.txt');
y_train = load('./data/y_train.txt');
y_test  = load('./data/y_test.txt');

%% One Vs All Classification
[ytest_pred, ytrain_pred] = svmOneVsAll(X_train, X_test, y_train, 'linear', 10);

sum([ytrain_pred == y_train])
sum([ytest_pred == y_test])

histogram(y_test(ytest_pred ~= y_test))
xlabel('Classes');
ylabel('Count of misclassified digits.');
title('one-vs-all model')
grid on
%% All vs All Classification
[ytest_pred, ytrain_pred] = svmAllVsAll(X_train, X_test, y_train, 'linear', 1);

