function y_pred = parzen_classify(Kx_train, Kx_train_test, y_train)
  m_pos = sum(y_train == 1);
  m_neg = length(y_train) - m_pos;
  % size(Kx_train) (800,800)
  % size(Kx_train_test) (800,200)
  b = zeros(2,1);
  negidxs = find(y_train == -1);
  for i = negidxs'
    b(1) = b(1) + sum(Kx_train(i,negidxs));
  end
  b(1) = b(1)/(2*m_neg^2);
  
  posidxs = find(y_train == 1);
  for i = posidxs'
    b(2) = b(2) + sum(Kx_train(i,posidxs));
  end
  b(2) = b(2)/(2*m_pos^2);
  
  const = b(1) - b(2);
  
  % here alpha is (800,1)
  alpha = (1/m_pos)*ones(size(y_train));
  alpha(negidxs) = (-1/m_neg);
  
  y_pred = sign((alpha' * Kx_train_test) + const);
end