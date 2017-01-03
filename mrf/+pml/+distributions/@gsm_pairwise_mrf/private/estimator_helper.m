function [this, report] = estimator_helper(this, func, x, options)
  
  nimages  = size(x, 2);
  nexperts = this.nexperts;
  nfilters = this.nfilters;
  
  % start weights
  alpha0 = log(this.weights);
  
  % stochastic gradient descent
  out = cell(1, max(1,nargout));
  [out{:}] = pml.numerical.sgd(func, alpha0, x, options);
  alpha = out{1};
  if nargout > 1, report = out{2}; end
  
  % set estimated weights (converted back from log-space)
  this.weights = exp(alpha);
end