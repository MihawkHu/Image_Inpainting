function [this, report] = estimator_helper(this, func, x, options)
  
  nimages  = size(x, 2);
  nexperts = this.nexperts;
  nfilters = this.nfilters;
  
  alpha0 = log(this.weights);
  J_tilde0 = this.J_tilde(:);
  theta0 = [alpha0; J_tilde0];
  
  % stochastic gradient descent
  out = cell(1, max(1,nargout));
  [out{:}] = pml.numerical.sgd(func, theta0, x, options);
  theta = out{1};
  if nargout > 1, report = out{2}; end
  
  nweights = length(this.weights);
  alpha = theta(1:nweights);
  J_tilde = theta(nweights+1:end);
  
  % set estimated weights (converted back from log-space)
  this.weights = exp(alpha);
  % set filter
  this.J_tilde(:) = J_tilde;
end