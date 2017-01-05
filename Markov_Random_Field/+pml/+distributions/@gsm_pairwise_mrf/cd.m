function [this, report] = cd(this, x, niters, options)
  
  if nargin < 2, niters = 1; end
  if nargin < 3, options = struct; end
  
  % init samples with data
  func = @(alpha, data) log_grad_log_weights(this, alpha, data, data, niters, true);
  
  out = cell(1, max(1,nargout));
  [out{:}] = estimator_helper(this, func, x, options);
  this = out{1};
  if nargout > 1, report = out{2}; end
  
end