function grad = log_grad_log_weights(this, alpha, x, s, niters, neg)
  
  if nargin < 6 || ~islogical(neg), neg = false; end
  if neg, sgn = -1; else sgn = 1; end
  
  nimages = size(x, 2);
  [npixels, nsamples] = size(s);
  nexperts = this.nexperts;
  
  % convert weights from log-space, set & normalize
  this.weights = exp(alpha);
  % get normalized weights
  omega = this.weights;
  
  for i = 1:nsamples
    for iter = 1:niters
      z = this.sample_z(s(:,i));
      s(:,i) = this.sample_x(z, s(:,i));
    end
  end
  
  % "scaled" weight gradient
  grad_x = sgn * this.energy_grad_weights(x);
  grad_s = sgn * this.energy_grad_weights(s);
  % gradient w.r.t. log of weights
  grad_x = grad_x .* omega;
  grad_s = grad_s .* omega;
  
  grad = (nimages/nsamples) * grad_s - grad_x;
  
end