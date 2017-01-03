function grad = log_grad_theta(this, theta, x, s, niters, neg)
  
  if nargin < 6 || ~islogical(neg), neg = false; end
  if neg, sgn = -1; else sgn = 1; end
  
  nimages = size(x, 2);
  [npixels, nsamples] = size(s);
  nexperts = this.nexperts;
  
  nweights = length(this.weights);
  alpha = theta(1:nweights);
  J_tilde = theta(nweights+1:end);
  
  % set filter
  this.J_tilde(:) = J_tilde;
  
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
  grad_x_alpha = sgn * this.energy_grad_weights(x);
  grad_s_alpha = sgn * this.energy_grad_weights(s);
  % gradient w.r.t. log of weights
  grad_x_alpha = grad_x_alpha .* omega;
  grad_s_alpha = grad_s_alpha .* omega;
  grad_alpha = (nimages/nsamples) * grad_s_alpha - grad_x_alpha;
  
  % "scaled" filter gradient
  grad_x_J_tilde = sgn * this.energy_grad_J_tilde(x);
  grad_s_J_tilde = sgn * this.energy_grad_J_tilde(s);
  % gradient w.r.t. J_tilde
  grad_J_tilde = (nimages/nsamples) * grad_s_J_tilde - grad_x_J_tilde;
  
  grad = [grad_alpha; grad_J_tilde];
end