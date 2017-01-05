function this = em(this, x, niters)
  
  
  if (nargin < 3)
    niters = 10;
  end
  
  ndims   = this.ndims;
  nscales = this.nscales;
  ndata   = size(x, 2);
  
  x_mu = bsxfun(@minus, x, this.mu);
  if (iscell(this.precision))
    norm_const = zeros(nscales, 1);
    maha       = zeros(nscales, ndata);
    for j = 1:nscales
      norm_const(j) = sqrt(det(this.precision{j})) / ((2 * pi) ^ (ndims / 2));
      maha(j, :) = sum(x_mu .* (this.precision{j} * x_mu), 1);
    end
  else
    norm_const = sqrt(det(this.precision)) / ((2 * pi) ^ (ndims / 2));
    maha = repmat(sum(x_mu .* (this.precision * x_mu), 1), nscales, 1);
  end
  
  for i = 1:niters
    y = repmat(norm_const .* this.weights(:) .* this.scales(:) .^ (ndims/2), 1, ndata) .* ... 
        exp(-0.5 * repmat(this.scales(:), 1, ndata) .* maha);
    
    invld = (sum(y, 1) <= 0);
    y(:, find(invld)) = 1;
    
    gamma = y ./ repmat(sum(y, 1), nscales, 1);
    
    this.weights = mean(gamma, 2)';
    % this.weights
  end
end