function p = eval(this, x)
  
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
    maha = sum(x_mu .* (this.precision * x_mu), 1);
  end
  
  p = bsxfun(@times, norm_const .* this.weights(:) .* (this.scales(:) .^ (ndims/2)), ...
      exp(bsxfun(@times, -0.5 * this.scales(:), maha)));
  p = sum(p, 1);
  
end