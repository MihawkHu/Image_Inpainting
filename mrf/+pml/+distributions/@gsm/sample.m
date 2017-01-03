function y = sample(this, nsamples)
  
  
  ndims   = length(this.mean);
  nscales = length(this.weights);
  
  this.weights = this.weights(end:-1:1);
  
  cw = cumsum(this.weights);
  mix_comp = sum(repmat(rand(1, nsamples), nscales, 1) <= ...
                 repmat(cw(:), 1, nsamples), 1);
  
  % Whitening transform of precision
  
  if (iscell(this.precision))
    tmp = zeros(ndims, nsamples);
    r   = randn(ndims, nsamples);
    for j = 1:nscales
      tmp2 = chol(this.precision{j}) \ r;
      idx = repmat(mix_comp == j, ndims, 1);
      tmp(idx) = tmp2(idx);
    end
  else
    tmp = chol(this.precision) \ randn(ndims, nsamples);
  end
  
  y = repmat(this.mean, 1, nsamples) + tmp ./ ...
      repmat(sqrt(this.scales(mix_comp)), ndims, 1);
end
  
