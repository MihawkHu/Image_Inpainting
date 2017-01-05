function this = mle(this, X)
  ndims = size(X, 1);
  if ndims > 1, error('NDIMS > 1 not supported.'), end
  
  %% Set up indices
  ind = cell(1, ndims);
  for i = 1:ndims
    m = max(max(X(i, :)), this.domain_start(i) + (size(this.weights, i) - 1) * this.domain_stride(i));
    ind{i} = [-Inf (this.domain_start(i) + 0.5 * this.domain_stride(i)):this.domain_stride(i):(m - 0.5 * this.domain_stride(i)) Inf];
  end
  
  %% Simply estimate the relative frequency of each bin
  w = histc(X', ind{:});
  
  %% Remove the last histogram bin corresponding to Inf
  for i = 1:ndims
    ind{i} = 1:(size(w, i) - 1);
  end
  this.weights = w(ind{:});
end
