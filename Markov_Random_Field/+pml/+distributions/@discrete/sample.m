function s = sample(this, nsamples)
  if (nargin < 2)
    nsamples = 1;
  end
  
  %% Sample from the cumulative weights
  ind = montecarlo(this.cum_weights, rand(nsamples, 1), nsamples);
  
  %% Convert linear indices into indices along each dimension
  sub = cell(1, this.ndims);
  [sub{:}] = ind2sub(size(this.weights), ind);
  s = cell2mat(sub)';
  
  s = bsxfun(@plus, bsxfun(@times, s - 1, this.domain_stride(:)), ...
             this.domain_start(:));
end
