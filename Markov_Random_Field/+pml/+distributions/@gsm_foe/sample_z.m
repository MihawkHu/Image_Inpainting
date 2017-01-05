function z = sample_z(this, x)
  
  z = cell(1, this.nfilters);
  pz = this.z_distribution(x);
  
  % sample scales
  for i = 1:this.nfilters
    % z{i} = sample_discrete(pz{i});
    z{i} = this.experts{min(i,this.nexperts)}.scales(sample_discrete(pz{i}));
  end
  
end


% draw a single sample from each discrete distribution (normalized weights in columns)
% domain start & stride is assumed to be 1
function xs = sample_discrete(pmfs)
  [nweights, ndists] = size(pmfs);
  % create cdfs
  cdfs = cumsum(pmfs, 1);
  % uniform sample for each distribution
  ys = rand(1, ndists);
  % subtract uniform sample and set 1s where difference >= 0
  inds = bsxfun(@minus, cdfs, ys) >= 0;
  % multiply each column with weight indices
  inds = bsxfun(@times, inds, (1:nweights)');
  % set 0s to NaN
  inds(inds == 0) = NaN;
  % min. weight index > 0 (NaNs are ignored by 'min')
  xs = min(inds, [], 1);
end