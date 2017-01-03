function g = energy_grad_weights(this, x)
  
  nimages = size(x, 2);
  nexperts = this.nexperts;
  
  % generate weight indices for each expert
  ntotalweights = 0;
  weight_idx = cell(1, nexperts);
  for i = 1:nexperts
    weight_idx{i} = ntotalweights+1:ntotalweights+this.experts{i}.nscales;
    ntotalweights = weight_idx{i}(end);
  end
  
  % 'big' weight vector containing all expert weights
  g = zeros(ntotalweights, 1);
  
  for i = 1:nimages
    % convert column images into 2D
    img = reshape(x(:, i), this.imdims);
    % add weight gradient of image i
    g = g + foe_gsm_weight_grad(this, img, weight_idx);
  end
  
  % weight gradient of energy, not log
  g = -g / nimages;
end


function g = foe_gsm_weight_grad(this, X, weight_idx)
  
  nfilters = this.nfilters;
  nexperts = this.nexperts;
  ntotalweights = weight_idx{end}(end);
  
  g = zeros(ntotalweights, 1);
  
  for i = 1:nfilters
    j = min(i,nexperts);
    d_filter = this.conv2(X, this.filter(i));
    g_filter = this.experts{j}.log_grad_weights(d_filter(:)');
    g(weight_idx{j}) = g(weight_idx{j}) + g_filter(:) / numel(d_filter);
  end
end
