function g = log_grad_x(this, x)

  npixels  = size(x, 1);
  nimages  = size(x, 2);
  
  g = zeros(npixels, nimages);
  for i = 1:nimages
    % Convert column images into 2D
    img = reshape(x(:, i), this.imdims);
    
    g(:, i) = -reshape(foe_grad(this, img), npixels, 1) - this.epsilon * x(:, i);
  end
end


function g = foe_grad(this, X)
  
  nfilters = this.nfilters;
  nexperts = this.nexperts;
  g = 0;
  
  for i = 1:nfilters
    d_filter = this.conv2(X, this.filter(i));
    d_filter_grad = reshape(-this.experts{min(i,nexperts)}.log_grad_x(d_filter(:)'), size(d_filter));
    g = g + this.imfilter(d_filter_grad, this.filter(i));
  end
end