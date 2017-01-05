function g = energy_grad_J_tilde(this, x)
  
  nimages = size(x, 2);
  nfilters = this.nfilters;
  nfilterparams = size(this.J_tilde, 1);
  
  g = zeros(nfilterparams*nfilters, 1);
  
  for j = 1:nimages
    % convert column images into 2D
    img = reshape(x(:, j), this.imdims);
    img_cliques = this.img_cliques(img);
    Ax = this.A * img_cliques;
    
    % compute gradients for all filters
    for i = 1:nfilters
      R = (i-1)*nfilterparams+1:i*nfilterparams;
      g(R) = g(R) + foe_grad_filter(this, i, img, Ax);
    end
  end
  
  g = -g / nimages;
end


function g = foe_grad_filter(this, i, img, Ax)
  d_filter = this.conv2(img, this.filter(i));
  grad_energy = this.experts{min(i,this.nexperts)}.log_grad_x(d_filter(:)');
  g = sum(bsxfun(@times, Ax, grad_energy(:)'), 2) / numel(d_filter);
end