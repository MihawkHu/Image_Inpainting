function L = energy(this, x)
  
  nimages  = size(x, 2);
  
  L = zeros(1, nimages);
  for i = 1:nimages      
    % Convert column images into 2D
    img = reshape(x(:, i), this.imdims);
    
    L(i) = foe_energy(this, img) + 0.5 * this.epsilon * x(:, i)' * x(:, i);
  end
end


function E = foe_energy(this, X)
  
  nfilters = this.nfilters;
  nexperts = this.nexperts;
  E = 0;

  for i = 1:nfilters
    d_filter = this.conv2(X, this.filter(i));
    E = E + sum(this.experts{min(i,nexperts)}.energy(d_filter(:)'));
  end
end