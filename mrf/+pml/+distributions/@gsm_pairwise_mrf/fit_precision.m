function [this, dz] = fit_precision(this, x)
  
  nimages  = size(x, 2);
  nexperts = this.nexperts;
  nfilters = this.nfilters;
  
  fil = cell(1, nfilters);
  nelements = zeros(1, nfilters);
  switch this.conv_method
    case 'valid'
      for j = 1:nfilters
        [frows, fcols] = size(this.filter(j));
        % #elements of convolved image with 'valid' option
        nelements(j) = (this.imdims(1)-frows+1) * (this.imdims(2)-fcols+1);
        fil{j} = zeros(1, nelements(j) * nimages);
      end
    case 'circular'
      for j = 1:nfilters
        nelements(j) = prod(this.imdims);
        fil{j} = zeros(1, nelements(j) * nimages);
      end
    otherwise
      error('Not implemented: ''%s''.', this.conv_method);
  end
  
  for i = 1:nimages
    img = reshape(x(:, i), this.imdims);
    for j = 1:nfilters
      % apply filter
      d_filter = this.conv2(img, this.filter(j));
      % collect filter responses in a big vector
      fil{j}(nelements(j)*(i-1)+1:nelements(j)*i) = d_filter(:);
    end
  end
  
  % special case of 1 expert
  if nexperts == 1
    fil{1} = horzcat(fil{1}, fil{2});
  end
  
  if nargout > 1
    dz = cell(1, nexperts);
  end
  
  for i = 1:nexperts
    if nargout > 1
      % create discrete distribution, set domain start and stride, fit to filter responses
      dz{i} = pml.distributions.discrete(ones(511,1));
      f = this.filter(i);
      % filter is assumed to be scaled according to pixel distance
      dz{i}.domain_stride = max(f(:));
      dz{i}.domain_start = -255*dz{i}.domain_stride;
      dz{i} = dz{i}.mle(fil{i});
    end
    
    % set expert's precision to inverse of empirical variance of filter
    % this.experts{i}.precision = 1 / dz{i}.covariance;
    
    this.experts{i}.precision = 1 / var(fil{i});
    % fprintf('Expert %d variance = %f\n', i, 1 / this.experts{i}.precision)
  end
  
end