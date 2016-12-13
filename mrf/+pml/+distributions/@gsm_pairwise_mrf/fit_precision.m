%% FIT_PRECISION - Fit expert's precision to empirical precision of X
% |[THIS, DZ] = FIT_PRECISION(THIS, X)| sets each expert's precision to the
% empirical of its filter responses of the images X. DZ is a cell
% array of length nfilters that contains the discrete distributions of filter
% respones of each filter for the given images X.
% 
% This file is part of the implementation as described in the papers:
% 
%  Uwe Schmidt, Qi Gao, Stefan Roth.
%  A Generative Perspective on MRFs in Low-Level Vision.
%  IEEE Conference on Computer Vision and Pattern Recognition (CVPR'10), San Francisco, USA, June 2010.
%
%  Uwe Schmidt, Kevin Schelten, Stefan Roth.
%  Bayesian Deblurring with Integrated Noise Estimation.
%  IEEE Conference on Computer Vision and Pattern Recognition (CVPR'11), Colorado Springs, Colorado, June 2011.
%
% Please cite the appropriate paper if you are using this code in your work.
% 
% The code may be used free of charge for non-commercial and
% educational purposes, the only requirement is that this text is
% preserved within the derivative work. For any other purpose you
% must contact the authors for permission. This code may not be
% redistributed without permission from the authors.
%
%  Author:  Uwe Schmidt, Department of Computer Science, TU Darmstadt
%  Contact: uwe.schmidt@gris.tu-darmstadt.de
% 
% Project page:  http://www.gris.tu-darmstadt.de/research/visinf/software/index.en.htm

% Copyright 2009-2011 TU Darmstadt, Darmstadt, Germany.
% $Id: fit_precision.m 240 2011-05-30 16:24:20Z uschmidt $

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