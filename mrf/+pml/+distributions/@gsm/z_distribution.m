%% Z_DISTRIBUTION - Distribution over scale mixture components Z, given X
% |P = Z_DISTRIBUTION(THIS, X)| computes a discrete distribution over scales Z
% for every X. P is a matrix of size [nscales x ndata], where every column
% contains normalized weights.
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
%  Author:  Stefan Roth, Department of Computer Science, TU Darmstadt
%  Contact: sroth@cs.tu-darmstadt.de
% 
% Project page:  http://www.gris.tu-darmstadt.de/research/visinf/software/index.en.htm

% Copyright 2009-2011 TU Darmstadt, Darmstadt, Germany.
% $Id: z_distribution.m 240 2011-05-30 16:24:20Z uschmidt $

function p = z_distribution(this, x)
  
  ndims   = this.ndims;
  nscales = this.nscales;
  ndata   = size(x, 2);
  
  x_mu = bsxfun(@minus, x, this.mu);
  if (iscell(this.precision))
    norm_const = zeros(nscales, 1);
    maha       = zeros(nscales, ndata);
    for j = 1:nscales
      norm_const(j) = sqrt(det(this.precision{j})) / ((2 * pi) ^ (ndims / 2));
      maha(j, :) = sum(x_mu .* (this.precision{j} * x_mu), 1);
    end
  else
    norm_const = sqrt(det(this.precision)) / ((2 * pi) ^ (ndims / 2));
    maha = sum(x_mu .* (this.precision * x_mu), 1);
  end
  
  y = bsxfun(@times, norm_const .* this.weights(:) .* (this.scales(:) .^ (ndims/2)), ...
      exp(bsxfun(@times, -0.5 * this.scales(:), maha)));
  p = bsxfun(@rdivide, y , sum(y, 1));
  
end