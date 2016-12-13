%% MLE - Maximum likelihood estimation of discrete distribution
% |THIS = MLE(THIS, X)| estimates the weights of the discrete distribution
% THIS from data X.
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
% $Id: mle.m 240 2011-05-30 16:24:20Z uschmidt $
    
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
