%% SAMPLE - Sample from a GSM probability density
% See help for the base class density.
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
% $Id: sample.m 240 2011-05-30 16:24:20Z uschmidt $

function y = sample(this, nsamples)
  
  
  ndims   = length(this.mean);
  nscales = length(this.weights);
  
  this.weights = this.weights(end:-1:1);
  
  cw = cumsum(this.weights);
  mix_comp = sum(repmat(rand(1, nsamples), nscales, 1) <= ...
                 repmat(cw(:), 1, nsamples), 1);
  
  % Whitening transform of precision
  
  if (iscell(this.precision))
    tmp = zeros(ndims, nsamples);
    r   = randn(ndims, nsamples);
    for j = 1:nscales
      tmp2 = chol(this.precision{j}) \ r;
      idx = repmat(mix_comp == j, ndims, 1);
      tmp(idx) = tmp2(idx);
    end
  else
    tmp = chol(this.precision) \ randn(ndims, nsamples);
  end
  
  y = repmat(this.mean, 1, nsamples) + tmp ./ ...
      repmat(sqrt(this.scales(mix_comp)), ndims, 1);
end
  
