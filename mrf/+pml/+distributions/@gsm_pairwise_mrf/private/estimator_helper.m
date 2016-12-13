%% ESTIMATOR_HELPER - Helper method for minimization
% |[THIS, REPORT] = ESTIMATOR_HELPER(THIS, FUNC, X, [OPTIONS])| minimizes
% FUNC with data X using stochastic gradient descent (using OPTIONS).
% The expert's weights are set to the estimated.
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
% $Id: estimator_helper.m 240 2011-05-30 16:24:20Z uschmidt $

function [this, report] = estimator_helper(this, func, x, options)
  
  nimages  = size(x, 2);
  nexperts = this.nexperts;
  nfilters = this.nfilters;
  
  % start weights
  alpha0 = log(this.weights);
  
  % stochastic gradient descent
  out = cell(1, max(1,nargout));
  [out{:}] = pml.numerical.sgd(func, alpha0, x, options);
  alpha = out{1};
  if nargout > 1, report = out{2}; end
  
  % set estimated weights (converted back from log-space)
  this.weights = exp(alpha);
end