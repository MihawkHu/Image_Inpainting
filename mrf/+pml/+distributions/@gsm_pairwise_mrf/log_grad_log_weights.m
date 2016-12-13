%% LOG_GRAD_LOG_WEIGHTS - Evaluate gradient w.r.t. log of weights
% |G = LOG_GRAD_LOG_WEIGHTS(THIS, ALPHA, X, S, NITERS, [NEG])| computes the
% gradient G w.r.t. to log of weights of the normalized MRF over all images X.
% ALPHA is assumed to be the log concatenated weight vector of all expert
% weights. Evaluation has to rely on samples since the partition function
% is unknown. Gibbs sampling for NITERS rounds is used where samples are
% initialized by S (one chain per sample). NEG cat be set to 'true' to
% compute the negative gradient G.
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
% $Id: log_grad_log_weights.m 240 2011-05-30 16:24:20Z uschmidt $

function grad = log_grad_log_weights(this, alpha, x, s, niters, neg)
  
  if nargin < 6 || ~islogical(neg), neg = false; end
  if neg, sgn = -1; else sgn = 1; end
  
  nimages = size(x, 2);
  [npixels, nsamples] = size(s);
  nexperts = this.nexperts;
  
  % convert weights from log-space, set & normalize
  this.weights = exp(alpha);
  % get normalized weights
  omega = this.weights;
  
  for i = 1:nsamples
    for iter = 1:niters
      z = this.sample_z(s(:,i));
      s(:,i) = this.sample_x(z, s(:,i));
    end
  end
  
  % "scaled" weight gradient
  grad_x = sgn * this.energy_grad_weights(x);
  grad_s = sgn * this.energy_grad_weights(s);
  % gradient w.r.t. log of weights
  grad_x = grad_x .* omega;
  grad_s = grad_s .* omega;
  
  grad = (nimages/nsamples) * grad_s - grad_x;
  
end