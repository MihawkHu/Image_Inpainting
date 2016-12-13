%% KL_DIVERGENCE - Compute the KL-divergence between two discrete distributions
% |D = KL_DIVERGENCE(THIS, OTHER)| computes the
% Kullback-Leibler divergence D(THIS || OTHER) between the two discrete
% distributions THIS and OTHER.
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
% $Id: kl_divergence.m 240 2011-05-30 16:24:20Z uschmidt $

function d = kl_divergence(this, other)
  if (~isequal(size(this.weights), size(other.weights)))
    d = Inf;
    return;
  end 
  
  d = sum(this.weights(:) .* (log(this.weights(:)) - log(other.weights(:))));
end
