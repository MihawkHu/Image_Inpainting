%% SAMPLE - Draw i.i.d. samples from the discrete distribution
% |S = SAMPLE(THIS[, NSAMPLES])| draws i.i.d. samples (default 1 sample)
% from the discrete distribution THIS and returns them as a matrix S. X is
% either a row vector (in case of 1 sample) or a matrix of row vectors (in
% case of multiple samples). The optional argument NSAMPLES specifies the
% number of i.i.d. samples to draw.
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
    
function s = sample(this, nsamples)
  if (nargin < 2)
    nsamples = 1;
  end
  
  %% Sample from the cumulative weights
  ind = montecarlo(this.cum_weights, rand(nsamples, 1), nsamples);
  
  %% Convert linear indices into indices along each dimension
  sub = cell(1, this.ndims);
  [sub{:}] = ind2sub(size(this.weights), ind);
  s = cell2mat(sub)';
  
  s = bsxfun(@plus, bsxfun(@times, s - 1, this.domain_stride(:)), ...
             this.domain_start(:));
end
