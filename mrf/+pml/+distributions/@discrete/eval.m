%% EVAL - Evaluate, i.e. compute, the discrete distribution
% |P = EVAL(THIS, X)| evaluates the discrete probability P of the
% distribution THIS at one or more points X. X is either a row vector or a
% matrix of row vectors (in case of multiple points).
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
% $Id: eval.m 240 2011-05-30 16:24:20Z uschmidt $
    
function p = eval(this, x)
  tmp = bsxfun(@minus, x, this.domain_start(:));
  idx = 1 + round(bsxfun(@rdivide, tmp, this.domain_stride(:)));
  
  ndims = size(x, 1);
  dims = mat2cell(idx, ones(1, ndims), size(x, 2));
  
  p = this.weights(sub2ind(size(this.weights), dims{:}));
  p = p(:)';
end
