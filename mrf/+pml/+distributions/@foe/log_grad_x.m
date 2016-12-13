%% LOG_GRAD_X - Evaluate gradient of log of FoE
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
%           Uwe Schmidt, Department of Computer Science, TU Darmstadt
%  Contact: sroth@cs.tu-darmstadt.de, uwe.schmidt@gris.tu-darmstadt.de
% 
% Project page:  http://www.gris.tu-darmstadt.de/research/visinf/software/index.en.htm

% Copyright 2009-2011 TU Darmstadt, Darmstadt, Germany.
% $Id: log_grad_x.m 240 2011-05-30 16:24:20Z uschmidt $

function g = log_grad_x(this, x)

  npixels  = size(x, 1);
  nimages  = size(x, 2);
  
  g = zeros(npixels, nimages);
  for i = 1:nimages
    % Convert column images into 2D
    img = reshape(x(:, i), this.imdims);
    
    g(:, i) = -reshape(foe_grad(this, img), npixels, 1) - this.epsilon * x(:, i);
  end
end


function g = foe_grad(this, X)
  
  nfilters = this.nfilters;
  nexperts = this.nexperts;
  g = 0;
  
  for i = 1:nfilters
    d_filter = this.conv2(X, this.filter(i));
    d_filter_grad = reshape(-this.experts{min(i,nexperts)}.log_grad_x(d_filter(:)'), size(d_filter));
    g = g + this.imfilter(d_filter_grad, this.filter(i));
  end
end