%% ENERGY_GRAD_J_TILDE - Evaluate filter gradient
% |G = ENERGY_GRAD_J_TILDE(THIS, X)| computes the filter gradient of
% the energy of THIS for each filter and stacks them in the output vector G.
% G is "scaled", i.e. each filter gradient is divided by its #cliques * #images.
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
% $Id: energy_grad_J_tilde.m 240 2011-05-30 16:24:20Z uschmidt $

function g = energy_grad_J_tilde(this, x)
  
  nimages = size(x, 2);
  nfilters = this.nfilters;
  nfilterparams = size(this.J_tilde, 1);
  
  g = zeros(nfilterparams*nfilters, 1);
  
  for j = 1:nimages
    % convert column images into 2D
    img = reshape(x(:, j), this.imdims);
    img_cliques = this.img_cliques(img);
    Ax = this.A * img_cliques;
    
    % compute gradients for all filters
    for i = 1:nfilters
      R = (i-1)*nfilterparams+1:i*nfilterparams;
      g(R) = g(R) + foe_grad_filter(this, i, img, Ax);
    end
  end
  
  g = -g / nimages;
end


function g = foe_grad_filter(this, i, img, Ax)
  d_filter = this.conv2(img, this.filter(i));
  grad_energy = this.experts{min(i,this.nexperts)}.log_grad_x(d_filter(:)');
  g = sum(bsxfun(@times, Ax, grad_energy(:)'), 2) / numel(d_filter);
end