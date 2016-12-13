%% ENERGY_GRAD_WEIGHTS - Evaluate GSM weight gradients
% |G = ENERGY_GRAD_WEIGHTS(THIS, X)| computes the weight gradients of
% the energy of THIS for each expert and stacks them in the output vector G.
% G is "scaled", i.e. each expert's gradient is divided by its #cliques * #images.
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
% $Id: energy_grad_weights.m 240 2011-05-30 16:24:20Z uschmidt $

function g = energy_grad_weights(this, x)
  
  nimages = size(x, 2);
  nexperts = this.nexperts;
  
  % generate weight indices for each expert
  ntotalweights = 0;
  weight_idx = cell(1, nexperts);
  for i = 1:nexperts
    weight_idx{i} = ntotalweights+1:ntotalweights+this.experts{i}.nscales;
    ntotalweights = weight_idx{i}(end);
  end
  
  % 'big' weight vector containing all expert weights
  g = zeros(ntotalweights, 1);
  
  for i = 1:nimages
    % convert column images into 2D
    img = reshape(x(:, i), this.imdims);
    % add weight gradient of image i
    g = g + foe_gsm_weight_grad(this, img, weight_idx);
  end
  
  % weight gradient of energy, not log
  g = -g / nimages;
end


function g = foe_gsm_weight_grad(this, X, weight_idx)
  
  nfilters = this.nfilters;
  nexperts = this.nexperts;
  ntotalweights = weight_idx{end}(end);
  
  g = zeros(ntotalweights, 1);
  
  for i = 1:nfilters
    j = min(i,nexperts);
    d_filter = this.conv2(X, this.filter(i));
    g_filter = this.experts{j}.log_grad_weights(d_filter(:)');
    g(weight_idx{j}) = g(weight_idx{j}) + g_filter(:) / numel(d_filter);
  end
end
