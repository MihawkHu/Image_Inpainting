%% ENERGY - Evaluate energy of FoE.
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
% $Id: energy.m 240 2011-05-30 16:24:20Z uschmidt $

function L = energy(this, x)
  
  nimages  = size(x, 2);
  
  L = zeros(1, nimages);
  for i = 1:nimages      
    % Convert column images into 2D
    img = reshape(x(:, i), this.imdims);
    
    L(i) = foe_energy(this, img) + 0.5 * this.epsilon * x(:, i)' * x(:, i);
  end
end


function E = foe_energy(this, X)
  
  nfilters = this.nfilters;
  nexperts = this.nexperts;
  E = 0;

  for i = 1:nfilters
    d_filter = this.conv2(X, this.filter(i));
    E = E + sum(this.experts{min(i,nexperts)}.energy(d_filter(:)'));
  end
end