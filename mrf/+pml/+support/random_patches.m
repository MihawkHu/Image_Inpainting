%% RANDOM_PATCHES - Extracts random patches from a cell array of images.
% |I = RANDOM_PATCHES(X, S, [N])| extracts random patches of size S
% from a cell array of images X. For each image, N (default 1) patches are
% uniformly chosen at random and stored as columns in I. Hence,
% SIZE(I) = PROD(S) x (N * LENGTH(X)). Note that, if N > 1, patches from the
% same image may overlap.
% 
% This file is part of the implementation as described in the paper:
% 
%  Uwe Schmidt, Qi Gao, Stefan Roth.
%  A Generative Perspective on MRFs in Low-Level Vision.
%  IEEE Conference on Computer Vision and Pattern Recognition (CVPR'10), San Francisco, USA, June 2010.
% 
% Please cite the paper if you are using this code in your work.
% 
% The code may be used free of charge for non-commercial and
% educational purposes, the only requirement is that this text is
% preserved within the derivative work. For any other purpose you
% must contact the authors for permission. This code may not be
% redistributed without permission from the authors.
%
%  Author:  Uwe Schmidt, Department of Computer Science, TU Darmstadt
%  Contact: mail@uweschmidt.org
% 
% Project page:  http://www.gris.tu-darmstadt.de/research/visinf/software/index.en.htm

% Copyright 2009-2010 TU Darmstadt, Darmstadt, Germany.
% $Id: random_patches.m 232 2010-07-08 13:18:35Z uschmidt $

function I = random_patches(X, s, npatches)
  
  nimages = length(X);
  if nargin < 3, npatches = 1; end
  I = zeros(prod(s), nimages * npatches);
  
  for i = 1:nimages
    img = X{i};
    [h w] = size(img);
    if ~all([h w] > s), error('Image patch larger than image.'); end
    for j = 1:npatches
      y = pml.support.randi(h-s(1)+1);
      x = pml.support.randi(w-s(2)+1);
      patch = img(y:y-1+s(1), x:x-1+s(2));
      I(:,(i-1)*npatches + j) = patch(:);
    end
  end
end