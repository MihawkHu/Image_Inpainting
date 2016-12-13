%% ASSEMBLE_IMAGES - Assemble a single big image from multiple, equally sized, images.
% |I = ASSEMBLE_IMAGES(X, SZ[, IPR])| assumes an image of size SZ in
% each column of X. The parameter IPR determines the number of
% images per row of the generated image I. If omitted, I will be made
% as square as possible.
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
% $Id: assemble_images.m 240 2011-05-30 16:24:20Z uschmidt $

function I = assemble_images(X, sz, ipr)
  
  [npixels nimages] = size(X);
  h = sz(1); w = sz(2);
  if nargin < 3, ipr = ceil(sqrt(nimages*(h/w))); end
  
  I = zeros(sz .* [ceil(nimages/min(ipr,nimages)) min(ipr,nimages)]);
  
  row = 1; col = 1;
  for i = 1:nimages
    y = (row-1)*h+1;
    x = (col-1)*w+1;
    I(y:y-1+h, x:x-1+w) = reshape(X(:,i), sz);
    col = col + 1;
    if col > ipr, col = 1; row = row + 1; end
  end
  
end