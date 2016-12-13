%% MIRROR_BOUNDARY - Mirror boundary of image I and return padded image
% |I_PADDED = MIRROR_BOUNDARY(I, B)| will mirror a b-pixel boundary of I
% and return the padded image I_PADDED. The mirrored boundary will start
% with the last pixel of the image, i.e. the border pixels of the image
% I will de duplicated
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
% $Id: mirror_boundary.m 240 2011-05-30 16:24:20Z uschmidt $

function I_padded = mirror_boundary(I, b)
  I_padded = zeros(size(I)+2*b);
  [rows, cols] = size(I_padded);
  rows_middle = 1+b:rows-b;  cols_middle = 1+b:cols-b;
  rows_start = 1:b;          cols_start = 1:b;
  rows_end = rows-b+1:rows;  cols_end = cols-b+1:cols;
  
  I_padded(rows_middle, cols_middle) = I;
  
  I_padded(rows_start,  cols_middle) = flipud(I(1:b, :));         % top
  I_padded(rows_end,    cols_middle) = flipud(I(end-b+1:end, :)); % bottom
  I_padded(rows_middle, cols_start)  = fliplr(I(:, 1:b));         % left
  I_padded(rows_middle, cols_end)    = fliplr(I(:, end-b+1:end)); % right
  
  % corners
  I_padded(rows_start,  cols_start)  = flipud(fliplr(I(1:b, 1:b)));                 % top left
  I_padded(rows_start,  cols_end)    = flipud(fliplr(I(1:b, end-b+1:end)));         % top right
  I_padded(rows_end,    cols_start)  = flipud(fliplr(I(end-b+1:end, 1:b)));         % bottom left
  I_padded(rows_end,    cols_end)    = flipud(fliplr(I(end-b+1:end, end-b+1:end))); % bottom right
end
