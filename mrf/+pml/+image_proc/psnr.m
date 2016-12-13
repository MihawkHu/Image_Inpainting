function n = psnr(x, y)
%PSNR   Compute the peak signal to noise ratio between two images
%   PSNR(X, Y) computes the peak signal to noise ratio in decibels (dB)
%   between images X and Y.
%   The images are assumed to have 256 gray levels (0 .. 255).
%
%   Class support for inputs X and Y:
%        float:   double, single
%        integer: uint*, int*
%
%   See also SNR, IMMSE.
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
%   Author:  Stefan Roth, Department of Computer Science, TU Darmstadt
%   Contact: sroth@cs.tu-darmstadt.de
%   $Date: 2011-05-30 18:24:20 +0200 (Mon, 30 May 2011) $
%   $Revision: 240 $
% 
% Project page:  http://www.gris.tu-darmstadt.de/research/visinf/software/index.en.htm

% Copyright 2004-2007 Brown University, Providence, RI.
% Copyright 2007-2011 TU Darmstadt, Darmstadt, Germany.
% 
%                         All Rights Reserved
% 
% Permission to use, copy, modify, and distribute this software and its
% documentation for any purpose other than its incorporation into a
% commercial product is hereby granted without fee, provided that the
% above copyright notice appear in all copies and that both that
% copyright notice and this permission notice appear in supporting
% documentation, and that the name of Brown University not be used in
% advertising or publicity pertaining to distribution of the software
% without specific, written prior permission.
% 
% BROWN UNIVERSITY DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,
% INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY
% PARTICULAR PURPOSE.  IN NO EVENT SHALL BROWN UNIVERSITY BE LIABLE FOR
% ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
% WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
% ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
% OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
  
  diff = double(x) - double(y);
  n = 20 * log10(255 / std(diff(:)));

