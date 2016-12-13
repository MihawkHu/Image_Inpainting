function M = convmtxn(F, sz, valid)
%CONVMTXN   N-D convolution matrix
%   M = CONVMTXN(F, SZ[, VALID]) returns the convolution matrix for the
%   matrix F.  SZ gives the size of the array that the convolution should
%   be applied to.  The returned matrix M is sparse.
%   If X is of size SZ, then reshape(M * X(:), SZ + size(F) - 1) is the
%   same as convn(X, F).
%   The optional parameter VALID controls which rows of M (corresponding
%   to the pixels of X) should be set to zero to suppress the convolution
%   result.  VALID must have size SZ+size(F)-1.
%  
%   See also CONVN, CONVMTX2, MAKE_CONVN_MTX.
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

  
  ndims  = length(sz);
  blksz  = prod(size(F));
  nblks  = prod(sz);
  nelems = blksz * nblks;

  
  % Build index array for all possible image positions
  tmp = zeros(size(F) + sz - 1);
  sub = cell(1, ndims);
  for d = 1:ndims
    sub{d} = 1:sz(d);
  end
  tmp(sub{:}) = 1;  
  imgpos = find(tmp(:));

  % Build index array for all possible filter positions
  tmp = zeros(size(F) + sz - 1);
  for d = 1:ndims
    sub{d} = 1:size(F, d);
  end
  tmp(sub{:}) = 1;  
  fltpos = find(tmp(:));
  
  
  % The loop code below is replaced with the vectorized version below  
% $$$   rows = zeros(nelems, 1);
% $$$   cols = zeros(nelems, 1);
% $$$   vals = zeros(nelems, 1);
% $$$   
% $$$   for i = 1:prod(sz)
% $$$     % For every possible image position (cols) insert filter values
% $$$     % into the appropriate output position (rows).
% $$$   
% $$$     j = 1 + (i-1) * blksz;
% $$$     k = i * blksz;
% $$$     rows(j:k) = imgpos(i) + fltpos - 1;
% $$$     cols(j:k) = i;
% $$$     vals(j:k) = F(:);
% $$$   end

  % Vectorized version of loop code above.
  rows = reshape(repmat(imgpos', blksz, 1), nelems, 1) + ...
         repmat(fltpos - 1, nblks, 1);
  cols = reshape(repmat(1:nblks, blksz, 1), nelems, 1);
  vals = repmat(F(:), nblks, 1);

  % Pick out valid rows
  if (nargin > 2)
    valid_idx = valid(rows);

    rows = rows(valid_idx);
    cols = cols(valid_idx);
    vals = vals(valid_idx);
  end
  
  % Build sparse output array
  M = sparse(rows, cols, vals, prod(size(F) + sz - 1), nblks);
