%% SLE_SPD_SOLVER - Returns function to solve a system of linear equations (SLE) with spd matrix
% |FUN = SLE_SPD_SOLVER(A)| returns function FUN(b) that solves the linear equation system Ax = b;
% A must be a symmetric, positive definite (spd) matrix.
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

% Copyright 2011 TU Darmstadt, Darmstadt, Germany.
% $Id: sle_spd_solver.m 240 2011-05-30 16:24:20Z uschmidt $

function fun = sle_spd_solver(A)
  [L p s] = chol(A, 'lower', 'vector');
  assert(p == 0, 'Matrix is not positive definite.');
  function x = solve_sle(b)
    x = zeros(size(b));
    x(s) = L' \ (L \ b(s));
  end
  fun = @solve_sle;
end