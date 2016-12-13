%% DENSITY - Abstract base class for modeling probability density
% Any derived class needs to supply the following methods:
%
% * EVAL:   Method to compute the probability at one or more points.
% * SAMPLE: Draw one or more i.i.d. samples from the density
%
% Data convention: 
% The class supports multivariate probability densities defined over row
% vectors. Multiple points or samples are stored in a matrix of size d x n,
% where d is the number of dimensions of each point or sample, and n is the
% number of distinct points or samples.
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
%  Contact: sroth@cs.tu-darmstadt.de
% 
% Project page:  http://www.gris.tu-darmstadt.de/research/visinf/software/index.en.htm

% Copyright 2009-2011 TU Darmstadt, Darmstadt, Germany.
% $Id: density.m 240 2011-05-30 16:24:20Z uschmidt $

classdef density < pml.distributions.distribution
  %% Abstract methods
  methods (Abstract)
    d = eval(this, x) 
    %EVAL - Evaluate, i.e. compute, the probability density
    %
    %   P = EVAL(THIS, X) computes the probability density P of the density
    %   THIS at one or more points X. X is either a row vector or a matrix
    %   of row vectors (in case of multiple points).
  end

  methods
    function E = energy(this, x)
    %ENERGY - Evaluate, i.e. compute, the energy
    %
    %   E = ENERGY(THIS, X) computes the energy E (negative log
    %   density) of the distribution THIS at one or more points X. The
    %   normalization term (partition function) is ignored in this
    %   computation. X is either a row vector or a matrix of row vectors
    %   (in case of multiple points).  
    %   This method is only helpful if numerical accuracy is improved
    %   (especially in exponential family distributions), or if
    %   computation can be saved by ignoring the normalization term. By
    %   default it calls UNNORM and returns its negative log.
    
      E = -log(this.unnorm(x));
    end
    
    function g = log_grad_x(this, x)
    %LOG_GRAD_X - Compute the gradient of the log density 
    %
    %   G = LOG_GRAD_X(THIS, X) computes the gradient of the log of the
    %   probability density THIS with respect to the random variables at one
    %   or more points X. X is either a row vector or a matrix of row vectors
    %   (in case of multiple points). The gradient is computed for each point
    %   in X and the returned gradient G has the same size as X. By default
    %   this method is not implemented.
      
      error('Function not implemented!');    
    end
    
    function s = sample(this, nsamples)
    %SAMPLE - Draw i.i.d. samples from the probability density
    %
    %   S = SAMPLE(THIS[, NSAMPLES]) draws i.i.d. samples (default 1
    %   sample) from the probability density THIS and returns them as a
    %   matrix S. X is either a row vector (in case of 1 sample) or a
    %   matrix of row vectors (in case of multiple samples). The optional
    %   argument NSAMPLES specifies the number of i.i.d. samples to
    %   draw. By default this method is not implemented.
      
      error('Function not implemented!');    
    end
  end
end