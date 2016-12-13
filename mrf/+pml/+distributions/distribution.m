%% DISTRIBUTION - Abstract base class for modeling probability distributions
% Any derived class needs to supply the following methods:
%
% * EVAL:   Method to compute the probability at one or more points.
% * SAMPLE: Draw one or more i.i.d. samples from the probability
%
% Data convention:
% The class supports multivariate probabilities defined over row
% vectors. Multiple points or samples are stored in a matrix of size
% d x n, where d is the number of dimensions of each point or sample,
% and n is the number of distinct points or samples.
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
% $Id: distribution.m 240 2011-05-30 16:24:20Z uschmidt $

classdef distribution
  %% Abstract methods
  methods (Abstract)
    p = eval(this, x)
    %EVAL - Evaluate, i.e. compute, the probability
    %
    %   P = EVAL(THIS, X) computes the probability P of the distribution
    %   THIS at one or more points X. X is either a row vector or a matrix
    %   of row vectors (in case of multiple points).
    
    
    s = sample(this, nsamples)
    %SAMPLE - Draw i.i.d. samples from the probability
    %
    %   S = SAMPLE(THIS[, NSAMPLES]) draws i.i.d. samples (default 1 sample)
    %   from the probability distribution THIS and returns them as a matrix
    %   S. X is either a row vector (in case of 1 sample) or a matrix of row
    %   vectors (in case of multiple samples). The optional argument
    %   NSAMPLES specifies the number of i.i.d. samples to draw.
  end

  %% Standard methods common to all distributions
  methods
    function p = unnorm(this, x)
    %UNNORM - Evaluate, i.e. compute, the unnormalized probability
    %
    %   P = UNNORM(THIS, X) computes the unnormalized probability P of the
    %   distribution THIS at one or more points X. X is either a row
    %   vector or a matrix of row vectors (in case of multiple points).
    %   This method is only helpful if computation can be saved by
    %   ignoring the normalization term. By default it calls EVAL.
      
      p = this.eval(x);
    end
    
    function lp = log(this, x)
    %LOG - Evaluate, i.e. compute, the log probability
    %
    %   LP = LOG(THIS, X) computes the log of the probability LP of the
    %   distribution THIS at one or more points X. X is either a row
    %   vector or a matrix of row vectors (in case of multiple points).
    %   This method is only helpful if numerical accuracy is improved
    %   (especially in exponential family distributions). By default it
    %   calls EVAL and returns its log.
      
      lp = log(this.eval(x));
    end
    
    function E = energy(this, x)
    %ENERGY - Evaluate, i.e. compute, the energy
    %
    %   E = ENERGY(THIS, X) computes the energy E (negative log
    %   probability) of the distribution THIS at one or more points X. The
    %   normalization term (partition function) is ignored in this
    %   computation. X is either a row vector or a matrix of row vectors
    %   (in case of multiple points).  
    %   This method is only helpful if numerical accuracy is improved
    %   (especially in exponential family distributions), or if
    %   computation can be saved by ignoring the normalization term. By
    %   default it calls UNNORM and returns its negative log.
      
      E = -log(this.unnorm(x));
    end
  end
end