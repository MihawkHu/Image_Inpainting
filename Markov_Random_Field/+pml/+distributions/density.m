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