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