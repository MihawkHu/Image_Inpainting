classdef gsm_pairwise_mrf < pml.distributions.pairwise_mrf & pml.distributions.gsm_foe
  
  %% Main methods
  methods
    function mrf = gsm_pairwise_mrf(varargin)
      %GSM_PAIRWISE_MRF - Constructs a GSM-based pairwise Markov random field.
      % See help for the base class density.
      mrf = mrf@pml.distributions.pairwise_mrf(varargin{:});
      mrf = mrf@pml.distributions.gsm_foe(varargin{:});
    end
    
    %% Actual methods
    [this, report] = cd(this, x, niters, options)
    [this, dz] = fit_precision(this, x)
    grad = log_grad_log_weights(this, alpha, x, s, niters, neg)
  end
end