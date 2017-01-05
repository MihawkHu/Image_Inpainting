classdef gsm_foe < pml.distributions.foe
  %% Constants
  properties (Constant, Hidden)
    default_expert = pml.distributions.gsm(1,1);
  end
  
  %% Convenience properties
  properties (Dependent)
    weights; % Concatenated weight vector of all experts (can be set)
  end
  
  %% Main methods
  methods
    function foe = gsm_foe(varargin)
      %GSM_FOE - Constructs a GSM-based FoE.
      % See help for the base class density.
      foe = foe@pml.distributions.foe(varargin{:});
    end
    
    %% Accessor functions
    function this = set.weights(this, weights)
    %SET.WEIGHTS - Set all expert weights at once
      for i = 1:this.nexperts
        nweights = this.experts{i}.nscales;
        % expert normalizes weights if necessary
        this.experts{i}.weights = weights(1:nweights);
        % remove "used" weights
        weights = weights(nweights+1:end);
      end
    end
    
    function weights = get.weights(this)
    %GET.WEIGHTS - Concatenate all expert weights
      weights = cellfun(@(expert) {expert.weights}, this.experts);
      weights = vertcat(weights{:});
    end
    
    %% Actual methods
    [this, report] = cd(this, x, niters, options)
    g = energy_grad_J_tilde(this, x)
    g = energy_grad_weights(this, x)
    g = log_grad_theta(this, theta, x, s, niters, neg)
    [x, q] = sample_x(this, z, x_cond)
    z = sample_z(this, x)
    pz = z_distribution(this, x)
  end
end