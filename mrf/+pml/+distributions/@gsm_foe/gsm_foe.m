%% GSM_FOE - GSM-based Field of Experts
% See help for the base class density.
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
% $Id: gsm_foe.m 240 2011-05-30 16:24:20Z uschmidt $

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