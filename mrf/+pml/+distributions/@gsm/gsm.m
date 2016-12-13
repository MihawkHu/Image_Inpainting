%% GSM - Uni- or multivariate GSM probability density
% Data convention: 
% The class supports multivariate densities defined over row
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
%           Uwe Schmidt, Department of Computer Science, TU Darmstadt
%  Contact: sroth@cs.tu-darmstadt.de, uwe.schmidt@gris.tu-darmstadt.de
% 
% Project page:  http://www.gris.tu-darmstadt.de/research/visinf/software/index.en.htm

% Copyright 2009-2011 TU Darmstadt, Darmstadt, Germany.
% $Id: gsm.m 240 2011-05-30 16:24:20Z uschmidt $

classdef gsm < pml.distributions.density
  
  %% Properties
  properties
    mu = 0;        % Mean (vector), shared by all mixture components
    precision = 1; % (Shared) precision value or matrix, or matrix for each mixture component in a cell array
    scales = 1;    % Scales (vector)
    weights = 1;   % Weights for each mixture component (automatically normalized to 1)
  end
  
  %% Convenience properties
  properties (Dependent)
    ndims;    % Number of dimensions
    mean;     % Mean (alias for mu) for the GSM density (can be set)
    nscales;  % Number of (scale) mixture components
  end
  
  %% Main methods
  methods
    function this = gsm(varargin)
    %GSM - Constructs a GSM probability density.
    % 
    %  OBJ = GSM(NDIMS, NSCALES) Constructs a GSM probability density with
    %  specified number of dimensions NDIMS and number of scale mixture
    %  components NSCALES.
    %
    % OBJ = GSM(OTHER)
    %   Constructs a GSM probability density copying all relevant data from
    %   OTHER.

      error(nargchk(0, 2, nargin));

      switch (nargin)
        case 1
          if isa(varargin{1}, 'pml.distributions.gsm')
            this = varargin{1};
          else
            error('Incompatible argument');
          end

        case 2
          ndims   = varargin{1};
          nscales = varargin{2};

          this.mu         = zeros(ndims, 1);
          this.precision  = eye(ndims);
          this.scales     = 1:nscales;
          this.weights    = ones(nscales, 1) / nscales;
      end
    end

    %% Accessor functions
    function this = set.weights(this, w)
    %SET.WEIGHTS - Set the weights and normalize.
      if (numel(w) ~= this.nscales)
        error('The number of weights must match the number of scale mixtures.');
      end
      % Make sure weights are a row vector
      this.weights = w(:) ./ sum(w(:));
    end
    
    function ndims = get.ndims(this)
    %GET.NDIMS - Get the number of dimensions.
      ndims = length(this.mu);
    end
    
    function mu = get.mean(this)
    %GET.MEAN - Accessor function for the mean alias.
      mu = this.mu;
    end
    
    function this = set.mean(this, mu)
    %SET.MEAN - Accessor function for the mean alias.
      this.mu = mu;
    end
    
    function nscales = get.nscales(this)
    %GET.NSCALES - Get the number of scale mixtures.
      nscales = length(this.scales);
    end
    
    %% Actual methods
    p    = eval(this, x)
    gx   = log_grad_x(this, x)
    gw   = log_grad_weights(this, x)
    s    = sample(this, nsamples)
    hc   = z_distribution(this, x)
    this = em(this, x, niters)
  end
end