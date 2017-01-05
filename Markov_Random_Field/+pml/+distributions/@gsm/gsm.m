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