classdef discrete < pml.distributions.distribution
  %% Properties
  properties
    weights       = []; % Weights for each bin (automatically normalized to 1)
    domain_start  = 1;  % Start of the domain of the random variables (default 1)
    domain_stride = 1;  % Stride of the domain of the random variables (default 1)
  end
  
  %% Hidden properties
  properties (Hidden, Access = protected)
    cum_weights = []; % Cumulative weights
  end
  
  %% Convenience properties
  properties (Dependent)
    ndims         % Number of dimensions of the discrete random vector.
    mean          % Mean of the discrete distribution.
    covariance    % Covariance matrix of the discrete distribution.
    kurtosis      % Kurtosis of the  discrete distribution.
    entropy       % Entropy of the discrete distribution.
    domain_grid   % Grid of the domain of the discrete random vector.
  end
  
  %% Main methods
  methods
    function this = discrete(varargin)
      %DISCRETE - Construct a discrete distribution over integers
      % 
      % OBJ = DISCRETE([WEIGHTS])
      %   Constructs a discrete distribution with optionally specified 
      %   WEIGHTS.
      %
      % OBJ = DISCRETE(OTHER)
      %   Constructs a discrete distribution by copying all relevant data
      %   from OTHER.

      error(nargchk(0, 1, nargin));
  
      switch (nargin)
        case 1
          if isa(varargin{1}, 'pml.distributions.discrete')
            this = varargin{1};
          else
            this.weights = varargin{1};
          end

      end
    end
    
    %% Accessor functions
    function this = set.weights(this, w)
      %SET.WEIGHTS - Set the weights and normalize
      
      ndims = pml.support.nefdims(w);
      
      w = w ./ sum(w(:));
      % Make sure weights are a row vector when 1D
      if (ndims == 1)
        w = w(:);
      end
      
      this.weights     = w;
      this.cum_weights = cumsum(w(:)); % Precompute the cumulative sum for sampling
      
      if (ndims > length(this.domain_start))
        this.domain_start(end+1:ndims) = 1;
      end
      this.domain_start(ndims+1:end) = [];
      
      if (ndims > length(this.domain_stride))
        this.domain_stride(end+1:ndims) = 1;
      end
      this.domain_stride(ndims+1:end) = [];
    end
      
    function ndims = get.ndims(this)
      %GET.NDIMS - Number of dimensions of the random vector
      ndims = pml.support.nefdims(this.weights);
    end
    
    function grid = get.domain_grid(this)
      %GET.DOMAIN_GRID - Compute grid of all bins in the discrete domain
      ndims = this.ndims;
      
      if (ndims == 1)
        m = this.domain_start + (numel(this.weights) - 1) * this.domain_stride;
        grid{1} = this.domain_start:this.domain_stride:m;
      else
        % Set up domain grid
        ind = cell(1, ndims); grid = cell(1, ndims);
        for i = 1:ndims
          m = this.domain_start(i) + (size(this.weights, i) - 1) * this.domain_stride(i);
          ind{i} = this.domain_start(i):this.domain_stride(i):m;
        end
        
        [grid{:}] = ndgrid(ind{:});
      end
    end
    
    function mu = get.mean(this)
      %GET.MEAN - Compute the mean of a discrete distribution.
      
      ndims = this.ndims;
      grid  = this.domain_grid;
      
      mu = zeros(ndims, 1);
      for i = 1:ndims
        mu(i) = sum(grid{i}(:) .* this.weights(:));
      end
    end
    
    function sigma = get.covariance(this)
      %GET.COVARIANCE - Compute the covariance of a discrete distribution.
      
      ndims = this.ndims;
      grid  = this.domain_grid;
      
      mu = zeros(ndims, 1);
      for i = 1:ndims
        mu(i) = sum(grid{i}(:) .* this.weights(:));
      end
      
      sigma = zeros(ndims);
      for i = 1:ndims
        for j = 1:ndims
          sigma(i, j) = sum((grid{i}(:) - mu(i)) .* (grid{j}(:) - mu(j)) .* this.weights(:));
        end
      end
    end
    
    function k = get.kurtosis(this)
      %GET.KURTOSIS - Compute the kurtosis of a discrete distribution.
      
      ndims = this.ndims;
      grid  = this.domain_grid;
      
      mu = zeros(ndims, 1); k = zeros(ndims, 1);
      for i = 1:ndims
        mu(i) = sum(grid{i}(:) .* this.weights(:));
        k(i)  = sum((grid{i}(:) - mu(i)).^4 .* this.weights(:)) / ...
          sum((grid{i}(:) - mu(i)).^2 .* this.weights(:))^2;
      end
    end

    function e = get.entropy(this)
      %GET.ENTROPY - Compute the entropy of a discrete distribution.
      %   ENTROPY(THIS) computes the entropy of the discrete
      %   distribution THIS.
    
      valid = (this.weights > 0);
      
      e = -sum(this.weights(valid) .* log(this.weights(valid)));
    end
    
    %% Actual methods
    p = eval(this, x)
    s = sample(this, nsamples)
    d = kl_divergence(this, other)
    plot(this, varargin)
    semilogy(this, varargin)
    this = mle(this, X)
    
  end
end

