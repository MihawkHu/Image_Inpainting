classdef foe < pml.distributions.density
  %% Abstract constants
  properties (Constant, Abstract, Hidden)
    default_expert;
  end
  
  %% Properties
  properties
    epsilon = 1.e-8;
    experts;        % Experts (cell array)
    imdims = [3 3]; % Image dimensions (1x2 vector)
    conv_method;    % 'valid' or 'circular'
    update_filter_matrices = true;  % Should only be set to false if one doesn't sample from the FoE
    zeromean_filter = true;         % Always subtract mean from filter J
    conditional_sampling = false;   % Sample while keeping underconstrained boundary pixels fixed
  end
  
  %% Internal properties
  properties (SetAccess = 'private', Hidden)
    A_ = eye(9);
    filter_size_ = [3 3];
    J_tilde_ = [1 -1 zeros(1,7); 1 0 0 -1 zeros(1,5)]';
    J = [1 -1 zeros(1,7); 1 0 0 -1 zeros(1,5)]';
    filter_matrices_uptodate = false;
    conv2;
    imfilter;
    makemat;
    filter_matrices;
  end
  
  %% Convenience properties
  properties (Dependent)
    nexperts; % Number of experts
    nfilters; % Number of filters that are actually used
  end

  properties (Dependent, Hidden)
    J_tilde;
    A;
  end
  
  %% Main methods
  methods
    function this = foe(varargin)
    %FOE - Constructs a Field of Experts
    %
    %  OBJ = FOE([EXPERTS]) Constructs a Field of Experts
    %  using the optionally specified EXPERTS.
    %
    %  OBJ = FOE(OTHER)
    %  Constructs a Field of Experts by copying all
    %  relevant data from OTHER.
    
      error(nargchk(0, 1, nargin));
      
      switch (nargin)
        case 1
          if isa(varargin{1}, class(this))
            this = varargin{1};
          else
            if iscell(varargin{1})
              this.experts = varargin{1};
            else
              error('Incompatible argument');
            end            
          end
        otherwise
          this.experts = {this.default_expert};
      end
      
      % set default convolution method (also updates other properties)
      this.conv_method = 'valid';
    end
    
    %% Accessor functions
    function this = set.imdims(this, imdims)
    %SET.IMDIMS - Set image dimensions
      if any(this.imdims ~= imdims)
        if numel(imdims) ~= 2 || any(imdims(:) <= 0)
          error('IMDIMS must be a positive 2-element vector.');
        end
        if any(imdims < this.filter_size)
          error('IMDIMS < (max) FILTER_SIZE')
        end
        this.imdims = ceil(imdims(:)');
        % update filter matrices
        this = this.update;
      end
    end
    
    function this = set.experts(this, experts)
    %SET.EXPERTS - Set experts.
      experts = experts(:)';
      if ~iscell(experts) || length(experts) == 0
        error('EXPERTS must be in a cell array of length >= 1.');
      end
      for i = 1:length(experts)
        if ~isa(experts{i}, class(this.default_expert)) || experts{i}.ndims ~= 1
          error('All EXPERTS must be univariate densities of type ''%s''.', class(this.default_expert));
        end
      end
      % no update in constructor call
      constructor_call = length(this.experts) == 0;
      % set experts
      this.experts = experts;
      % update filter_matrices if necessary
      if ~constructor_call && this.update_filter_matrices && this.nfilters ~= length(this.filter_matrices)
        % update filter matrices
        this = this.update;
      end
    end
    
    function nexperts = get.nexperts(this)
    %GET.NEXPERTS - Get the number of experts.
      nexperts = length(this.experts);
    end

    function nfilters = get.nfilters(this)
    %GET.NFILTERS - Get the number of filters that are actually used.
      nfilters = max(2, length(this.filter));
    end
    
    function img_cliques = img_cliques(this, img)
    %GET.IMG_CLIQUES - The cliques of an image.
      switch this.conv_method
        case 'valid'
          img_cliques = flipud(im2col(img, this.filter_size_, 'sliding'));
        case 'circular'
          ncliques = prod(this.imdims);
          nfilterpixel = prod(this.filter_size_);
          img_cliques = zeros(nfilterpixel, ncliques);
          for m = 1:nfilterpixel
            f = zeros(this.filter_size_); f(m) = 1;
            pixel_m = this.conv2(img, f);
            img_cliques(m,:) = pixel_m(:);
          end
        otherwise
          error('Invalid convolution method: ''%s''', this.conv_method)
      end
    end
    
    function f = filter(this, i)
    %FILTER - Normal way to access filter
      if nargin > 1
        % just one filter
        f = reshape(this.J(:,i), this.filter_size_);
      else
        % all filters in a cell array
        f = arrayfun(@(i) {this.filter(i)}, 1:max(2, min(this.nexperts, size(this.J,2))));
      end
    end
    
    function this = set.conv_method(this, conv_method)
    %SET.CONV_METHOD - Set convolution method => set filter, create functions.
      switch conv_method
        case 'valid'
          this.conv2 = @(img, f) conv2(img, f, 'valid');
          this.imfilter = @(img, f) imfilter(img, f, 'full');
          this.makemat = @(this, f) pml.image_proc.make_convn_mat(f, this.imdims, 'valid');
        case 'circular'
          this.conv2 = @(img, f) imfilter(img, f, 'same', 'circular', 'conv');
          this.imfilter = @(img, f) imfilter(img, f, 'same', 'circular', 'corr');
          this.makemat = @(this, f) pml.image_proc.make_imfilter_mat(f, this.imdims, 'circular');
        otherwise
          error('Invalid value: ''%s''', conv_method)
      end
      this.conv_method = conv_method;
      % update filter matrices
      this = this.update;
    end
    
    function this = set_filter(this, A, J_tilde, filter_size)
    %SET_FILTER - Set all filter parameters at once
      [nfilterparams, nfilters] = size(J_tilde);
      if any(size(A) ~= [nfilterparams, prod(filter_size)])
        error('Sizes of A and J_TILDE don''t match.')
      end
      if nfilters < 2
        error('There must be at least 2 filter.')
      end
      this.filter_size_ = filter_size;
      this.A_ = A;
      this.J_tilde_ = J_tilde;
      this.J = this.A' * this.J_tilde;
      if this.zeromean_filter
        this.J = bsxfun(@minus, this.J, mean(this.J));
      end
      % update filter matrices
      this = this.update;
    end
    
    function A = get.A(this), A = this.A_; end
    function this = set.A(this, A)
    %SET.FILTER_SIZE - Set filter basis
      this = this.set_filter(A, this.J_tilde, this.filter_size_);
    end

    function J_tilde = get.J_tilde(this), J_tilde = this.J_tilde_; end
    function this = set.J_tilde(this, J_tilde)
    %SET.J_TILDE - Set filter
      this = this.set_filter(this.A, J_tilde, this.filter_size_);
    end
    
    function filter_size = filter_size(this, i)
      % numeric argument i => get size of filter i
      if nargin > 1 && isnumeric(i)
        filter_size = size(this.filter(i));
      % argument == 'min' => minimum of filter sizes, else maximum
      else
        if nargin > 1 && isstr(i) && strcmp(i, 'min'), op = @min; else, op = @max; end
        % maximum of with and height of all filter sizes
        f = arrayfun(@(i) {size(this.filter(i))}, 1:this.nfilters);
        filter_size = op(vertcat(f{:}));
      end
    end
    
    function this = update(this)
    %UPDATE - Update filter matrices
      if this.update_filter_matrices
        this.filter_matrices = this.create_filter_matrices;
        this.filter_matrices_uptodate = true;
      else
        this.filter_matrices_uptodate = false;
      end
    end
    
    function filter_matrices = get.filter_matrices(this)
      if ~this.filter_matrices_uptodate
        warning('FILTER_MATRICES might not be up-to-date.')
      end
      filter_matrices = this.filter_matrices;
    end
    
    %% Actual methods
    p = eval(this, x)
    p = unnorm(this, x)
    l = energy(this, x)
    g = log_grad_x(this, x)
  end
  
  methods (Access = 'private')
    function F = create_filter_matrices(this)
      nfilters = this.nfilters;
      F = cell(1,nfilters);
      for i = 1:nfilters
        F{i} = this.makemat(this, this.filter(i));
      end
    end
  end
end
