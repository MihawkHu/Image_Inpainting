classdef pairwise_mrf < pml.distributions.foe
  %% Constants
  properties (Constant, Hidden)
    % 2x1 filters for x- and y-derivatives
    filter_valid = {[1 -1], [1 -1]'};
    filter_circular = {[1 -1 0], [1 -1 0]'};
  end
  
  properties (SetAccess = 'private', Hidden)
    filter_current;
  end
  
  %% Main methods
  methods
    function foe = pairwise_mrf(varargin)
      %PAIRWISE_MRF - Constructs a pairwise MRF.
      % See help for the base class density.
      foe = foe@pml.distributions.foe(varargin{:});
    end
    
    function f = filter(this, i)
    %FILTER - Override superclass
      if nargin > 1
        % just one filter
        f = this.filter_current{i};
      else
        % all filters in a cell array
        f = this.filter_current;
      end
      
    end
    
    function this = update(this)
      % change filter before 'foe'-update is called
      switch this.conv_method
        case 'valid'
          this.filter_current = this.filter_valid;
        case 'circular'
          this.filter_current = this.filter_circular;
        otherwise
          error('Invalid value: ''%s''', this.conv_method)
      end
      % call update-method of superclass
      this = update@pml.distributions.foe(this);
    end
    
    function img_cliques = img_cliques(this, img)
      error('IMG_CLIQUES is not implemented for PAIRWISE_MRF.')
    end
    
    function this = set_filter(this, A, J_tilde, filter_size)
      error('PAIRWISE_MRF doesn''t use A, J_TILDE, and J.')
    end
    
  end
end