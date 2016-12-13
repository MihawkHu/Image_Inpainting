%% PAIRWISE_MRF - Abstract base class for modeling pairwise Markov random fields
% See help for the base class density.
%
% The MRF should either use 1 or 2 experts:
%
% * 1 expert:  The expert will be used for x- and y-derivatives.
% * 2 experts: The first expert is used for x-derivatives,
%              the second expert for y-derivatives.
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
% $Id: pairwise_mrf.m 240 2011-05-30 16:24:20Z uschmidt $

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