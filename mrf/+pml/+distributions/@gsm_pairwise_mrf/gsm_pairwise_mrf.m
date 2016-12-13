%% GSM_PAIRWISE_MRF - GSM-based pairwise Markov random field
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
% $Id: gsm_pairwise_mrf.m 240 2011-05-30 16:24:20Z uschmidt $

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