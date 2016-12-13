%% EPSR - "estimated potential scale reduction" (Gelman and Rubin).
% |R_HAT = EPSR(ESTIMANDS)| computes the EPSR for all k estimands, where
% ESTIMANDS(i,j,k) is the value of estimand k at iteration i, sampler j.
% A value of R_hat(k) below 1.1 can be considered to denote convergence
% of estimand k.
% 
% See Bayesian Data Analysis, p. 294-299 (Second Edition)
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
% $Id: epsr.m 240 2011-05-30 16:24:20Z uschmidt $

function [R_hat] = epsr(estimands)
  [niters, nsamplers, nestimands] = size(estimands);
  R_hat = zeros(nestimands, 1);

  for k = 1:nestimands
    mean_sampler = mean(estimands(:,:,k), 1);
    mean_overall = mean(mean_sampler);

    % between sequence variance
    B = (niters/(nsamplers-1)) * sum((mean_sampler - mean_overall).^2);
    % within sequence variance
    % W = (1/nsamplers) * sum(sum(bsxfun(@minus, estimands(:,:,k), mean_sampler).^2) / (niters-1))
    W = (1/nsamplers) * sum(var(estimands(:,:,k)));

    v_hat = ((niters-1)/niters)*W + (1/niters)*B;
    R_hat(k) = sqrt(v_hat / W);
  end
end