%% GEN_SAMPLES - Draw samples from the MRF
% |X = GEN_SAMPLES(MRF, IMDIMS, SAMPLES_INIT[, CONDITIONAL_SAMPLING, DOPLOT])|
% will draw as many samples from MRF as there are image patches in SAMPLES_INIT.
% IMDIMS determines the size of the generated samples and must be consistent with SAMPLES_INIT. 
% CONDITIONAL_SAMPLING defaults to true and causes conditional sampling of the MRF prior,
% in order to avoid extreme values at the sample boundary.
% Set DOPLOT to true if you want to see the sampling progress (defaults to false).
% 
% This file is part of the implementation as described in the paper:
% 
%  Uwe Schmidt, Qi Gao, Stefan Roth.
%  A Generative Perspective on MRFs in Low-Level Vision.
%  IEEE Conference on Computer Vision and Pattern Recognition (CVPR'10), San Francisco, USA, June 2010.
% 
% Please cite the paper if you are using this code in your work.
% 
% The code may be used free of charge for non-commercial and
% educational purposes, the only requirement is that this text is
% preserved within the derivative work. For any other purpose you
% must contact the authors for permission. This code may not be
% redistributed without permission from the authors.
%
%  Author:  Uwe Schmidt, Department of Computer Science, TU Darmstadt
%  Contact: mail@uweschmidt.org
% 
% Project page:  http://www.gris.tu-darmstadt.de/research/visinf/software/index.en.htm

% Copyright 2009-2010 TU Darmstadt, Darmstadt, Germany.
% $Id: gen_samples.m 232 2010-07-08 13:18:35Z uschmidt $

function x = gen_samples(mrf, imdims, samples_init, conditional_sampling, doplot)
  
  if nargin < 4, conditional_sampling = true; end
  if nargin < 5, doplot = false; end
  
  if ~isa(mrf, 'pml.distributions.gsm_foe'), error('MRF unsuitable'), end
  
  mrf.update_filter_matrices = true;
  mrf.conv_method = 'valid';
  mrf.imdims = imdims;
  mrf.conditional_sampling = conditional_sampling;
  max_filter_size = mrf.filter_size;
  if conditional_sampling
    mr = max_filter_size(1) - 1; mc = max_filter_size(2) - 1;
  else
    mr = 0; mc = 0;
  end
  
  % indices of "active" pixels (=> all if not conditional sampling)
  [rs, cs] = ndgrid(1+mr:mrf.imdims(1)-mr, 1+mc:mrf.imdims(2)-mc);
  ind_act = sub2ind(mrf.imdims, rs(:), cs(:));
  imdims_int = imdims - 2*[mr,mc];
  % indices of interior pixels (=> all except underconstrained boundary)
  mr = max_filter_size(1) - 1; mc = max_filter_size(2) - 1;
  [rs, cs] = ndgrid(1+mr:mrf.imdims(1)-mr, 1+mc:mrf.imdims(2)-mc);
  ind_int = sub2ind(mrf.imdims, rs(:), cs(:));
  npixels = prod(mrf.imdims);
  
  if conditional_sampling, nsamplers = 1; else
    % number of parallel samplers which are used to assess convergence
    nsamplers = 1;
  end
  
  nsamples = size(samples_init,2);
  
  % number of estimands to be used
  nestimands = 1;
  
  % init samples with natural image patches
  x = samples_init;
  % add noise to interior of patches
  % x(ind_act,:) = x0(ind_act,:) + 25*randn(length(ind_act), nsamples);

  if doplot, figure(1), clf, colormap(gray(256)), end
  maxiters = 500;
  
  for start_idx = 1:nsamplers:nsamples
    
    % indices of samples to process
    idx = start_idx:min(nsamples, start_idx-1+nsamplers);
    
    % sample median-filtered and noisy image (with natural image boundary in case of conditional sampling)
    % to assess convergence
    c = [x(:,idx), repmat(x(:,start_idx), 1, 2)];
    c(ind_act,end-1) = 255*reshape(medfilt2(reshape(c(ind_act,end-1), imdims_int) / 255), numel(ind_act), 1);
    c(ind_act,end) = c(ind_act,end) + 15 * randn(size(c(ind_act,end)));
    c0 = c;
    
    estimands = zeros(maxiters, size(c,2), nestimands);
    R_hat = zeros(maxiters, nestimands);
    
    iter = 1;
    while true
      for i = 1:size(c,2)
        z = mrf.sample_z(c(:,i));
        c(:,i) = mrf.sample_x(z, c(:,i));
        estimands(iter,i,1) = mrf.energy(c(:,i));
      end
      
      % estimated potential scale reduction R_hat, discarding first half of iterations
      R_hat(iter,:) = pml.support.epsr(estimands(ceil(iter/2):iter,:,:));
      
      fprintf('\rSamples %03d-%03d/%03d, Iteration %03d, R_hat = %5.2f', idx(1), idx(end), nsamples, iter, R_hat(iter,1))
      if doplot
        subplot(3,1,1), plot(1:iter, R_hat(1:iter,:)), title 'R\_hat (estimated potential scale reduction)'
        subplot(3,1,2), plot(1:iter, estimands(1:iter,:,1)), title 'Energies of burn-in samples'
        subplot(3,1,3), imagesc(pml.support.assemble_images([c0, c], mrf.imdims, 2*size(c,2))), axis image on
        drawnow
      end
      
      % stop at convergence, do at least 20 iterations
      if iter > 20 && all(R_hat(iter,:) < 1.1), break; end
      
      if iter > maxiters
        warning('Exceeded maximum number of iterations without reaching convergence.');
        break;
      end
      
      iter = iter + 1;
    end
    fprintf('\n')
    % store samples
    x(:,idx) = c(:,1:length(idx));
  end
end