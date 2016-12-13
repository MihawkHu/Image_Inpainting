%% DENOISE_MMSE - Image denoising by approximating the MMSE estimate with samples
% |IMG_DENOISED = DENOISE_MMSE(MRF, IMG_CLEAN, IMG_NOISY, SIGMA[, RB, DOPLOT])| will use
% MRF to denoise IMG_NOISY, assuming additive white Gaussian noise with standard deviation SIGMA.
% IMG_CLEAN is used to display the PSNR and SSIM during denoising.
% Set RB to true to use a Rao-Blackwellized MMSE estimator (not used in the paper, suggested by George Papandreou).
% Set DOPLOT to true if you want to see detailed plots during denoising.
% Optional arguments default to false.
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

% Copyright 2009-2011 TU Darmstadt, Darmstadt, Germany.
% $Id: denoise_mmse.m 238 2011-05-23 14:11:56Z uschmidt $


function img_denoised = denoise_mmse(mrf, img_clean, img_noisy, sigma, rb, doplot)
  
  if ~isa(mrf, 'pml.distributions.gsm_foe'), error('MRF unsuitable'), end
  % set border pixel
  if isa(mrf, 'pml.distributions.pairwise_mrf')
    border = 5;
  else
    border = 9;
  end
  
  mrf.update_filter_matrices = true;
  mrf.conv_method = 'valid';
  
  nsamplers = 4;
  max_iters = ceil(1000 / nsamplers); % use at most this many samples to obtain the denoised image
  max_burnin_iters = 100; % only the second half of these will be considered (i.e. throw away 50 initial samples at most)
  mean_mapd_threshold = 1; % convergence threshold
  
  % use Rao-Blackwellization?
  if ~exist('rb', 'var'), rb = false; end
  % plot while denoising?
  if ~exist('doplot', 'var'), doplot = false; end
  if doplot, figure(2), clf, figure(1), clf, colormap(gray(256)), end
  
  % original clean image (to compute PSNR & SSIM)
  I = img_clean;
  
  mrf.imdims = size(I) + 2*border;
  npixels = prod(mrf.imdims);
  mr = border;  mc = border;
  ridx = 1+mr:mrf.imdims(1)-mr; cidx = 1+mc:mrf.imdims(2)-mc;
  [rs, cs] = ndgrid(ridx, cidx);
  % indices of interior pixels
  ind_int = sub2ind(mrf.imdims, rs(:), cs(:));
  
  N = img_noisy;
  % add mirrored boundary (will be discarded in the end)
  N_padded = pml.support.mirror_boundary(N, border);
  
  % initialize sampler with noisy image
  x = repmat(N_padded(:), 1, nsamplers);
  
  % initialize 3 samplers with simple denoised images
  % (gaussian blur, wiener filter, median filter)
  x(:,1) = reshape(imfilter(N_padded, fspecial('gaussian', 3, sigma)), numel(N_padded), 1);
  x(:,2) = 255*reshape(wiener2(N_padded/255), numel(N_padded), 1);
  x(:,3) = 255*reshape(medfilt2(N_padded/255), numel(N_padded), 1);
  
  % to store denoised image under each sampler
  x_denoised = zeros(npixels, nsamplers);
  % to store the means for the Gaussian distributions (for Rao-Blackwellization)
  x_mu = zeros(npixels, nsamplers);
  
  psnrs = zeros(nsamplers+1, max_iters);
  ssims = zeros(nsamplers+1, max_iters);
  mapd = zeros(nsamplers, max_iters);
  time = zeros(1, max_iters);
  cpu_time = zeros(1, max_iters);
  
  % save all samples of the burn-in
  x_burnin = zeros(npixels, nsamplers, max_burnin_iters);
  % only one scalar estimand for the energy
  estimands = zeros(max_burnin_iters, nsamplers);
  R_hat = zeros(max_burnin_iters, 1);
  
  iter = 0; c = 0;
  converged = false; burnin = true;
  
  % start timer
  tic; tstart = cputime;
  
  % loop until convergence or max_iters depleted
  while true
    iter = iter + 1;
    
    % advance all samplers
    for i = 1:nsamplers
      z = mrf.sample_z(x(:,i));
      [x(:,i), x_mu(:,i)] = sample_x_denoising(mrf, z, N_padded(:), sigma);
      % save all samples in the burn-in phase
      if burnin
        x_burnin(:, i, iter) = x(:,i);
        estimands(iter, i) = mrf.energy(x(:,i));
      end
    end
    
    % check for convergence in burn-in phase
    if burnin
      % compute epsr, ignoring first half of samples
      R_hat(iter) = pml.support.epsr(estimands(ceil(iter/2):iter,:));
      fprintf('\rBurn-in %2d / %2d, R_hat = %.3f', iter, max_burnin_iters, R_hat(iter))
      if doplot
        subplot(211), plot(1:iter, R_hat(1:iter)), title 'R\_hat (estimated potential scale reduction)'
        subplot(212), plot(1:iter, estimands(1:iter,:)), title 'Energies of burn-in samples'
        drawnow
      end
      % discard at least 5 samples
      if (iter >= 10) && ((R_hat(iter) < 1.1) || (iter > max_burnin_iters))
        burnin = false;
        % use second half of samples to compute denoised images under each sampler
        x_denoised = mean(x_burnin(:, :, ceil(iter/2):iter), 3);
        % set counter c to enable running average
        c = length(ceil(iter/2):iter);
        psnrs(:,1:c) = nan; ssims(:,1:c) = nan; mapd(:,1:c) = nan;
        if iter > max_burnin_iters
          warning('Didn''t reach convergence in burn-in phase.')
        end
      end
    % after burn-in phase
    else
      if rb
        % Rao-Blackwellized estimator, which can lead to faster convergence.
        % This was kindly suggested to us by George Papandreou and has not been used in the paper.
        x_denoised = (x_mu + c * x_denoised) / (c + 1);  c = c + 1;
      else
        % update denoised images under each sampler (running average)
        x_denoised = (x + c * x_denoised) / (c + 1);  c = c + 1;
      end
      
      % overall denoised image as average of all samplers
      x_final = mean(x_denoised, 2);
      
      % compute PSNR, SSIM for the denoised images under each sampler,
      % and mean absolute pixel deviation of the individual denoised images from its average
      psnrs(1:nsamplers,c) = arrayfun(@(i) pml.image_proc.psnr( ...
                                reshape(x_denoised(ind_int,i), size(I)), I), 1:nsamplers);
      ssims(1:nsamplers,c) = arrayfun(@(i) pml.image_proc.ssim_index( ...
                                reshape(x_denoised(ind_int,i), size(I)), I), 1:nsamplers);
      mapd(:,c) = mean(abs(bsxfun(@minus, x_denoised, x_final)));
      
      O = reshape(x_final(ind_int), size(I));
      
      % compute PSNR, SSIM for the overall denoised image
      psnrs(end,c) = pml.image_proc.psnr(O, I);
      ssims(end,c) = pml.image_proc.ssim_index(O, I);
      time(c) = toc;
      cpu_time(c) = cputime-tstart;
      
      % display current status
      fprintf(['\rDenoising, sigma = %3d :: sample %3d / %3d, PSNR = %.2fdB, ' ...
               'SSIM = %.3f, avg. MAPD = %7.4f'], ...
              sigma, c, max_iters, psnrs(end,c), ssims(end,c), mean(mapd(:,c)))
      
      if doplot
        figure(1)
        subplot(1,3,1), imshow(uint8(I), 'InitialMagnification', 'fit'), title 'Original'
        subplot(1,3,2), imshow(uint8(N), 'InitialMagnification', 'fit')
        title(sprintf('Noisy (sigma = %d), PSNR = %.2fdB, SSIM = %.3f', sigma, ...
          pml.image_proc.psnr(N, I), pml.image_proc.ssim_index(N, I)));
        subplot(1,3,3), imshow(uint8(O), 'InitialMagnification', 'fit')
        title(sprintf('Denoised, PSNR = %.2fdB, SSIM = %.3f', ...
          psnrs(end,c), ssims(end,c)));
        % drawnow
        
        figure(2)
        subplot(4,1,1), plot(1:c, psnrs(:,1:c)'), ylabel 'PSNR'
        subplot(4,1,2), plot(1:c, ssims(:,1:c)'), ylabel 'SSIM'
        subplot(4,1,3), plot(1:c, mapd(:,1:c)', 1:c, mean(mapd(:,1:c))', '--k'),  ylabel 'MAPD'
        subplot(4,1,4), plot(time(1:c), psnrs(end,1:c), cpu_time(1:c), psnrs(end,1:c))
        xlabel 'Time (seconds)', ylabel 'PSNR (dB)'
        legend 'Time' 'CPU time' 'location' 'southeast'
        drawnow
      end
      
      % converged?
      converged = mean(mapd(:,c)) < mean_mapd_threshold;
      
      if c > max_iters
        warning('Maximum number of iterations exceeded.');
        break
      end
      
      if converged, break, end
    end
    
  end
  fprintf('\n')
  img_denoised = O;
end


% Sample from the posterior distribution p(x|z,y)
% The variable names W, Z, x, z, y, and sigma are used as described in the paper (and suppl. material)
function [x, x_mu] = sample_x_denoising(this, z, y, sigma)
  
  npixels = prod(this.imdims);
  nfilters = this.nfilters;
  nexperts = this.nexperts;
  
  N = 0;
  for i = 1:nfilters
    expert_precision = this.experts{min(i,nexperts)}.precision;
    z{i} = expert_precision * z{i}(:);
    N = N + length(z{i});
  end
  
  Wt = {this.filter_matrices{1:nfilters}, speye(npixels)};
  z = {z{:}, (this.epsilon + (1 / sigma^2)) * ones(npixels, 1)};
  N = N + npixels;
  
  Wt = vertcat(Wt{:});
  Z = spdiags(vertcat(z{:}), 0, N, N);
  
  r = randn(N, 1);
  W_sqrtZ_r = Wt' * sqrt(Z) * r;
  W_Z_Wt = Wt' * Z * Wt;
  
  solve_sle = pml.numerical.sle_spd_solver(W_Z_Wt);
  x_mu = solve_sle(y / sigma^2);
  x = x_mu + solve_sle(W_sqrtZ_r);
  
end