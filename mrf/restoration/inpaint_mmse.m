%% INPAINT_MMSE - Image inpainting by approximating the MMSE estimate with samples
% |IMG_INPAINTED = INPAINT_MMSE(MRF, IMG, MASK[, RB, DOPLOT])| will use MRF to inpaint
% the pixels of IMG that are specified by MASK (boolean matrix of same size as IMG).
% IMG can either be a gray-scale or RGB image.
% Set RB to true to use a Rao-Blackwellized MMSE estimator (not used in the paper, suggested by George Papandreou).
% Set DOPLOT to true if you want to see the inpainting progress.
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
% $Id: inpaint_mmse.m 238 2011-05-23 14:11:56Z uschmidt $

function img_inpainted = inpaint_mmse(mrf, img, mask, rb, doplot)
  
  if ~isa(mrf, 'pml.distributions.gsm_foe'), error('MRF unsuitable'), end
  
  mrf.update_filter_matrices = true;
  mrf.conv_method = 'valid';
  
  nsamplers = 3;
  max_iters = ceil(1000 / nsamplers); % use at most this many samples to obtain the inpainting image
  max_burnin_iters = 100; % only the second half of these will be considered (i.e. throw away 50 initial samples at max)
  mean_mapd_threshold = 1; % convergence threshold
  
  % use Rao-Blackwellization?
  if ~exist('rb', 'var'), rb = false; end
  % plot while inpainting?
  if ~exist('doplot', 'var'), doplot = false; end
  if doplot, figure(1), clf, end
  
  nlayers = size(img, 3);
  img_corrupted = img;
  
  % assuming RGB image, convert to YCbCr
  if nlayers == 3, img = 255 * rgb2ycbcr(double(img) / 255); end
  
  I = img;
  
  mrf.imdims = size(I(:,:,1));
  npixels = numel(I(:,:,1));
  ind_int = setdiff((1:npixels)' .* mask(:), 0);
  ind_ext = setdiff(1:npixels, ind_int);
  
  img_vector = reshape(I, npixels, nlayers);
  
  % init all samplers with the image
  x = repmat(img_vector, [1, 1, nsamplers]);
  
  % overwrite first 2 samplers with simple "inpainted" images (smooth & noisy)
  for j = 1:nlayers
    x(:,j,1) = reshape(255 * imfilter(I(:,:,j)/255, fspecial('gaussian', 10, 5)), npixels, 1);
    x(ind_ext,j,1) = img_vector(ind_ext,j);
    x(ind_int,j,2) = mean(img_vector(ind_ext,j)) + 10 * randn(numel(ind_int), 1);
  end
  
  % to store inpainted image under each sampler
  x_inpainted = zeros(npixels, nlayers, nsamplers);
  % to store the means for the Gaussian distributions (for Rao-Blackwellization)
  x_mu = zeros(npixels, nlayers, nsamplers);
  
  mapd = zeros(nsamplers, max_iters);
  
  % save all samples of the burn-in
  x_burnin = zeros(npixels, nlayers, nsamplers, max_burnin_iters);
  % only one scalar estimand for the energy
  estimands = zeros(max_burnin_iters, nsamplers, nlayers);
  R_hat = zeros(nlayers, max_burnin_iters);
  
  iter = 0; c = 0;
  converged = false; burnin = true;
      
  % loop until convergence or max_iters depleted
  while true
    iter = iter + 1;
    
    % advance all samplers
    for i = 1:nsamplers
      for j = 1:nlayers
        z = mrf.sample_z(x(:,j,i));
        [x(:,j,i), x_mu(:,j,i)] = sample_x_inpainting(mrf, z, mask(:), img_vector(:,j));
        % save all samples in the burn-in phase
        if burnin
          x_burnin(:,j,i,iter) = x(:,j,i);
          % average estimands in case of color image
          estimands(iter,i,j) = mrf.energy(x(:,j,i));
        end
      end
    end
    
    % check for convergence in burn-in phase
    if burnin
      % compute epsr, ignoring first half of samples
      R_hat(:, iter) = pml.support.epsr(estimands(ceil(iter/2):iter,:,:));
      fprintf('\rBurn-in %2d / %2d, R_hat = %.3f', iter, max_burnin_iters, max(R_hat(:,iter)))
      if doplot
        subplot(211), plot(1:iter, R_hat(:,1:iter)), title 'R\_hat (estimated potential scale reduction)'
        subplot(212), title 'Energies of burn-in samples'
        for j = 1:nlayers, plot(1:iter, estimands(1:iter,:,j)), hold on, end
        drawnow
      end
      % discard at least 5 samples
      if (iter >= 10) && (all(R_hat(:,iter) < 1.1) || (iter > max_burnin_iters))
        burnin = false;
        % use second half of samples to compute inpainted images under each sampler
        x_inpainted = mean(x_burnin(:, :, :, ceil(iter/2):iter), 4);
        % set counter c to enable running average
        c = length(ceil(iter/2):iter);
        mapd(:,1:c) = nan;
        if iter > max_burnin_iters
          warning('Didn''t reach convergence in burn-in phase.')
        end
      end
    % after burn-in phase
    else
      if rb
        % Rao-Blackwellized estimator, which can lead to faster convergence.
        % This was kindly suggested to us by George Papandreou and has not been used in the paper.
        x_inpainted = (x_mu + c * x_inpainted) / (c + 1);  c = c + 1;
      else
        % update inpainted images under each sampler (running average)
        x_inpainted = (x + c * x_inpainted) / (c + 1);  c = c + 1;
      end
      
      % overall inpainted image as average of all samplers
      x_final = mean(x_inpainted, 3);
      
      img_inpainted = reshape(x_final, [mrf.imdims, nlayers]);
      if nlayers == 3, img_inpainted = 255 * ycbcr2rgb(img_inpainted / 255); end
      
      % (averaged over layers) mean absolute pixel distance (considering only inpainted pixels)
      mapd(:,c) = mean(mean(abs(bsxfun(@minus, x_inpainted(ind_int,:,:), x_final(ind_int,:)))));
      
      % display current status
      fprintf('\rInpaining :: sample %3d / %3d, avg. MAPD = %7.4f', ...
              c, max_iters, mean(mapd(:,c)))
      
      if doplot
        subplot(221), imshow(uint8(img_corrupted), 'InitialMagnification', 'fit'), title 'Corrupted'
        subplot(222), imshow(uint8(img_inpainted), 'InitialMagnification', 'fit'), title 'Inpainted'
        subplot(223), imshow(mask, 'InitialMagnification', 'fit'), title 'Mask'
        subplot(224), plot(1:c, mapd(:,1:c)', 1:c, mean(mapd(:,1:c))', '--k'), title 'MAPD'
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
    
end

% Sample from the conditional distribution p(x_masked|x_fixed,z)
% The variable names W, Z, x, and z are used as described in the paper (and suppl. material)
function [x, x_mu] = sample_x_inpainting(this, z, mask, x_fixed)
  
  npixels = prod(this.imdims);
  nfilters = this.nfilters;
  nexperts = this.nexperts;
  
  N = 0;
  for i = 1:nfilters
    expert_precision = this.experts{min(i,nexperts)}.precision;
    z{i} = expert_precision * z{i}(:);
    N = N + length(z{i});
  end
  
  ind_int = setdiff((1:npixels)' .* mask, 0);
  ind_ext = setdiff(1:npixels, ind_int);
  npixels_int = length(ind_int);
  npixels_ext = length(ind_ext);
  
  x_ext = x_fixed(ind_ext);
  
  Wt_ext = cellfun(@(F) {F(:,ind_ext)}, this.filter_matrices(1:nfilters));
  Wt_int = cellfun(@(F) {F(:,ind_int)}, this.filter_matrices(1:nfilters));
  C = vertcat(Wt_int{:})' * (spdiags(vertcat(z{:}), 0, N, N) * vertcat(Wt_ext{:}));
  
  Wt = {Wt_int{:}, speye(npixels_int)};
  z = {z{:}, this.epsilon(ones(npixels_int, 1))};
  N = N + npixels_int;
  
  Wt = vertcat(Wt{:});
  Z = spdiags(vertcat(z{:}), 0, N, N);
  
  r = randn(N, 1);
  W_sqrtZ_r = Wt' * sqrt(Z) * r;
  W_Z_Wt = Wt' * Z * Wt;
  
  solve_sle = pml.numerical.sle_spd_solver(W_Z_Wt);
  x_int_mu = - solve_sle(C * x_ext);
  x_int = x_int_mu + solve_sle(W_sqrtZ_r);
  
  x = x_fixed; x_mu = x_fixed;
  x(ind_int) = x_int;
  x_mu(ind_int) = x_int_mu;
  
end