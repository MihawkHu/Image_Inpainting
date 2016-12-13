%% DEMO_LEARNING - Demonstrate MRF learning (useful initialization and learning parameters; visualization of learning progress)
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
% $Id: demo_learning.m 234 2010-07-09 13:19:26Z uschmidt $

function demo_learning
  [prev_dir, base_dir] = adjust_path;
  
  % load suitable training data
  imdims = [50 50]; patches_per_image = 10;
  data = image_patches.load('training', imdims, patches_per_image);
  
  warning('The learning parameters use here are just rough guidelines. Learning pairwise MRFs is fairly robust towards the choice of parameters; learning high-order MRFs is somewhat dependent on the parameter choice on the other hand. You may thus have to tune the parameters for your data. Press any key to continue...'); pause
  
  %%==================================     options for stochastic gradient descent
  options = struct;
  options.MaxBatches      = 100;
  options.MinibatchSize   = 20;
  options.LearningRate    = 0.5;
  options.ConvergenceCheck = 0;
  options.LearningRateFactor = @(batch,minibatch) 1;
  options.LatestGradientWeight = 0.1;
  
  %%===========================================================     initialize mrf
  choice = input('(1) for pairwise MRF, or (2) for 3x3 FoE with 8 filters: ');
  if choice == 2, mrf = init_foe(imdims);          else
                  mrf = init_pw_mrf(imdims, data); end
  
  % learn mrf with 1-step contrastive divergence
  tic, [mrf, learning_report] = mrf.cd(data, 1, options); toc
  
  display_report(mrf, learning_report);
  
  adjust_path(prev_dir, base_dir);
end

function [prev_dir, base_dir] = adjust_path(prev_dir, base_dir)
  if nargin == 2
    % restore working directory
    % restoring the path sometimes confuses MATLAB when running the code again ("clear classes" helps)
    cd(prev_dir); % rmpath(base_dir); 
  else
    % save working directory and go to correct directory
    prev_dir = pwd; file_dir = fileparts(mfilename('fullpath')); cd(file_dir);
    last = @(v) v(end); base_dir = file_dir(1:last(strfind(file_dir, filesep))-1);
    % add base directory to path
    addpath(base_dir);
  end
end

function mrf = init_foe(imdims)
  mrf = pml.distributions.gsm_foe;
  mrf.conditional_sampling = true;
  mrf.imdims = imdims;
  mrf.experts{1}.precision = 1 / 500;
  mrf.experts{1}.scales = exp([-9,-7,-5:5,7,9]);
  mrf.experts{1}.weights = ones(mrf.experts{1}.nscales, 1);
  nexperts = 8; filter_size = [3 3];
  J = randn(prod(filter_size), nexperts);
  mrf = mrf.set_filter(eye(prod(filter_size)), J, filter_size);
  mrf.experts = repmat({mrf.experts{1}}, 1, nexperts);
end

function mrf = init_pw_mrf(imdims, data)
  mrf = pml.distributions.gsm_pairwise_mrf;
  mrf.conditional_sampling = true;
  mrf.imdims = imdims;
  mrf = mrf.fit_precision(data);
  mrf.experts{1}.scales = exp([-9,-7,-5:5,7,9]);
  mrf.experts{1}.weights = ones(mrf.experts{1}.nscales, 1);
end


function display_report(mrf, report)
  
  fhandle = 1;
  is_pairwise = isa(mrf, 'pml.distributions.gsm_pairwise_mrf');
  
  if is_pairwise, colormap(lines(mrf.nexperts)); else colormap(jet(mrf.nexperts)); end
  colors = colormap; colormap jet
  
  % generate weight indices for each expert
  ntotalweights = 0;
  weight_idx = cell(1, mrf.nexperts);
  for i = 1:mrf.nexperts
    weight_idx{i} = ntotalweights+1:ntotalweights+mrf.experts{i}.nscales;
    ntotalweights = weight_idx{i}(end);
  end
  
  %%=========================================================     weights progress
  figure(fhandle), clf, fhandle = fhandle + 1;
  func = @(weights) bsxfun(@rdivide, exp(weights), sum(exp(weights),2));
  for i = 1:mrf.nexperts
    weights = func(report.iter_x(weight_idx{i},:)');
    [nminibatches, nweights] = size(weights);
    fprintf('weights of expert %d: ', i), disp(mrf.experts{i}.weights')
    subplot(mrf.nexperts,1,i), plot(repmat((0:nminibatches-1)',1,nweights), weights)
    if i == 1, title 'Weight progress', end
    axis tight
  end
  
  if ~is_pairwise
    %%============================================     progress of filter parameters
    figure(fhandle), clf, fhandle = fhandle + 1;
    nweights = length(mrf.weights);
    nfilterparams = size(mrf.J_tilde,1);
    Fmeans = zeros(mrf.nfilters, size(report.iter_x,2));
    for i = 1:mrf.nfilters
      s = nweights+1+(i-1)*nfilterparams;
      F = report.iter_x(s:s-1+nfilterparams,:);
      F = mrf.A' * F;
      Fmeans(i,:) = mean(F);
      plot(F'), hold on
    end
    title 'Filter Progress'
    
    %%============================================================     final filters
    figure(fhandle), clf, colormap(gray(256)), fhandle = fhandle + 1;
    sqr = ceil(sqrt(mrf.nfilters));
    for i = 1:mrf.nfilters
      subplot(sqr,sqr,i), imagesc(mrf.filter(i)), axis image off
      colorbar, title(sprintf('Filter %d', i))
      ax = axis;
      line([ax(1) ax(3); ax(1) ax(4); ax(2) ax(4); ax(2) ax(3)], ...
           [ax(1) ax(4); ax(2) ax(4); ax(2) ax(3); ax(1) ax(3)], 'color', colors(i,:), 'linewidth', 5)
    end
    
  end
  
  %%===================================     experts: weight distribution and shape
  figure(fhandle), clf, fhandle = fhandle + 1;
  R = -255:.1:255;
  w = arrayfun(@(i) {mrf.experts{i}.weights}, 1:mrf.nexperts);
  l = arrayfun(@(i) {mrf.experts{i}.eval(R)'}, 1:mrf.nexperts);
  w = horzcat(w{:}); l = horzcat(l{:});
  if is_pairwise, colormap(lines(mrf.nexperts)); else colormap(jet(mrf.nexperts)); end
  
  subplot(2,1,1)
  if is_pairwise, bar(log(mrf.experts{1}.scales), w); else bar(log(mrf.experts{1}.scales), w, 2); end
  axis tight; ax = axis; axis([ax(1), ax(2), 0, 1]); title 'Weights'
  
  subplot(2,1,2)
  for i = 1:size(l,2)
    semilogy(R, l(:,i), 'color', colors(i,:), 'linewidth', 1); hold on
  end
  axis tight, title 'Experts'
  
end