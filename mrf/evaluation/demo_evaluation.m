%% DEMO_EVALUATION - Demonstrate model evaluation by comparing filter marginals of MRF samples and natural images
% 
%  Author:  Uwe Schmidt, Department of Computer Science, TU Darmstadt
%  Contact: mail@uweschmidt.org

% Copyright 2010 TU Darmstadt, Darmstadt, Germany.
% $Id: demo_evaluation.m 232 2010-07-08 13:18:35Z uschmidt $

function demo_evaluation
  [prev_dir, base_dir] = adjust_path;
  
  choice = input('(1) for pairwise MRF, or (2) for 3x3 FoE with 8 filters: ');
  if choice == 2, mrf = learned_models.cvpr_3x3_foe; else
                  mrf = learned_models.cvpr_pw_mrf;  end
  
  imdims = [50 50]; patches_per_image = 1;
  init_patches       = image_patches.load('training',   imdims, patches_per_image);
  validation_patches = image_patches.load('validation', imdims, patches_per_image+1);
  % make sure to use same amount of samples and natural image patches
  npatches = min(size(init_patches,2), size(validation_patches,2));
  init_patches = init_patches(:,randsample(size(init_patches,2), npatches));
  validation_patches = validation_patches(:,randsample(size(validation_patches,2), npatches));
  
  conditional_sampling = true;
  discard_boundary = 10; % #boundary-pixels to ignore for sample stats
  is_pw_mrf = isa(mrf, 'pml.distributions.pairwise_mrf');
  
  fprintf('Drawing many samples can take a long time, you might want to modify the code to save them for later use.\n')
  samples = gen_samples(mrf, imdims, init_patches, conditional_sampling, true);
  
  marginal_stats(validation_patches, samples, discard_boundary, mrf.filter, is_pw_mrf, ~is_pw_mrf);
  
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