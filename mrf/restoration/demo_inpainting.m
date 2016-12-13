%% DEMO_INPAINTING - Demonstrate MMSE-based inpainting
% 
%  Author:  Uwe Schmidt, Department of Computer Science, TU Darmstadt
%  Contact: mail@uweschmidt.org

% Copyright 2010 TU Darmstadt, Darmstadt, Germany.
% $Id: demo_inpainting.m 230 2010-07-07 11:09:46Z uschmidt $

function demo_inpainting
  [prev_dir, base_dir] = adjust_path;
  
  choice = input('(1) for pairwise MRF, or (2) for 3x3 FoE with 8 filters: ');
  if choice == 2, mrf = learned_models.cvpr_3x3_foe; else
                  mrf = learned_models.cvpr_pw_mrf;  end
  
  img_names = {'3ch', 'new'};
  idx  = 1; % choose index of img_names
  img  = double(imread(sprintf('images/inpainting/%s_original.png', img_names{idx})));
  mask = logical(imread(sprintf('images/inpainting/%s_mask.png', img_names{idx})));
  sc = 1/3; % choose scale (1 => original size)
  img = imresize(img, sc, 'nearest'); mask = imresize(mask, sc, 'nearest');
  
  img_inpainted = inpaint_mmse(mrf, img, mask, false, true);
  
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