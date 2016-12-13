% The code and models in this package demonstrate learning, evaluation, and image restoration with the generic MRF model described in the paper:
% 
%  Uwe Schmidt, Qi Gao, Stefan Roth.
%  A Generative Perspective on MRFs in Low-Level Vision.
%  IEEE Conference on Computer Vision and Pattern Recognition (CVPR'10), San Francisco, USA, June 2010.
% 
% Please cite the paper if you are using this code in your work.
% 
% 
% Contents of evaluation:
% 
%   demo_evaluation                - Demonstrate model evaluation by comparing filter marginals of MRF samples and natural images
%   gen_samples                    - Draw samples from the MRF
%   marginal_stats                 - Compare filter marginals of natural image patches and MRF samples
% 
% Contents of restoration:
% 
%   demo_denoising                 - Demonstrate MMSE-based denoising
%   demo_inpainting                - Demonstrate MMSE-based inpainting
%   denoise_mmse                   - Image denoising by approximating the MMSE estimate with samples
%   inpaint_mmse                   - Image inpainting by approximating the MMSE estimate with samples
% 
% Contents of learning:
% 
%   demo_learning                  - Demonstrate MRF learning (useful initialization and learning parameters; visualization of learning progress)
% 
% Contents of +learned_models:
% 
%   cvpr_3x3_foe                   - Our learned 3x3 FoE as described in the paper
%   cvpr_pw_mrf                    - Our learned pairwise MRF as described in the paper
% 
% Contents of +image_patches:
% 
%   load                           - Load/crop image patches from folder of natural images
% 
% Contents of +pml:
% 
%   Core implementation of our models and auxiliary functions

% Copyright 2011 TU Darmstadt, Darmstadt, Germany.
% $Id: Contents.m 241 2011-06-17 11:54:12Z uschmidt $