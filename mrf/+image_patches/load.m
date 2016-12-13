%% LOAD - Load/crop image patches from folder of natural images
% |DATA = LOAD(STR, IMDIMS, PATCHES_PER_IMAGE)| will crop PATCHES_PER_IMAGE
% many patches of size IMDIMS from each image in the subfolder STR of the script-directory.
% The image patches are returned in the matrix DATA where each patch is stored
% as a column. Hence, SIZE(DATA) = PROD(IMDIMS) x (#images * PATCHES_PER_IMAGE).
% 
%  Author:  Uwe Schmidt, Department of Computer Science, TU Darmstadt
%  Contact: mail@uweschmidt.org

% Copyright 2010 TU Darmstadt, Darmstadt, Germany.
% $Id: load.m 234 2010-07-09 13:19:26Z uschmidt $

function data = load(str, imdims, patches_per_image)
  message = 'You can download suitable natural images from the Berkeley Segmentation Dataset (BSDS) at http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/. Put all BSDS training images in the folder ''+image_patches/training'' and place the BSDS test images in ''+image_patches/evaluation''.';
  file_dir = fileparts(mfilename('fullpath'));
  img_list = dir(sprintf('%s/%s', file_dir, str));
  if length(img_list) == 0
    error('Folder "%s/%s" doesn''t exist.\n%s', file_dir, str, message)
  end
  c = 1;
  % guess for data size
  data = zeros(prod(imdims), length(img_list)*patches_per_image);
  
  for i = 1:length(img_list)
    try
      img = double(imread(sprintf('%s/%s/%s', file_dir, str, img_list(i).name)));
      if size(img,3) == 3, img = 255*rgb2gray(img/255); end
      data(:,c:c-1+patches_per_image) = pml.support.random_patches({img}, imdims, patches_per_image);
      c = c + patches_per_image;
    catch me
      % ignore
    end
  end
  
  if c > 1
    data = data(:,1:c-1);
  else
    error('Folder "%s/%s" doesn''t contain any images.\n%s', file_dir, str, message)
  end
  
end