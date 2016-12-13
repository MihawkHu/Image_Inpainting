%% Z_DISTRIBUTION - Distribution over scales Z, given image X
% |PZ = Z_DISTRIBUTION(THIS, X)| computes discrete distributions over scales Z
% for every clique under each filter. PZ is a cell array of length nfilters,
% where the ith entry contains a [nscales x ncliques] matrix, where nscales is
% the number of scales of filter i's expert, and ncliques is the number of
% cliques under filter i. Each column of the matrix contains normalized
% weights, where the weight in row j denotes the probability for the jth scale
% of the expert.
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
% $Id: z_distribution.m 240 2011-05-30 16:24:20Z uschmidt $

function pz = z_distribution(this, x)
  
  pz = cell(1, this.nfilters);
  
  % convert to 2D image
  img = reshape(x, this.imdims);
  
  for i = 1:this.nfilters
    % filter responses
    d_filter = this.conv2(img, this.filter(i));
    % filter's expert
    expert = this.experts{min(i,this.nexperts)};
    % discrete distributions over scales
    pz{i} = expert.z_distribution(d_filter(:)');
  end
  
end