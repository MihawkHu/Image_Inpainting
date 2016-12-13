%% SAMPLE_Z - Samples from the distribution over scales Z, given image X
% |Z = SAMPLE_Z(THIS, X)| draws one sample from every discrete distribution
% given by Z_DISTRIBUTION. Z is a cell array of length nfilters, where the
% ith entry contains a [1 x ncliques] vector, where ncliques is the number of
% cliques under filter i. Z contains actual scales, not scales indices.
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
% $Id: sample_z.m 240 2011-05-30 16:24:20Z uschmidt $

function z = sample_z(this, x)
  
  z = cell(1, this.nfilters);
  pz = this.z_distribution(x);
  
  % sample scales
  for i = 1:this.nfilters
    % z{i} = sample_discrete(pz{i});
    z{i} = this.experts{min(i,this.nexperts)}.scales(sample_discrete(pz{i}));
  end
  
end


% draw a single sample from each discrete distribution (normalized weights in columns)
% domain start & stride is assumed to be 1
function xs = sample_discrete(pmfs)
  [nweights, ndists] = size(pmfs);
  % create cdfs
  cdfs = cumsum(pmfs, 1);
  % uniform sample for each distribution
  ys = rand(1, ndists);
  % subtract uniform sample and set 1s where difference >= 0
  inds = bsxfun(@minus, cdfs, ys) >= 0;
  % multiply each column with weight indices
  inds = bsxfun(@times, inds, (1:nweights)');
  % set 0s to NaN
  inds(inds == 0) = NaN;
  % min. weight index > 0 (NaNs are ignored by 'min')
  xs = min(inds, [], 1);
end