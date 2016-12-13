%% MARGINAL_STATS - Compare filter marginals of natural image patches and MRF samples
% |MARGINAL_STATS(PATCHES, SAMPLES, DISCARD_BOUNDARY, FILTERS, COMBINE_FR, UNITNORM)| compares
% the marginal distributions of FILTERS on natural image PATCHES and SAMPLES. It is assumed
% that PATCHES and SAMPLES are of equal size (npixels x npatches).
% DISCARD_BOUNDARY must be set to the number of boundary pixels that will be ignored.
% Optional arguments:
% - COMBINE_FR (default=false): If true, combined first derivative filters will be used (for pairw. MRF).
% - UNITNORM (default=false): Convert each filter to have norm 1 (set for high-order model)
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
% $Id: marginal_stats.m 236 2010-07-09 23:30:23Z uschmidt $

function marginal_stats(patches, samples, discard_boundary, filters, combine_fr, unitnorm)
  
  [npixels_samples, nsamples] = size(samples);
  [npixels_patches, npatches] = size(patches);
  
  if nargin < 6, unitnorm = false; end
  if nargin < 5, combine_fr = false; end
  
  % assume square patches
  imdims = repmat(sqrt(npixels_samples),1,2);
  
  if nsamples ~= npatches, error('Must be same number of natural image patches and MRF samples'), end
  if ~((prod(imdims) == npixels_patches) && (npixels_patches == npixels_samples))
    error('Image dimensions don''t match.')
  end
  
  figure(1), clf
  
  % linewidth for plots
  lw = 2;
  
  % bounds of filter responses to consider
  b = 255;
  
  % number of histogram bins
  nbins = 2*b+1;
  
  nfilters = length(filters);
  
  if unitnorm, filters = cellfun(@(f) {f/norm(f(:))}, filters); end
  
  patches_stats = filter_responses(imdims, patches, filters, b, nbins, discard_boundary);
  sample_stats = filter_responses(imdims, samples, filters, b, nbins, discard_boundary);
  
  % combine all filter responses
  if combine_fr
    all_patches_stats = cellfun(@(d) {d.weights(:)}, patches_stats);
    all_patches_stats = horzcat(all_patches_stats{:});
    patches_stats{1}.weights = sum(all_patches_stats, 2);
    patches_stats = {patches_stats{1}};
    
    all_sample_stats = cellfun(@(d) {d.weights(:)}, sample_stats);
    all_sample_stats = horzcat(all_sample_stats{:});
    sample_stats{1}.weights = sum(all_sample_stats, 2);
    sample_stats = {sample_stats{1}};
  end
  
  % compute marginal KL-divergence
  kld = arrayfun(@(i) patches_stats{i}.kl_divergence(sample_stats{i}), 1:length(sample_stats));
  
  if combine_fr, str = 'Filter (combined):  '; else str = 'Filter:             '; end
  fprintf(str),                      fprintf('%8d', 1:length(sample_stats))
  fprintf('\nKL-divergence:      '), fprintf('%8.3f', kld), fprintf('\n')
  
  if combine_fr
    % plot
    patches_stats{1}.semilogy('-', 'LineWidth', lw, 'Color', 'b')
    hold on
    sample_stats{1}.semilogy('--', 'LineWidth', lw, 'Color', 'r')

    title(sprintf('%d %dx%d patches, ignoring %d-pixel boundary, KLD = %.4f', ...
                  nsamples, imdims(1), imdims(2), discard_boundary, kld), 'FontWeight', 'bold')
    legend('Image patches', 'MRF samples')

    % adjust figure
    axis tight; ax = axis; axis([-b b ax(3) 1])
    % set(gca, 'LineWidth', lw)
    set(gca, 'FontSize', 12)
    set(gca, 'FontWeight', 'bold')
  else
    
    colormap(jet(nfilters)); colors = colormap;
    
    %%============================================================     Image patches
    minvals = inf(1, nfilters);
    for i = 1:nfilters
      minvals(i) = min(patches_stats{i}.weights(find(patches_stats{i}.weights(:))));
      semilogy(patches_stats{i}.domain_grid{1}, patches_stats{i}.weights,...
               '-', 'Color', colors(i,:), 'LineWidth', lw)
      hold on
    end
    lowest = min(minvals);
    axis([-b b lowest 1]); ax = axis;
    
    % set(gca, 'xtick', [-150,-75,0,75,150])
    set(gca, 'ytick', [1.e-5 1.e-3 1.e-1]);
    grid on
    title(sprintf('%d filters on %d %dx%d image patches, ignoring %d-pixel boundary', nfilters, size(patches, 2), imdims(1), imdims(2), discard_boundary));
    
    set(gca, 'FontSize', 12)
    set(gca, 'FontWeight', 'bold')
    
    %%==================================================================     Samples
    figure(2), clf, colormap(jet(nfilters));
    
    minvals = inf(1, nfilters);
    for i = 1:nfilters
      minvals(i) = min(sample_stats{i}.weights(find(sample_stats{i}.weights(:))));
      semilogy(sample_stats{i}.domain_grid{1}, sample_stats{i}.weights,...
               '-', 'Color', colors(i,:), 'LineWidth', lw)
      hold on
    end
    axis(ax)
    % set(gca, 'xtick', [-150,-75,0,75,150])
    set(gca, 'ytick', [1.e-5 1.e-3 1.e-1]);
    grid on
    title(sprintf('%d filters on %d %dx%d samples, ignoring %d-pixel boundary, KLD (min, avg, max) = (%.3f, %.3f, %.3f)', nfilters, nsamples, imdims(1), imdims(2), discard_boundary, min(kld), mean(kld), max(kld)));
    
    set(gca, 'FontSize', 12)
    set(gca, 'FontWeight', 'bold')
    
    
    %%=======================================================     Bar chart for KLDs
    axes('position',[0.7036    0.4929    0.2012    0.4309]);
    
    h = bar(kld); axis tight
    ch = get(h,'Children');
    fvd = get(ch,'Faces');
    fvcd = get(ch,'FaceVertexCData');
    for i = 1:nfilters, fvcd(fvd(i,:)) = i; end
    set(ch,'FaceVertexCData',fvcd);
    
    mkld = str2num(sprintf('%.2f', mean(kld)));
    line([0 nfilters+1], repmat(mkld,1,2), 'color', 'k', 'linestyle', '--')
    set(gca, 'xtick', []); set(gca, 'ytick', [0 mkld]);
    ax = axis; axis([ax(1) ax(2) 0 1.1*max(kld)]);
    
  end
  
end

% compute discrete distribution for filter responses of each filter
function dz = filter_responses(imdims, x, filters, b, nbins, discard_boundary)
  
  % indices of "interior" pixels
  [rs, cs] = ndgrid(1+discard_boundary:imdims(1)-discard_boundary, ...
                    1+discard_boundary:imdims(2)-discard_boundary);
  ind_int = sub2ind(imdims, rs(:), cs(:));
  
  imdims = imdims - 2*discard_boundary;
  
  nimages  = size(x, 2);
  nfilters = length(filters);
  
  % preallocate vector of filter responses
  fil = cell(1, nfilters);
  nelements = zeros(1, nfilters);
  for j = 1:nfilters
    [frows, fcols] = size(filters{j});
    % #elements of convolved image with 'valid' option
    nelements(j) = (imdims(1)-frows+1) * (imdims(2)-fcols+1);
    fil{j} = zeros(1, nelements(j) * nimages);
  end
  
  conv2_fun = @(img, f) conv2(img, f, 'valid');
  
  for i = 1:nimages
    img = reshape(x(ind_int, i), imdims);
    fprintf('\rProcessing %d / %d', i, nimages)
    for j = 1:nfilters
      % apply filter
      d_filter = conv2_fun(img, filters{j});
      % collect filter responses in a big vector
      fil{j}(nelements(j)*(i-1)+1:nelements(j)*i) = d_filter(:);
    end
  end
  
  dz = cell(1, nfilters);
  
  for i = 1:nfilters
    % create discrete distribution, set domain start and stride, fit to filter responses
    nweights = nbins;
    dz{i} = pml.distributions.discrete(ones(nweights,1));
    dz{i}.domain_stride = 2*b / (nweights-1);
    dz{i}.domain_start = -b;
    % discard all filter responses outside the given bounds
    % add one count to each bin
    fil{i} = [fil{i}(fil{i} >= -b & fil{i} <= b), -b:dz{i}.domain_stride:b];
    dz{i} = dz{i}.mle(fil{i});
  end
  fprintf('\n')
  
end
