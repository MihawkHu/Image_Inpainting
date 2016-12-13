%% SGD - Stochastic Gradient Descent
% |[X, REPORT] = SGD(FUNC, X0, DATA, [OPTIONS])| attempts to minimize FUNC by
% doing stochastic gradient descent. FUNC is assumed to be the handle for a
% function that takes 2 input arguments X and DATA and outputs GRADX. DATA
% is assumed to be a matrix of size [ndims x nexamples], where each column
% denotes a training example. X, X0, and GRADX must all be column vectors.
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
% $Id: sgd.m 232 2010-07-08 13:18:35Z uschmidt $

function [x, report] = sgd(func, x0, data, options)
  
  ndata = size(data, 2);
  iter = 1;
  xs = x0;
  x = x0;
  
  if nargin < 4, options = struct; end
  
  % how many training examples should be used together in a mini-batch
  [minibatchsize, options] = option(options, 'MinibatchSize', 1);
  % start with batch number
  [startbatch, options] = option(options, 'StartBatch', 1);
  % maximum number of full batches
  [maxbatches, options] = option(options, 'MaxBatches', 500);
  % check for convergence after every X batches (by evaluating the objective function on the whole training set)
  % set to 1 to enable
  [convergence_check, options] = option(options, 'ConvergenceCheck', 0);
  % convergence is assumed to be reached if change of norm(x-x_)/norm(x+x_) x is less than this value
  [convergence_change, options] = option(options, 'ConvergenceChange', eps);
  % learning rate (to be adjusted with MinibatchSize)
  [eta, options] = option(options, 'LearningRate', 0.001);
  % function to decrease learning rate over time
  [eta_func, options] = option(options, 'LearningRateFactor', @(batch,minibatch) 1);
  % permute the order in which training examples are processed (in each batch)
  [permutedata, options] = option(options, 'PermuteData', true);
  % print status information
  [verbosive, options] = option(options, 'Verbosive', true);
  % set to index of output argument of FUNC that contains the objective function f(x) (0 = disabled)
  [recordobjective, options] = option(options, 'RecordObjective', 0);
  % Smooth gradient, set weight alpha of latest gradient
  [alpha, options] = option(options, 'LatestGradientWeight', 1);
  
  if verbosive
    fprintf('Options:\n'), disp(options), fprintf('\n')
    print = @(varargin) fprintf(varargin{:});
  else
    print = @(varargin) fprintf('');
  end
  
  if nargout > 1
    report = struct;
    nmaxiters = ceil(ndata/minibatchsize)*(maxbatches-startbatch+1);
    report.iter_x = zeros(length(x0),nmaxiters);
    report.iter_gradx = zeros(length(x0),nmaxiters);
    report.iter_eta = zeros(1,nmaxiters);
    if recordobjective > 0
      report.iter_fx = zeros(1,nmaxiters);
      %report.batch_x = zeros(length(x0),maxbatches);
      %report.batch_gradx = zeros(length(x0),maxbatches);
      %report.batch_fx = zeros(1,maxbatches);
    end
  end
  
  % previous gradient, used for smoothing
  last_grad = zeros(length(x0),1);
  
  % cell array for output argument of FUNC
  out = cell(1, max(1, recordobjective));
    
  % limit minibatchsize to ndata
  minibatchsize = min(minibatchsize, ndata);
  % data indices if not permuting
  data_idx = 1:ndata;
  
  for batch = startbatch:maxbatches
    if permutedata
      % random permutation of data indices
      data_idx = randperm(ndata);
    end
    
    % run mini-batches
    for i = 1:minibatchsize:ndata
      % indices of training examples in this mini-batch
      minibatch_idx = i:min(i+minibatchsize-1, ndata);
      
      % evaluate gradient
      % grad_x = func(x, data(:, data_idx(minibatch_idx)));
      [out{:}] = func(x, data(:, data_idx(minibatch_idx)));
      grad_x = out{1};
      
      % take a step
      smooth_grad = (1 - alpha) * last_grad + alpha * grad_x;
      curreta = eta_func(batch, iter) * eta;
      x = x - curreta * smooth_grad;
      
      last_grad = smooth_grad;
            
      if nargout > 1
        report.iter_x(:,iter) = x;
        report.iter_gradx(:,iter) = grad_x;
        if recordobjective > 0, report.iter_fx(iter) = out{recordobjective}; end
        report.iter_eta(iter) = curreta;
      end
      
      print('\rBatch: %03d, Minibatch: %04d', batch, iter);
      iter = iter + 1;
      
    end
    
    % evaluate objective function on the whole data set
    %if nargout > 1 && recordobjective > 0
    %  [out{:}] = func(x, data);
    %  report.batch_x(:,batch) = x;
    %  report.batch_gradx(:,batch) = out{1};
    %  report.batch_fx(batch) = out{recordobjective};
    %end    
    
    % time to check convergence?
    if convergence_check ~= 0 && mod(batch, convergence_check) == 0
      xs(:,end+1) = x;
      
      if size(xs, 2) > 1,
        % how much x has changed
        prev_x = xs(:,end-1);
        dx = norm(x - prev_x) / norm(x + prev_x);
        % print progress
        print(', dx = %f\n', dx);
        % check for convergence
        if dx < convergence_change
          print('Convergence reached.\n')
          break;
        end
      else
        print('\n');
      end
    end
    
  end
  print('\n');
  
end

function [value, options] = option(options, fieldname, default)
  if isfield(options, fieldname)
    value = getfield(options, fieldname);
  else
    value = default;
    options = setfield(options, fieldname, default);
  end
end
