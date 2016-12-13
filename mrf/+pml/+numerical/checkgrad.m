function d = checkgrad(f, X, e, varargin);

% checkgrad checks the derivatives in a function, by comparing them to finite
% differences approximations. The partial derivatives and the approximation
% are printed and the norm of the diffrence divided by the norm of the sum is
% returned as an indication of accuracy.
%
% usage: checkgrad('f', X, e, P1, P2, ...)
%
% where X is the argument and e is the small perturbation used for the finite
% differences. and the P1, P2, ... are optional additional parameters which
% get passed to f. The function f should be of the type 
%
% [fX, dfX] = f(X, P1, P2, ...)
%
% where fX is the function value and dfX is a vector of partial derivatives.
%
% Carl Edward Rasmussen, 2001-08-01.
%
% Changed to use FEVAL instead of EVAL to allow using function handles for 'f'.
% $Id: checkgrad.m 209 2010-04-21 17:31:39Z uschmidt $

[y dy] = feval(f, X, varargin{:});             % get the partial derivatives dy

dh = zeros(length(X),1) ;
for j = 1:length(X)
  dx = zeros(length(X),1);
  dx(j) = dx(j) + e;                               % perturb a single dimension
  y2 = feval(f, X+dx, varargin{:});
  dx = -dx ;
  y1 = feval(f, X+dx, varargin{:});
  dh(j) = (y2 - y1)/(2*e);
end

disp([dy dh])                                          % print the two vectors
d = norm(dh-dy)/norm(dh+dy);       % return norm of diff divided by norm of sum
