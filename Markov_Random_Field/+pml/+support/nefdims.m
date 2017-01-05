function N = nefdims(X)
%NEFDIMS   Number of (effective) dimensions.
%   NEFDIMS(X) returns the number of dimensions in the array X.  In
%   contrast to NDIMS(X) it will return 1 if X is a row or column vector,
%   or a scalar. 
  
  N = ndims(X);

  if (N == 2)
    s = size(X);
    if (prod(s) == max(s))
      N = 1;
    end
  end