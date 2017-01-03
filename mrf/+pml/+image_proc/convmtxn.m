function M = convmtxn(F, sz, valid)
  
  ndims  = length(sz);
  blksz  = prod(size(F));
  nblks  = prod(sz);
  nelems = blksz * nblks;

  
  % Build index array for all possible image positions
  tmp = zeros(size(F) + sz - 1);
  sub = cell(1, ndims);
  for d = 1:ndims
    sub{d} = 1:sz(d);
  end
  tmp(sub{:}) = 1;  
  imgpos = find(tmp(:));

  % Build index array for all possible filter positions
  tmp = zeros(size(F) + sz - 1);
  for d = 1:ndims
    sub{d} = 1:size(F, d);
  end
  tmp(sub{:}) = 1;  
  fltpos = find(tmp(:));
  
  
  % The loop code below is replaced with the vectorized version below  
% $$$   rows = zeros(nelems, 1);
% $$$   cols = zeros(nelems, 1);
% $$$   vals = zeros(nelems, 1);
% $$$   
% $$$   for i = 1:prod(sz)
% $$$     % For every possible image position (cols) insert filter values
% $$$     % into the appropriate output position (rows).
% $$$   
% $$$     j = 1 + (i-1) * blksz;
% $$$     k = i * blksz;
% $$$     rows(j:k) = imgpos(i) + fltpos - 1;
% $$$     cols(j:k) = i;
% $$$     vals(j:k) = F(:);
% $$$   end

  % Vectorized version of loop code above.
  rows = reshape(repmat(imgpos', blksz, 1), nelems, 1) + ...
         repmat(fltpos - 1, nblks, 1);
  cols = reshape(repmat(1:nblks, blksz, 1), nelems, 1);
  vals = repmat(F(:), nblks, 1);

  % Pick out valid rows
  if (nargin > 2)
    valid_idx = valid(rows);

    rows = rows(valid_idx);
    cols = cols(valid_idx);
    vals = vals(valid_idx);
  end
  
  % Build sparse output array
  M = sparse(rows, cols, vals, prod(size(F) + sz - 1), nblks);
