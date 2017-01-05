function M = imfiltermtx(F, sz, bndry);
  import pml.image_proc.convmtxn;
  import pml.image_proc.make_convn_mat;


  % Size of padding for boundary handling 
  pad_size = 0.5 * (size(F) - 1);
  
  % Corners of the original image in the padded image
  corner_lo = 1 + pad_size;
  corner_hi = size(F) + sz - 1 - pad_size;
  

  % Pixel index of the output image M
  M_idx = zeros(sz + size(F) - 1);
  M_idx(:) = 1:numel(M_idx);
  
  % Pixel index of the "input" image X
  X_idx = zeros(sz);
  X_idx(:) = 1:numel(X_idx);
  % Pad it to the size of M
  X_idx = [zeros(pad_size(1), sz(2) + size(F, 2) - 1); ...
           zeros(sz(1), pad_size(2)), X_idx, zeros(sz(1), pad_size(2)); ...
           zeros(pad_size(1), sz(2) + size(F, 2) - 1)];
  
  switch(bndry)
    case 'symmetric'
      error('Not implemented yet!')
      
    case 'replicate'
      % Replicate pixel values from the edges & corners
      
      % Fill all 4 corners
      for c = 0:3
        
        % Position or corner and range of block next to corner
        pos = cell(1, 2);
        rng = cell(1, 2);
        for d = 1:2
          if bitand(c, 2^(d-1))
            pos{d} = corner_hi(d);
            rng{d} = corner_hi(d)+1:size(M_idx, d);
          else
            pos{d} = corner_lo(d);
            rng{d} = 1:corner_lo(d)-1;
          end
        end
        
        % Get index of corner pixel
        idx = sub2ind(size(M_idx), pos{:});
        
        % Set entire block to corner pixel index
        M_idx(rng{:}) = idx;
        
      end
      
      % Fill all 4 edges
      for d = 1:2
        for c = 0:3
          
          % I don't quite understand this code anymore, but it appears to
          % work fine :)
          pos = cell(1, 2);
          rng = cell(1, 2);
          for e = 1:2
            if (d == e)
              pos{d} = corner_lo(e):corner_hi(e);
              rng{d} = corner_lo(e):corner_hi(e);
            else
              if bitand(c, 2^(e-1))
                pos{e} = repmat(corner_hi(e), 1, sz(d));
                rng{e} = corner_hi(e)+1:size(M_idx, e);
              else
                pos{e} = repmat(corner_lo(e), 1, sz(d));
                rng{e} = 1:corner_lo(e)-1;
              end
            end
          end
          
          % Get indices of edge pixels
          idx = sub2ind(size(M_idx), pos{:});
          
          % Generate fill indices from the nearest edge pixel
          tmp_sz = pad_size;
          tmp_sz(d) = 1;
          tmp2_sz = ones(1, 2);
          tmp2_sz(d) = sz(d);
          M_idx(rng{:}) = repmat(reshape(idx, tmp2_sz), tmp_sz);
        end
      end
      
      % Convert indices from the output image to the input image
      M_idx = X_idx(M_idx);
      
    case 'circular'
      % Create padded indices of rows and columns
      r = -pad_size(1)+1:sz(1)+pad_size(1);
      c = -pad_size(2)+1:sz(2)+pad_size(2);
      
      % Take the modolus to make sure they are in the valid bounds
      r = 1 + mod(r - 1, sz(1));
      c = 1 + mod(c - 1, sz(2));
      
      M_idx = sub2ind(sz, repmat(r(:), 1, length(c)), ...
                      repmat(c(:)', length(r), 1));
      
      
    otherwise
      % Assume all zeros outside of the boundaries
      M = convmtxn(F, sz);
      return;
    
  end
  
  % The pixel indices give a mapping of the columns of the padded image
  % to the output image
  col_map = M_idx(:);

  % Get convolution matrix and split the resulting sparse array
  M = make_convn_mat(F, sz + size(F) - 1, 'same');
  [rows, cols, vals] = find(M);

  % Re-assemble the sparse array, but use the column map to accumulate
  % rows appropriately
  M = sparse(rows, col_map(cols), vals, prod(size(F) + sz - 1), prod(sz));
