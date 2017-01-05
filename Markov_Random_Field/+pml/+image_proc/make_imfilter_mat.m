function M = make_imfilter_mat(F, sz, bndry, shape)

  import pml.image_proc.imfiltermtx;
  
  
  % Default to 'same' matrix
  if (nargin < 4)
    shape = 'same';
  end
  
  % Default to zero boundaries
  if (nargin < 3)
    bndry = '0';
  end
  
  ndims = length(sz);
  
  % Border sizes for 'same'
  Fsize_2 = round((size(F) - 1) / 2);

  switch(shape)
    case 'same'
      % Mark valid and invalid pixels (i.e. the ones within and outside
      % of the part to be returned)
      valid = true(sz+size(F)-1);
      
      for d = 1:ndims
        for e = 1:ndims
          sub{e} = ':';
        end
        
        sub{d} = 1:Fsize_2(d);      
        valid(sub{:}) = false;
        sub{d} = size(valid, d)-Fsize_2(d)+1:size(valid, d);      
        valid(sub{:}) = false;
      end
      
      % Image filter matrix with appropriate boundary handling
      M = imfiltermtx(F, sz, bndry);
      % Suppress rows of M outside of the valid area
      M = M(valid, :);
      
    case 'valid'
      % Mark valid and invalid pixels (i.e. the ones within and outside
      % of the part to be returned)
      valid = true(sz+size(F)-1);
      
      for d = 1:ndims
        for e = 1:ndims
          sub{e} = ':';
        end
        
        sub{d} = 1:2*Fsize_2(d);      
        valid(sub{:}) = false;
        sub{d} = size(valid, d)-2*Fsize_2(d)+1:size(valid, d);      
        valid(sub{:}) = false;
      end
      
      % Image filter matrix with appropriate boundary handling
      M = imfiltermtx(F, sz, bndry);
      % Suppress rows of M outside of the valid area
      M = M(valid, :);
      
    otherwise
      % Full convolution; return everything
      M = imfiltermtx(F, sz, bndry);
      
  end
