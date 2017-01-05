function M = make_convn_mat(F, sz, shape, pad)

  import pml.image_proc.convmtxn;
  
  
  % Default to full matrix
  if (nargin < 3)
    shape = 'full';
  end
  
  ndims = length(sz);
  sub   = cell(1, ndims);
  
  % Border sizes for 'same' and 'sameswap'
  Fsize_lo_2 = ceil((size(F) - 1) / 2);
  Fsize_hi_2 = floor((size(F) - 1) / 2);
  
  % Border sizes for 'valid'
  Fsize = size(F) - 1;

  switch(shape)
    case 'same'
      % Mark valid and invalid pixels (i.e. the ones within and outside
      % of the part to be returned)
      valid = true(sz+size(F)-1);
      
      for d = 1:ndims
        for e = 1:ndims
          sub{e} = ':';
        end
        
        sub{d} = 1:Fsize_lo_2(d);      
        valid(sub{:}) = false;
        sub{d} = size(valid, d)-Fsize_hi_2(d)+1:size(valid, d);      
        valid(sub{:}) = false;
      end

      if (nargin > 3 && strcmp(pad, 'full'))
        % If we're padding to 'full' set the coefficients on the border
        % to zero
        M = convmtxn(F, sz, valid);
      else
        % If we're *not* padding, then suppress the rows of M that
        % correspond to the border
        M = convmtxn(F, sz);
        M = M(valid, :);
      end
    
   case 'sameswap'
     % Mark valid and invalid pixels (i.e. the ones within and outside
     % of the part to be returned), but round the other way
     valid = true(sz+size(F)-1);
     
     for d = 1:ndims
       for e = 1:ndims
         sub{e} = ':';
       end
       
       sub{d} = 1:Fsize_hi_2(d);      
       valid(sub{:}) = false;
       sub{d} = size(valid, d)-Fsize_lo_2(d)+1:size(valid, d);      
       valid(sub{:}) = false;
     end
     
     if (nargin > 3 && strcmp(pad, 'full'))
       % If we're padding to 'full' set the coefficients on the border
       % to zero
       M = convmtxn(F, sz, valid);
     else
       % If we're *not* padding, then suppress the rows of M that
       % correspond to the border
       M = convmtxn(F, sz);
       M = M(valid, :);
     end
    
   case 'valid'
     % Mark valid and invalid pixels (i.e. the ones within and outside
     % of the part to be returned)
     valid = true(sz+size(F)-1);
     
     for d = 1:ndims
       for e = 1:ndims
         sub{e} = ':';
       end
       
       sub{d} = 1:Fsize(d);      
       valid(sub{:}) = false;
       sub{d} = size(valid, d)-Fsize(d)+1:size(valid, d);      
       valid(sub{:}) = false;
     end
     
     if (nargin > 3)
       % If we're padding, then figure out the area to be padded       

       switch (pad)
         case 'same'
           % Mark valid and invalid pixels (i.e. the ones within and outside
           % of the part to be padded)
           pad_valid = true(sz+size(F)-1);
           
           for d = 1:ndims
             for e = 1:ndims
               sub{e} = ':';
             end
             
             sub{d} = 1:Fsize_lo_2(d);      
             pad_valid(sub{:}) = false;
             sub{d} = size(valid, d)-Fsize_hi_2(d)+1:size(valid, d);      
             pad_valid(sub{:}) = false;
           end
           
           % Set coefficients on the border to zero
           M = convmtxn(F, sz, valid);
           
           % Suppress rows of M outside of the padded area
           M = M(pad_valid, :);
           
         case 'sameswap'
           % Mark valid and invalid pixels (i.e. the ones within and outside
           % of the part to be padded), but round the other way
           pad_valid = true(sz+size(F)-1);
           
           for d = 1:ndims
             for e = 1:ndims
               sub{e} = ':';
             end
             
             sub{d} = 1:Fsize_hi_2(d);      
             pad_valid(sub{:}) = false;
             sub{d} = size(valid, d)-Fsize_lo_2(d)+1:size(valid, d);      
             pad_valid(sub{:}) = false;
           end
           
           % Set coefficients on the border to zero
           M = convmtxn(F, sz, valid);
           
           % Suppress rows of M outside of the padded area
           M = M(pad_valid, :);
           
         otherwise
           % Padding to 'full'; only set coefficients on the border to zero
           M = convmtxn(F, sz, valid);           
       end
     else
       % No padding; suppress all rows on the border
       M = convmtxn(F, sz);
       M = M(valid, :);
     end
     
    otherwise
      % Full convolution; return everything
      M = convmtxn(F, sz);
      
  end
