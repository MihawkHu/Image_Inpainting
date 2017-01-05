%% MIRROR_BOUNDARY - Mirror boundary of image I and return padded image

function I_padded = mirror_boundary(I, b)
  I_padded = zeros(size(I)+2*b);
  [rows, cols] = size(I_padded);
  rows_middle = 1+b:rows-b;  cols_middle = 1+b:cols-b;
  rows_start = 1:b;          cols_start = 1:b;
  rows_end = rows-b+1:rows;  cols_end = cols-b+1:cols;
  
  I_padded(rows_middle, cols_middle) = I;
  
  I_padded(rows_start,  cols_middle) = flipud(I(1:b, :));         % top
  I_padded(rows_end,    cols_middle) = flipud(I(end-b+1:end, :)); % bottom
  I_padded(rows_middle, cols_start)  = fliplr(I(:, 1:b));         % left
  I_padded(rows_middle, cols_end)    = fliplr(I(:, end-b+1:end)); % right
  
  % corners
  I_padded(rows_start,  cols_start)  = flipud(fliplr(I(1:b, 1:b)));                 % top left
  I_padded(rows_start,  cols_end)    = flipud(fliplr(I(1:b, end-b+1:end)));         % top right
  I_padded(rows_end,    cols_start)  = flipud(fliplr(I(end-b+1:end, 1:b)));         % bottom left
  I_padded(rows_end,    cols_end)    = flipud(fliplr(I(end-b+1:end, end-b+1:end))); % bottom right
end
