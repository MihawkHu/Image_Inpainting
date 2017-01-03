%% ASSEMBLE_IMAGES - Assemble a single big image from multiple, equally sized, images.
function I = assemble_images(X, sz, ipr)
  
  [npixels nimages] = size(X);
  h = sz(1); w = sz(2);
  if nargin < 3, ipr = ceil(sqrt(nimages*(h/w))); end
  
  I = zeros(sz .* [ceil(nimages/min(ipr,nimages)) min(ipr,nimages)]);
  
  row = 1; col = 1;
  for i = 1:nimages
    y = (row-1)*h+1;
    x = (col-1)*w+1;
    I(y:y-1+h, x:x-1+w) = reshape(X(:,i), sz);
    col = col + 1;
    if col > ipr, col = 1; row = row + 1; end
  end
  
end