function I = random_patches(X, s, npatches)
  
  nimages = length(X);
  if nargin < 3, npatches = 1; end
  I = zeros(prod(s), nimages * npatches);
  
  for i = 1:nimages
    img = X{i};
    [h w] = size(img);
    if ~all([h w] > s), error('Image patch larger than image.'); end
    for j = 1:npatches
      y = pml.support.randi(h-s(1)+1);
      x = pml.support.randi(w-s(2)+1);
      patch = img(y:y-1+s(1), x:x-1+s(2));
      I(:,(i-1)*npatches + j) = patch(:);
    end
  end
end