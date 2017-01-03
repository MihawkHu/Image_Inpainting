function p = eval(this, x)
  tmp = bsxfun(@minus, x, this.domain_start(:));
  idx = 1 + round(bsxfun(@rdivide, tmp, this.domain_stride(:)));
  
  ndims = size(x, 1);
  dims = mat2cell(idx, ones(1, ndims), size(x, 2));
  
  p = this.weights(sub2ind(size(this.weights), dims{:}));
  p = p(:)';
end
