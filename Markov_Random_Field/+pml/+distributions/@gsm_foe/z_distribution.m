function pz = z_distribution(this, x)
  
  pz = cell(1, this.nfilters);
  
  % convert to 2D image
  img = reshape(x, this.imdims);
  
  for i = 1:this.nfilters
    % filter responses
    d_filter = this.conv2(img, this.filter(i));
    % filter's expert
    expert = this.experts{min(i,this.nexperts)};
    % discrete distributions over scales
    pz{i} = expert.z_distribution(d_filter(:)');
  end
  
end