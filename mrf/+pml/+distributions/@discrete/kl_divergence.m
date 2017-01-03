function d = kl_divergence(this, other)
  if (~isequal(size(this.weights), size(other.weights)))
    d = Inf;
    return;
  end 
  
  d = sum(this.weights(:) .* (log(this.weights(:)) - log(other.weights(:))));
end
