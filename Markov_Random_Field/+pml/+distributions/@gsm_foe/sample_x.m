function [x, W_Z_Wt] = sample_x(this, z, x_cond)
  
  npixels = prod(this.imdims);
  nfilters = this.nfilters;
  nexperts = this.nexperts;
  
  cond_sampler = this.conditional_sampling && nargin > 2;
  
  N = 0;
  for i = 1:nfilters
    expert_precision = this.experts{min(i,nexperts)}.precision;
    z{i} = expert_precision * z{i}(:);
    N = N + length(z{i});
  end

  if cond_sampler
    max_filter_size = this.filter_size;
    mr = max_filter_size(1) - 1; mc = max_filter_size(2) - 1;
    [rs, cs] = ndgrid(1+mr:this.imdims(1)-mr, 1+mc:this.imdims(2)-mc);
    ind_int = sub2ind(this.imdims, rs(:), cs(:));
    ind_ext = setdiff(1:npixels, ind_int);
    npixels_int = length(ind_int);
    npixels_ext = length(ind_ext);
    
    x_ext = x_cond(ind_ext);
    
    Wt_ext = cellfun(@(F) {F(:,ind_ext)}, this.filter_matrices(1:nfilters));
    Wt_int = cellfun(@(F) {F(:,ind_int)}, this.filter_matrices(1:nfilters));
    C = vertcat(Wt_int{:})' * (spdiags(vertcat(z{:}), 0, N, N) * vertcat(Wt_ext{:}));
    
    % add epsilon
    Wt = {Wt_int{:}, speye(npixels_int)};
    z = {z{:}, this.epsilon(ones(npixels_int, 1))};
    N = N + npixels_int;
    
    Wt = vertcat(Wt{:});
    Z = spdiags(vertcat(z{:}), 0, N, N);
    
    y = randn(N, 1);
    W_sqrtZ_y = Wt' * sqrt(Z) * y;
    W_Z_Wt = Wt' * Z * Wt;
    
    solve_sle = pml.numerical.sle_spd_solver(W_Z_Wt);
    x_int_mu = - solve_sle(C * x_ext);
    x_int = x_int_mu + solve_sle(W_sqrtZ_y);
    
    x = zeros(npixels, 1);
    x(ind_int) = x_int;
    x(ind_ext) = x_ext;
    
  else
    % add epsilon
    Wt = {this.filter_matrices{1:nfilters}, speye(npixels)};
    z = {z{:}, this.epsilon(ones(npixels, 1))};
    N = N + npixels;
    
    Wt = vertcat(Wt{:});
    Z = spdiags(vertcat(z{:}), 0, N, N);
    
    y = randn(N, 1);
    W_sqrtZ_y = Wt' * sqrt(Z) * y;
    W_Z_Wt = Wt' * Z * Wt;
    
    solve_sle = pml.numerical.sle_spd_solver(W_Z_Wt);
    x = solve_sle(W_sqrtZ_y);
  end
  
end