function fun = sle_spd_solver(A)
  [L p s] = chol(A, 'lower', 'vector');
  assert(p == 0, 'Matrix is not positive definite.');
  function x = solve_sle(b)
    x = zeros(size(b));
    x(s) = L' \ (L \ b(s));
  end
  fun = @solve_sle;
end