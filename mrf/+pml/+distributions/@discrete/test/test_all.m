function [] = test_all()
  function [] = test(str)
    disp(str);
    eval(str);
  end

  function [] = test_probs(d, x)
    p = d.eval(x);
    u = d.unnorm(x);
    l = d.log(x);
    e = d.energy(x);
    
    max(abs((p ./ sum(p)) - (u ./ sum(u))))
    max(abs(log(p) - l))
    max(abs(log(u) + e))
  end

  
  %% 1D tests
  d = pml.distributions.discrete;
  d.weights = 1:10;
  d.domain_start = 0;
  d.domain_stride = 0.1;
  
  test_probs(d, 0:0.1:0.9);
  pause
  
  test('d.weights')
  test('d.ndims')
  test('d.mean')
  test('d.covariance')
  test('d.kurtosis')
  test('d.entropy')
  test('d.domain_grid{1}')
  
  test('d.eval(0:0.1:0.9)')
  test('d.unnorm(0:0.1:0.9)')
  test('d.log(0:0.1:0.9)')
  test('d.energy(0:0.1:0.9)')
  test('d.sample(10)')
  
  test('d.plot')
  pause
  test('d.semilogy')
  pause
  
  d.weights = 1:100;
  d2 = d;
  
  g = pml.distributions.gaussian(5, 1);
  d2 = from_density(d2, g);
  
  test('d2.plot')
  pause
  
  test('d.bhattacharyya(d2)')
  test('d.kl_divergence(d2)')
  test('d.bhattacharyya(d)')
  test('d.kl_divergence(d)')
  
  s = g.sample(1000);
  d = mle(d, s);

  d.weights = d.weights + 1e-6;
  
  test('d.bhattacharyya(d2)')
  test('d.kl_divergence(d2)')
  
  test('d.plot')
  pause


  %% 2D tests with dependencies
  g = pml.distributions.gaussian([0; 1], 0.25 * [2*sqrt(2) sqrt(2); ...
                      -2*sqrt(2) sqrt(2)]);
  d.weights = zeros(701, 801);
  d.domain_start = [-2; -2];
  d.domain_stride = [0.01; 0.01];
  d = from_density(d, g);
  
  test('size(d.weights)')
  test('d.ndims')
  test('d.mean')
  test('d.covariance')
  test('d.kurtosis')
  test('d.entropy')
  test('d.mi')
  test('size(d.domain_grid{1})')
  test('size(d.domain_grid{2})')
  pause
  
  test('d.eval([0:0.1:0.9; 0.9:-0.1:0])')
  test('d.sample(10)')
  
  test('d.surf')
  pause
  test('d.logsurf')
  pause
  test('d.contour')
  pause
  test('d.logcontour')
  pause


  %% 2D tests without dependencies
  g = pml.distributions.gaussian([0; 1], 0.5 * eye(2));
  d = from_density(d, g);
  
  test('d.mean')
  test('d.covariance')
  test('d.kurtosis')
  test('d.entropy')
  test('d.mi')
  
end

