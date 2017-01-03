function xs = montecarlo(distribution, ys, N);

% oldN >= N
oldN = length(distribution);

% Binary search with all ys at the same time
lows = zeros(N,1);
highs = ones(N,1)*oldN;

while (max(highs - lows) > 1)
  xs = round((highs + lows)/2);
  lows = (distribution(xs) < ys).*xs + (distribution(xs) >= ys).*lows;
  highs = (distribution(xs) < ys).*highs + (distribution(xs) >= ys).*xs;
end

xs = highs;

