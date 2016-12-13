function xs = montecarlo(distribution, ys, N);
% 
% xs = montecarlo(distribution, ys, N)
% Returns N x:s given y for a cumulative distribution y(x)
%
% Written by: Hedvig Sidenbladh, KTH, Sweden
% http://www.nada.kth.se/~hedvig/
% Date: March 2002


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

