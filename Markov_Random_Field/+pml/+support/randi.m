function i = randi(n)
%RANDI   Random integer between 1 and n.
%   RANDI(N) draws a single random integer between 1 and n.
  [ignore, i] = min(rand(1, n));