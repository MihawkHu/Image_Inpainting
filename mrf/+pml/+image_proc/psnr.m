function n = psnr(x, y)

  diff = double(x) - double(y);
  n = 20 * log10(255 / std(diff(:)));

