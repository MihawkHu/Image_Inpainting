%% EPSR - "estimated potential scale reduction" (Gelman and Rubin).

function [R_hat] = epsr(estimands)
  [niters, nsamplers, nestimands] = size(estimands);
  R_hat = zeros(nestimands, 1);

  for k = 1:nestimands
    mean_sampler = mean(estimands(:,:,k), 1);
    mean_overall = mean(mean_sampler);

    % between sequence variance
    B = (niters/(nsamplers-1)) * sum((mean_sampler - mean_overall).^2);
    % within sequence variance
    % W = (1/nsamplers) * sum(sum(bsxfun(@minus, estimands(:,:,k), mean_sampler).^2) / (niters-1))
    W = (1/nsamplers) * sum(var(estimands(:,:,k)));

    v_hat = ((niters-1)/niters)*W + (1/niters)*B;
    R_hat(k) = sqrt(v_hat / W);
  end
end