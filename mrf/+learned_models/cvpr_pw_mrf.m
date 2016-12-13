%% CVPR_PW_MRF - Our learned pairwise MRF as described in the paper
% 
%  Uwe Schmidt, Qi Gao, Stefan Roth.
%  A Generative Perspective on MRFs in Low-Level Vision.
%  IEEE Conference on Computer Vision and Pattern Recognition (CVPR'10), San Francisco, USA, June 2010.

% Copyright 2010 TU Darmstadt, Darmstadt, Germany.
% $Id: cvpr_pw_mrf.m 239 2011-05-30 12:34:09Z uschmidt $

function mrf = cvpr_pw_mrf
  gsm = pml.distributions.gsm;
  gsm.precision = 0.003228502953588;
  gsm.scales = exp([-9,-7,-5:5,7,9]);
  gsm.weights = [0.041455394458946   0.050543704668592   0.101362002161222   0.234096619655871   0.233440713570801   0.085300228433082   0.051357256864545   0.044820782129556   0.037734694750911   0.027167475968450   0.023184083069899   0.032863685840126   0.016786126204713   0.000693422654205   0.019193809569083]';
  mrf = pml.distributions.gsm_pairwise_mrf;
  mrf.experts = {gsm};
end