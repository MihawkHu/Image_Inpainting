function demo_inpainting

  img = double(imread('../dataset/images/zouyikai.png'));
  mask = logical(imread('../dataset/images/zouyikai_mask.png'));
  
  sc = 1/3;
  img = imresize(img, sc, 'nearest'); mask = imresize(mask, sc, 'nearest');
  
  img_inpainted = inpaint_mmse(learned_models.cvpr_pw_mrf, img, mask, false, true);
  
  imwrite(uint8(img_inpainted), 'zouoyikai_result.png');
end
