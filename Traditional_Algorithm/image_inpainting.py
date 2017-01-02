import numpy as np
from math import sqrt
from PIL import Image
from scipy import ndimage
from scipy.misc import imsave
from skimage.morphology import erosion, disk

input_img = Image.open("./input_image/001.jpg")
img = np.array(input_img)

input_mask = Image.open("./input_mask/002.bmp")
mask = np.array(input_mask, dtype=np.uint8)


# get the patch with center (x, y)
def get_patch(x, y, image, patch_size):
    p = patch_size // 2

    x_begin = x - p
    if x_begin < 0:
        x_begin = 0

    y_begin = y - p
    if y_begin < 0:
        y_begin = 0

    patch = image[x_begin: x + p + 1, y_begin: y + p + 1]

    return patch


# compute priorities of each patch
# it's based on the fomula in paper criminisi_tip2004
# alpha is a parameter uggested by the author
def get_max_priority_patch(fill_front_x, fill_front_y, dx, dy, nx, ny,
                           confidence, patch_size, alpha=255.0):
    # computer prioritie
    # P(p) = C(p) * D(p)
    # where C is confidence term and D is data term

    # computer confidence term
    confidence_value = np.sum(get_patch(fill_front_x[0], fill_front_y[0],
                                        confidence, patch_size)) / (patch_size ** 2)

    # computer data term

    # computer gradients of each fill_front
    # grad = dx^2 + dy^2
    grad = np.hypot(dx, dy)

    # get the maximum gradient magnitude at first
    grad_patch = abs(get_patch(fill_front_x[0], fill_front_y[0], grad, patch_size))

    xx = np.where(grad_patch == np.max(grad_patch))[0][0]
    yy = np.where(grad_patch == np.max(grad_patch))[1][0]

    max_gradx = dx[xx][yy]
    max_grady = dy[xx][yy]

    Nx = nx[fill_front_x[0]][fill_front_x[0]]
    Ny = ny[fill_front_y[0]][fill_front_y[0]]

    x = fill_front_x[0]
    y = fill_front_y[0]

    # gradient part of data term
    data = abs(max_gradx * Nx + max_grady * Ny)

    if (Nx ** 2 + Ny ** 2) != 0:
        data /= (Nx ** 2 + Ny ** 2)

    # computer all priorities and get the max prioritie
    max_temp = confidence_value * (data / alpha)
    i = 1
    while i < len(fill_front_x):
        # computer confidence term
        curr_patch = get_patch(fill_front_x[i], fill_front_y[i],
                               confidence, patch_size)
        curr_conf = np.sum(curr_patch) / (patch_size ** 2)

        # computer data term
        grad_patch = abs(get_patch(fill_front_x[i], fill_front_y[i],
                                   grad, patch_size))

        xx = np.where(grad_patch == np.max(grad_patch))[0][0]
        yy = np.where(grad_patch == np.max(grad_patch))[1][0]

        max_gradx = dx[xx][yy]
        max_grady = dy[xx][yy]

        Nx = nx[fill_front_x[i]][fill_front_y[i]]
        Ny = ny[fill_front_x[i]][fill_front_y[i]]

        curr_data = abs(max_gradx * Nx + max_grady * Ny)

        if (Nx ** 2 + Ny ** 2) != 0:
            curr_data /= (sqrt(Nx ** 2 + Ny ** 2))

        curr_p = curr_conf * (curr_data / alpha)

        if curr_p > max_temp:
            max_temp = curr_p
            x = fill_front_x[i]
            y = fill_front_y[i]
        i += 1

    return x, y


# computer the ssd between two patches
def patch_ssd(patch_dst, patch_src):
    m = patch_dst.shape[0]
    n = patch_dst.shape[1]

    # get the patch value in RGB layer
    # reshape them into one dimension
    patch_srcc = patch_src[:m, :n, :]
    patch_dst_r = patch_dst[:, :, 0].flatten()
    patch_dst_g = patch_dst[:, :, 1].flatten()
    patch_dst_b = patch_dst[:, :, 2].flatten()
    patch_src_r = patch_srcc[:, :, 0].flatten()
    patch_src_g = patch_srcc[:, :, 1].flatten()
    patch_src_b = patch_srcc[:, :, 2].flatten()

    # computer the sum of each pix of patch
    i = 0
    length = patch_dst_r.shape[0]
    cnt = 0
    while i <= length - 1:
        if (patch_dst_r[i] != 0.0 and patch_dst_g[i] != 0.9999 and
                patch_dst_b[i] != 0.0):
            # ignore unfilled pixels 
            cnt += (patch_dst_r[i] - patch_src_r[i]) ** 2
            cnt += (patch_dst_g[i] - patch_src_g[i]) ** 2
            cnt += (patch_dst_b[i] - patch_src_b[i]) ** 2
        i += 1

    return cnt


# find the soure patch that target patch will copy from
# using sum of squared differences as standard
def get_exemplar_patch(patch, x, y, image, patch_size):
    p = patch_size // 2
    x_boundary = image.shape[0]
    y_boundary = image.shape[1]

    # source patch that will copy from
    img_copy = image[p: x_boundary - p + 1, p: y_boundary - p + 1]

    # locations of the unfilled region
    filled_r = np.where(img_copy[:, :, 1] != 0.9999)

    xx = filled_r[0]
    yy = filled_r[1]

    i = 0
    min_ssd = np.inf
    while i < len(xx) - 1:
        exemplar_patch = get_patch(xx[i] + p, yy[i] + p, image, patch_size)

        # check weather they are the same patch 
        if (exemplar_patch.shape[0] == patch_size and
                exemplar_patch.shape[1] == patch_size):
            if ((xx[i] + p) != x and (yy[i] + p) != y and
                    np.where(exemplar_patch[:, :, 1] == 0.9999)[0].shape[0] == 0):

                # computer the ssd between two patches
                ssd = patch_ssd(patch, exemplar_patch)

                if ssd < min_ssd:
                    best_patch = exemplar_patch
                    best_x = xx[i] + p
                    best_y = yy[i] + p
                    min_ssd = ssd
        i += 1

    return best_patch, best_x, best_y


# copy pix value from source patch target patch
def copy_patch(patch_dst, patch_src):
    # find locations of unfilled pixels
    unfilled_pixels = np.where(patch_dst[:, :, 1] == 0.9999)

    unfilled_x = unfilled_pixels[0]
    unfilled_y = unfilled_pixels[1]

    i = 0
    while i <= len(unfilled_x) - 1:
        patch_dst[unfilled_x[i]][unfilled_y[i]] = \
            patch_src[unfilled_x[i]][unfilled_y[i]]
        i += 1

    return patch_dst


# update confidence and mask value of target patch
# paste changed patch to image
def update_patch(x, y, patch, image, patch_size):
    p = patch_size // 2

    x_begin = x - p
    x_end = x + p + 1
    y_begin = y - p
    y_end = y + p + 1

    s = 0
    t = 0
    for i in range(x_begin, x_end):
        for j in range(y_begin, y_end):
            image[i, j] = patch[s, t]
            t += 1
        s += 1
        t = 0

    return image


# update confidence and mask parameters
# if the pix is already processed, set it to 1
# directly return updated parameters
def update_parameter(x, y, confidence, mask, patch_size):
    p = patch_size // 2

    x_begin = x - p
    x_end = x + p + 1
    y_begin = y - p
    y_end = y + p + 1

    for i in range(x_begin, x_end):
        for j in range(y_begin, y_end):
            confidence[i, j] = 1
            mask[i, j] = 1

    return confidence, mask


# do image inpainting
# the procedure is 
#   1 Extract the manually selected initial front.
#   2 Repeat until done:
#       2.a Identify the fill front, if zero, exit
#       2.b Compute priorities P (p)
#       2.c Find the patch with the maximum priority
#       2.d Find the exemplar that minimizes
#       2.e Copy image data
#   3 Update C(p)
def image_inpainting(img, input_mask, patch_size=9):
    unfilled_img = img / 255.0
    mask = input_mask / 255

    # couputer the gray scale of input image
    grayscale = (unfilled_img[:, :, 0] * .2125 +
                 unfilled_img[:, :, 1] * .7154 +
                 unfilled_img[:, :, 2] * .0721)

    # initialize confidence value
    # set p in mask area to 0, other to 1
    confidence = np.zeros(img.shape[0:2])
    confidence[np.where(mask != 0)] = 1

    # initialize image with mask
    unfilled_img[np.where(mask == 0)] = [0.0, 0.9999, 0.0]

    loop_cnt = 0
    while np.where(mask == 0)[0].any():
        # boundary
        fill_front = mask - erosion(mask, disk(1)) / 255

        # x and y coordinates of boundary
        fill_front_x = np.where(fill_front > 0)[0]
        fill_front_y = np.where(fill_front > 0)[1]

        # computer gradients
        dx = ndimage.sobel(grayscale, 0)
        dy = ndimage.sobel(grayscale, 1)

        # mark region to inpaint
        dx[np.where(mask == 0)] = 0.0
        dy[np.where(mask == 0)] = 0.0

        # compute normals
        nx = ndimage.sobel(mask, 0)
        ny = ndimage.sobel(mask, 1)

        # get the max priority patch
        max_priority_patch = get_max_priority_patch(fill_front_x, fill_front_y,
                                                    dy, -dx, -ny, nx,
                                                    confidence, patch_size)

        max_x = max_priority_patch[0]
        max_y = max_priority_patch[1]

        # the first patch that we need to process
        max_patch = get_patch(max_x, max_y, unfilled_img, patch_size)

        # get the best patch that is "nearest" to target patch
        source_patch = get_exemplar_patch(max_patch, max_x, max_y, unfilled_img,
                                          patch_size)

        # copy the pix value from soure patch to target patch 
        copied_patch = copy_patch(max_patch, source_patch[0])

        # paste changed patch to image and update mask and confidence
        unfilled_img = update_patch(max_x, max_y, copied_patch, unfilled_img,
                                    patch_size)

        # update parameters such as confidence and mask
        updated_parameters = update_parameter(max_x, max_y, confidence, mask,
                                              patch_size)
        confidence = updated_parameters[0]
        mask = updated_parameters[1]

        # print the total loop number up to now
        loop_cnt += 1
        print loop_cnt, ": From ", max_priority_patch[0:], 'to', source_patch[1:]

        # save the temperate image
        imsave("./result_image/result1.jpg", unfilled_img)


image_inpainting(img, mask)
