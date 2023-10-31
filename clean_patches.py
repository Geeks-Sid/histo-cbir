import cv2
import numpy as np
from skimage import io
from skimage.color import hed2rgb, rgb2hed
from skimage.color.colorconv import rgb2hsv
from skimage.exposure import rescale_intensity

patch_size = 256
intensity_thresh = 225
intensity_thresh_b = 50

patch_array = np.asarray(
    slide.read_region((x, y), patch_level, (patch_size, patch_size)).convert("RGB")
)
# io.imshow(patch_array)
# plt.show()
count_white_pixels = np.where(
    np.logical_and(
        patch_array[:, :, 0] > intensity_thresh,
        patch_array[:, :, 1] > intensity_thresh,
        patch_array[:, :, 2] > intensity_thresh,
    )
)[0]
percent_pixels = len(count_white_pixels) / (patch_size * patch_size)

ihc_hed = rgb2hed(patch_array)
patch_hsv_1 = cv2.cvtColor(patch_array, cv2.COLOR_RGB2HSV)[:, :, 0]
e = rescale_intensity(
    ihc_hed[:, :, 1],
    out_range=(0, 255),
    in_range=(0, np.percentile(ihc_hed[:, :, 1], 99)),
)
if (
    len(e[e < 50]) / (patch_size * patch_size) > 0.9
    or (len(np.where((patch_hsv_1[patch_hsv_1 < 128]))[0]) / (patch_size * patch_size))
    > 0.95
):
    new_ds_arr.remove(list(j))
    # io.imsave(os.path.join('/home/shubham/clam_workstation/tcga_gbm_lgg_20x_final/all_patches_removed_10x/'+slide_name)+'/'+slide_name +'_' +str(x) +'_'+str(y)+'.jpg',patch_array)
    continue


intensity_thresh_b_1 = 128
count_white_pixels_b = np.where(
    np.logical_and(
        patch_array[:, :, 0] < intensity_thresh_b_1,
        patch_array[:, :, 1] < intensity_thresh_b_1,
        patch_array[:, :, 2] < intensity_thresh_b_1,
    )
)[0]
percent_pixel_b = len(count_white_pixels_b) / (patch_size * patch_size)
patch_hsv_2 = cv2.cvtColor(patch_array, cv2.COLOR_RGB2HSV)[:, :, 1]
percent_pixel_2 = len(np.where((patch_hsv_2 < intensity_thresh_b))[0]) / (
    patch_size * patch_size
)
patch_hsv_3 = cv2.cvtColor(patch_array, cv2.COLOR_RGB2HSV)[:, :, 2]
percent_pixel_3 = len(np.where((patch_hsv_3 > intensity_thresh))[0]) / (
    patch_size * patch_size
)


if percent_pixel_2 > 0.96 or np.mean(patch_hsv_2) < 5 or percent_pixel_3 > 0.96:
    if not percent_pixel_2 < 0.25:
        new_ds_arr.remove(list(j))
        # io.imsave(os.path.join('/cbica/home/innanis/comp_space/clam_idh/tcga_gbm_lgg_'+patching+'_'+base_mag+'/all_patches_removed/',slide_name)+'/'+slide_name +'_' +str(x) +'_'+str(y)+'.tiff',patch_array)
    elif (
        (percent_pixel_2 > 0.9 and percent_pixel_3 > 0.9)
        or percent_pixel_b > 0.9
        or percent_pixels > 0.65
    ):
        new_ds_arr.remove(list(j))
