import os
import numpy as np
import rasterio
from Fast_CLEAR_utils import *
from functions import read_raster, write_raster
from datetime import datetime

np.set_printoptions(suppress=True, precision=5)

ref_cloud_threshold = 5
class_num = 10
PCA_pct = 95
sample_num = 2000
similar_num = 20

image_dir = rf"F:\Code\Gap_Filling\CLEAR\github\test data\Site2_Apr_Oct_images"
mask_dir = rf"F:\Code\Gap_Filling\CLEAR\github\test data\Site2_Apr_Oct_masks"
cloudy_name = "Omaha-20220723-low.tif"
Fast_CLEAR_path = rf"F:\Code\Gap_Filling\CLEAR\github\test data\Omaha-20220723-low-Fast-CLEAR.tif"


def get_date(image_name):
    splited = image_name.split(sep="-")
    return splited[1][:8]


if __name__ == "__main__":
    file_names = os.listdir(image_dir)
    file_names.sort(key=get_date)
    image_names = []
    for file_name in file_names:
        if file_name.endswith(".tif"):
            image_names.append(file_name)
    file_names = os.listdir(mask_dir)
    file_names.sort(key=get_date)
    mask_names = []
    for file_name in file_names:
        if file_name.endswith(".tif"):
            mask_names.append(file_name)

    images = []
    masks = []
    for image_idx in range(len(image_names)):
        image_name = image_names[image_idx]
        mask_name = mask_names[image_idx]
        date = get_date(image_name)
        image_path = os.path.join(image_dir, image_name)
        mask_path = os.path.join(mask_dir, mask_name)
        image_dataset = rasterio.open(image_path)
        image_profile = image_dataset.profile
        image = image_dataset.read()
        image = np.transpose(image, (1, 2, 0))
        image = image.astype(np.float32)
        images.append(image)
        mask_dataset = rasterio.open(mask_path)
        mask_profile = mask_dataset.profile
        mask = mask_dataset.read()
        mask = np.transpose(mask, (1, 2, 0))
        mask = np.where(mask == 1, True, False)
        masks.append(mask)
        if image_name == cloudy_name:
            cloudy_image = image.copy()
            cloud_mask = mask.copy()
            cloud_mask = cloud_mask.squeeze()
            cloudy_profile = image_profile.copy()
    images = np.stack(images, axis=3)
    masks = np.concatenate(masks, axis=2)
    print(f"There are total {images.shape[3]} images.")

    """
    Select reference images.
    """
    ref_images = []
    ref_masks = []
    ref_indices = []
    ref_cloud_coverages = []
    for image_idx in range(images.shape[3]):
        mask = masks[:, :, image_idx]
        cloud_coverage_pct = np.count_nonzero(mask) / \
                             (mask.shape[0] * mask.shape[1]) * 100
        if cloud_coverage_pct <= ref_cloud_threshold:
            image = images[:, :, :, image_idx]
            ref_images.append(image)
            ref_masks.append(mask)
            ref_indices.append(image_idx)
            ref_cloud_coverages.append(cloud_coverage_pct)

    # shape = (row_num, col_num, band_num, ref_num)
    ref_images = np.stack(ref_images, axis=3)
    # shape = (row_num, col_num, ref_num)
    ref_masks = np.stack(ref_masks, axis=2)
    print(f"Selected {ref_images.shape[3]} reference images.")

    """
    Fill gaps in reference images using linear interpolation.
    """
    print(f"Start linear interpolation to fill gaps in the reference images.")
    interp_images = linear_interpolation(images, masks, ref_images, ref_masks, ref_indices)

    """
    PCA transformation and MiniBatchKmeans classification.
    """
    ref_PC_images = PCA_transform(ref_images, PCA_pct)
    print(ref_PC_images.shape)
    class_map, class_centers = MiniBatchKMeans_classification(ref_PC_images, class_num)

    """
    Gap-filling of reference images.
    """
    filled_ref_images = interp_images.copy()
    for idx in range(interp_images.shape[3]):
        image_idx = np.argmax(ref_cloud_coverages)
        time_series_idx = ref_indices[image_idx]
        if ref_cloud_coverages[image_idx] > 0:
            print(f"\tStart gap-filling of the {image_idx}th reference image, "
                  f"cloud coverage: {ref_cloud_coverages[image_idx]:.2f}%.")
            ref_image = ref_images[:, :, :, image_idx]
            ref_mask = ref_masks[:, :, image_idx]
            filled_ref_image = fill_single_image_fast(ref_PC_images, ref_image, ref_mask,
                                                      class_map, sample_num, similar_num)[1]
            filled_ref_images[:, :, :, image_idx] = filled_ref_image
            ref_cloud_coverages[image_idx] = 0
    print(f"Finished gap-filling of the reference images.")

    """
    Fill the cloudy image
    """
    ref_PC_images = PCA_transform(filled_ref_images, PCA_pct)
    class_map, class_centers = MiniBatchKMeans_classification(ref_PC_images, class_num)

    print(f"Start gap-filling of the target cloudy images.")
    t1 = datetime.now()
    reg_prediction, final_prediction = fill_single_image_fast(ref_PC_images, cloudy_image, cloud_mask.squeeze(),
                                                              class_map, sample_num, similar_num)
    t2 = datetime.now()
    time_span = t2 - t1
    total_minutes = time_span.total_seconds() / 60
    print(f"Finished the gap-filling of target cloudy image. Used {total_minutes:.2f} min.")
    write_raster(final_prediction, cloudy_profile, Fast_CLEAR_path)
