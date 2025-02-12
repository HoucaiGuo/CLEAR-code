import os
import rasterio
import numpy as np
from CLEAR_utils import *
from datetime import datetime

np.set_printoptions(suppress=True, precision=5)

site = "Omaha"
cloudy_date = "20220723"
cloud_threshold = 5
cloud_coverage = "low"

ref_num = 10
class_num = 8
win_size = 101
expand_size = 10
similar_num = 20

image_dir = f"data/{site}/all images"
mask_dir = f"data/{site}/all masks"
cloudy_image_path = rf"data/{site}/{cloud_coverage}/{site}-{cloudy_date}-{cloud_coverage}.tif"
cloud_mask_path = rf"data/{site}/{cloud_coverage}/{site}-{cloudy_date}-{cloud_coverage}-mask.tif"
CLEAR_path = rf"data/{site}/{cloud_coverage}/{site}-{cloudy_date}-{cloud_coverage}-n-{ref_num}-K-{class_num}-w-{win_size}-S-{similar_num}-CLEAR.tif"


def read_raster(raster_path):
    dataset = rasterio.open(raster_path)
    raster_profile = dataset.profile
    raster = dataset.read()
    raster = np.transpose(raster, (1, 2, 0))
    raster = raster.astype(np.dtype(raster_profile["dtype"]))

    return raster, raster_profile


def write_raster(raster, raster_profile, raster_path):
    raster_profile["dtype"] = str(raster.dtype)
    raster_profile["height"] = raster.shape[0]
    raster_profile["width"] = raster.shape[1]
    raster_profile["count"] = raster.shape[2]
    image = np.transpose(raster, (2, 0, 1))
    dataset = rasterio.open(raster_path, mode='w', **raster_profile)
    dataset.write(image)
    dataset.close()


def get_date(image_name):
    splited = image_name.split(sep="-")
    return splited[1][:8]


if __name__ == "__main__":
    t1 = datetime.now()

    """
    Read the cloudy image and the cloud mask
    """
    cloudy_image, cloudy_profile = read_raster(cloudy_image_path)
    cloudy_image = cloudy_image.astype(np.float32) / 10000.0
    cloud_mask, mask_profile = read_raster(cloud_mask_path)
    cloud_mask = np.where(cloud_mask == 1, True, False)

    """
    Read all available images
    """
    file_names = os.listdir(image_dir)
    file_names.sort(key=get_date)
    image_names = []
    for file_name in file_names:
        if file_name.endswith(".tif"):
            image_names.append(file_name)
    images = []
    masks = []
    dates = []
    for image_idx in range(len(image_names)):
        image_name = image_names[image_idx]
        date = get_date(image_name)
        dates.append(date)
        if date != cloudy_date:
            image_path = os.path.join(image_dir, f"{site}-{date}.tif")
            image, profile = read_raster(image_path)
            image = image.astype(np.float32) / 10000.0
            images.append(image)
            mask_path = os.path.join(mask_dir, f"{site}-{date}-mask.tif")
            mask = read_raster(mask_path)[0]
            mask = np.where(mask == 1, True, False)
            masks.append(mask)
        else:
            images.append(cloudy_image)
            masks.append(cloud_mask)
    images = np.stack(images, axis=3)
    masks = np.concatenate(masks, axis=2)
    cloud_mask = cloud_mask.squeeze()
    print(f"There are total {images.shape[3]} images.")

    """
    Select reference images
    """
    candidate_images = []
    candidate_masks = []
    candidate_indices = []
    candidate_cloud_coverages = []
    candidate_DOYs = []
    for image_idx in range(images.shape[3]):
        candidate_mask = masks[:, :, image_idx]
        candidate_cloud_coverage = np.count_nonzero(candidate_mask) / \
                                   (candidate_mask.shape[0] * candidate_mask.shape[1]) * 100
        # cloud cover lower than the threshold
        if candidate_cloud_coverage <= cloud_threshold:
            candidate_images.append(images[:, :, :, image_idx])
            candidate_masks.append(candidate_mask)
            candidate_indices.append(image_idx)
            candidate_cloud_coverages.append(candidate_cloud_coverage)
            candidate_date = dates[image_idx]
            candidate_date_fmt = f"{candidate_date[:4]}-{candidate_date[4:6]}-{candidate_date[6:8]}"
            candidate_DOY = datetime.strptime(candidate_date_fmt, "%Y-%m-%d").timetuple().tm_yday
            candidate_DOYs.append(candidate_DOY)
    # shape = (row_num, col_num, band_num, ref_num)
    candidate_images = np.stack(candidate_images, axis=3)
    # shape = (row_num, col_num, ref_num)
    candidate_masks = np.stack(candidate_masks, axis=2)
    candidate_indices = np.array(candidate_indices)
    candidate_cloud_coverages = np.array(candidate_cloud_coverages)
    candidate_DOYs = np.array(candidate_DOYs)
    print(f"There are {candidate_images.shape[3]} candidate images.")

    # time interval
    cloudy_date_fmt = f"{cloudy_date[:4]}-{cloudy_date[4:6]}-{cloudy_date[6:8]}"
    cloudy_DOY = datetime.strptime(cloudy_date_fmt, "%Y-%m-%d").timetuple().tm_yday
    DOY_diffs = candidate_DOYs - cloudy_DOY
    DOY_diffs_abs = np.abs(DOY_diffs)
    DOY_diffs_indices = np.argsort(DOY_diffs_abs)
    indices = DOY_diffs_indices[:ref_num]
    ref_images = candidate_images[:, :, :, indices]
    ref_masks = candidate_masks[:, :, indices]
    ref_cloud_coverages = candidate_cloud_coverages[indices]
    ref_indices = candidate_indices[indices]
    print(f"Selected {ref_images.shape[3]} reference images.")

    """
    Fill gaps in reference images using linear interpolation
    """
    print(f"Start linear interpolation to fill gaps in the reference images.")
    interp_images = linear_interpolation(images, masks, ref_images, ref_masks, ref_indices)

    """
    Kmeans classification
    """
    print(f"Start KMeans classification of the linearly interpolated reference images.")
    class_map, cluster_centers = classify_reference_images(ref_images, class_num)

    """
    Gap-filling of reference images
    """
    filled_ref_images = ref_images.copy()
    for idx in range(ref_images.shape[3]):
        image_idx = np.argmax(ref_cloud_coverages)
        if ref_cloud_coverages[image_idx] > 0:
            print(f"Start gap-filling of the {image_idx}th reference image, "
                  f"cloud coverage: {ref_cloud_coverages[image_idx]:.2f}%.")
            ref_image = ref_images[:, :, :, image_idx]
            ref_mask = ref_masks[:, :, image_idx]
            class_map_1 = check_class_map_validity(class_map, cluster_centers, ref_mask)
            # choose other interp images
            if image_idx == 0:
                # image that need to be filled is the first one
                other_ref_images = interp_images[:, :, :, 1:]
            elif image_idx == interp_images.shape[3] - 1:
                # image that need to be filled is the last one
                other_ref_images = interp_images[:, :, :, :-1]
            else:
                left = interp_images[:, :, :, :image_idx]
                right = interp_images[:, :, :, image_idx + 1:]
                other_ref_images = np.concatenate([left, right], axis=3)

            filled_ref_image = fill_single_image(other_ref_images,
                                                 ref_image, ref_mask,
                                                 class_map_1, class_num,
                                                 win_size, expand_size,
                                                 similar_num)[1]
            filled_ref_images[:, :, :, image_idx] = filled_ref_image
            ref_cloud_coverages[image_idx] = 0
    print(f"Finished the gap-filling of reference images.")

    """
    Kmeans classification of the gap-filled reference images
    """
    print(f"Start KMeans classification of the gap-filled reference images.")
    class_map, cluster_centers = classify_reference_images(filled_ref_images, class_num)
    class_map = check_class_map_validity(class_map, cluster_centers, cloud_mask)

    """
    Fill the cloudy image
    """
    print(f"Start gap-filling of the target cloudy image.")
    reg_prediction, final_prediction, residuals, inter_images, inter_residuals, inter_cloud_masks = fill_single_image(
        filled_ref_images,
        cloudy_image, cloud_mask,
        class_map, class_num,
        win_size, expand_size,
        similar_num)
    t2 = datetime.now()
    time_span = t2 - t1
    total_minutes = time_span.total_seconds() / 60
    print(f"Finished the gap-filling of target cloudy image. Used {total_minutes:.2f} min.")
    write_raster(final_prediction, cloudy_profile, CLEAR_path)
