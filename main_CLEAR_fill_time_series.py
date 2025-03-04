"""
This is the official Python implementation of our paper
"CLEAR: A Novel Gap-filling Method for Optical Remote Sensing Images
Combining Class-based Linear Regression and Iterative Residual Compensation"

Author: Houcai GUO, PhD student at the University of Trento, Italy

E-mail: houcai.guo@unitn.it; guohoucai@qq.com

If you have any questions about our code or need my help in implementing CLEAR for your own research,
please feel free to contact me via email.
"""
import matplotlib.pyplot as plt
import numpy as np

from CLEAR_utils import *
from datetime import datetime

np.set_printoptions(suppress=True, precision=5)

"""
Parameters
"""
# Threshold for selecting reference images (T);
# Images with cloud cover (percentage) lower than the threshold are regarded as reference images;
# A larger value can improve accuracy but reduces computational efficiency;
# A value higher than 5 is recommended.
ref_cloud_threshold = 15

# Threshold for gap-filling;
# Images with cloud cover (percentage) higher than the threshold will not be filled.
fill_cloud_threshold = 75

# Number of used reference images (N), either an integer or "all";
# A larger value can improve accuracy but reduces computational time;
# A value higher than 10 is recommended.
ref_num = "all"

# Number of land-cover classes (K);
# A larger value can improve computational efficiency;
# A value higher than 12 is recommended.
class_num = 12

# Size of the local window for selecting similar pixels (w);
# A larger value can improve accuracy but reduces computational efficiency;
# A value higher than 61 is recommended.
win_size = 61

# Size for expanding the window
expand_size = 10

# Number of required similar pixels (S)
similar_num = 20

# Save the results or not
save = True

# Show each gap-filling result or not
show = False

# Number of spectral bands
band_num = 4

"""
Input and output files
"""
# stacked time-series images
images_path = r"CLEAR_data/Jacksonville/Jacksonville-real-images.jp2"
# stacked cloud masks
masks_path = r"CLEAR_data/Jacksonville/Jacksonville-real-masks.jp2"
# comma-separated day of year (DOY) txt file
DOYs_path = r"CLEAR_data/Jacksonville/Jacksonville-DOYs.txt"
# save path of CLEAR
CLEAR_save_path = r"CLEAR_data/Jacksonville/Jacksonville-real-CLEAR.jp2"

if __name__ == "__main__":
    t1 = datetime.now()
    """
    Read images, masks, and DOYs
    """
    images, images_profile = read_raster(images_path)
    row_num, col_num, _ = images.shape
    # (row_num, column_num, band_num * image_num) to (row_num, column_num, band_num, image_num)
    images = images.reshape((row_num, col_num, -1, band_num))
    images = images.transpose((0, 1, 3, 2))
    images = images.astype(np.float32)
    print(f"There are total {images.shape[3]} images.")

    masks, masks_profile = read_raster(masks_path)
    masks = np.where(masks == 1, True, False)

    DOYs = []
    DOY_strings = open(DOYs_path).readlines()[0].split(sep=",")
    for DOY_string in DOY_strings:
        DOYs.append(eval(DOY_string))
    DOYs = np.array(DOYs, dtype=np.int32)

    cloud_covers = []
    for image_idx in range(masks.shape[2]):
        mask = masks[:, :, image_idx]
        cloud_cover = np.count_nonzero(mask) / (mask.shape[0] * mask.shape[1]) * 100
        cloud_covers.append(cloud_cover)
    cloud_covers = np.array(cloud_covers, dtype=np.float32)

    """
    Gap-filling
    """
    filled_images = images.copy()
    filled_masks = masks.copy()

    """
    Select candidate images
    """
    candidate_images = []
    candidate_masks = []
    candidate_indices = []
    candidate_cloud_covers = []
    candidate_DOYs = []
    for image_idx in range(filled_images.shape[3]):
        mask = filled_masks[:, :, image_idx]
        cloud_cover = np.count_nonzero(mask) / (mask.shape[0] * mask.shape[1]) * 100
        if cloud_cover < ref_cloud_threshold:
            candidate_images.append(images[:, :, :, image_idx])
            candidate_masks.append(mask)
            candidate_indices.append(image_idx)
            candidate_cloud_covers.append(cloud_cover)
            candidate_DOY = DOYs[image_idx]
            candidate_DOYs.append(candidate_DOY)
    # shape = (row_num, col_num, band_num, candidate_num)
    candidate_images = np.stack(candidate_images, axis=3)
    # shape = (row_num, col_num, candidate_num)
    candidate_masks = np.stack(candidate_masks, axis=2)
    candidate_indices = np.array(candidate_indices)
    candidate_cloud_covers = np.array(candidate_cloud_covers)
    candidate_DOYs = np.array(candidate_DOYs)
    print(f"There are {candidate_images.shape[3]} candidate images.")

    """
    Fill gaps in candidate images using linear interpolation
    """
    print(f"Start linear interpolation to fill gaps in the candidate images.")
    candidate_images = linear_interpolation(images, filled_masks, candidate_images, candidate_masks, candidate_indices)

    """
    Mini Batch Kmeans classification
    """
    print(f"Start Mini Batch KMeans classification of the linearly interpolated candidate images.")
    class_map, cluster_centers = classify_reference_images_fast(candidate_images, class_num)

    """
    Gap-filling of the candidate images
    """
    filled_candidate_images = candidate_images.copy()
    for idx in range(candidate_images.shape[3]):
        image_idx = np.argmax(candidate_cloud_covers)
        candidate_DOY = candidate_DOYs[image_idx]
        candidate_cloud_cover = candidate_cloud_covers[image_idx]
        if candidate_cloud_cover > 0:
            print(f"\tStart to fill the image on DOY {candidate_DOY}, "
                  f"cloud cover {candidate_cloud_cover:.2f}%")
            candidate_image = candidate_images[:, :, :, image_idx]
            candidate_mask = candidate_masks[:, :, image_idx]
            class_map_1 = check_class_map_validity(class_map, cluster_centers, candidate_mask)
            # choose other candidate images
            if image_idx == 0:
                # candidate image that need to be filled is the first one
                other_candidate_images = candidate_images[:, :, :, 1:]
            elif image_idx == candidate_images.shape[3] - 1:
                # candidate image that need to be filled is the last one
                other_candidate_images = candidate_images[:, :, :, :-1]
            else:
                left = candidate_images[:, :, :, :image_idx]
                right = candidate_images[:, :, :, image_idx + 1:]
                other_candidate_images = np.concatenate([left, right], axis=3)

            filled_candidate_image = fill_single_image_fast(other_candidate_images,
                                                            candidate_image, candidate_mask,
                                                            class_map_1, class_num,
                                                            win_size, expand_size,
                                                            similar_num)[1]
            filled_candidate_images[:, :, :, image_idx] = filled_candidate_image
            candidate_cloud_covers[image_idx] = 0.0
    print(f"Finished gap-filling of the candidate images.")
    filled_images[:, :, :, candidate_indices] = filled_candidate_images
    filled_masks[:, :, candidate_indices] = False
    cloud_covers[candidate_indices] = 999

    """
    Gap-filling of other images
    """
    fill_flags = np.all(np.stack([cloud_covers >= ref_cloud_threshold,
                                  cloud_covers <= fill_cloud_threshold], axis=1), axis=1)
    fill_images_num = np.count_nonzero(fill_flags)
    for idx in range(fill_images_num):
        image_idx = np.argmin(cloud_covers)
        target_DOY = DOYs[image_idx]
        cloud_cover = cloud_covers[image_idx]
        target_image = filled_images[:, :, :, image_idx]
        target_mask = filled_masks[:, :, image_idx]

        print(f"Start to fill the image on DOY {target_DOY}, cloud cover {cloud_cover:.2f}%")
        t1_ = datetime.now()

        DOY_diffs = candidate_DOYs - target_DOY
        DOY_diffs_abs = np.abs(DOY_diffs)
        DOY_diffs_indices = np.argsort(DOY_diffs_abs)
        if ref_num == "all":
            indices = DOY_diffs_indices
        else:
            indices = DOY_diffs_indices[:ref_num]
        ref_images = candidate_images[:, :, :, indices]
        ref_masks = candidate_masks[:, :, indices]
        ref_cloud_covers = candidate_cloud_covers[indices]
        ref_indices = candidate_indices[indices]
        print(f"\tSelected {ref_images.shape[3]} reference images.")

        """
        Mini Batch Kmeans classification
        """
        print(f"\tStart Mini Batch KMeans classification of the gap-filled reference images.")
        class_map, cluster_centers = classify_reference_images_fast(ref_images, class_num)
        class_map = check_class_map_validity(class_map, cluster_centers, target_mask)

        """
        Fill the cloudy image
        """
        print(f"\tStart gap-filling of the target cloudy image.")
        reg_prediction, final_prediction = fill_single_image_fast(ref_images,
                                                                  target_image, target_mask,
                                                                  class_map, class_num,
                                                                  win_size, expand_size,
                                                                  similar_num)
        t2_ = datetime.now()
        time_span = t2_ - t1_
        total_minutes = time_span.total_seconds() / 60
        print(f"\tFinished the gap-filling of target cloudy image. Used {total_minutes:.2f} min.")

        final_prediction = final_prediction.astype(np.uint16)
        # write_raster(final_prediction, images_profile, CLEAR_save_path)

        """
        Show the results
        """
        if show:
            fig, axes = plt.subplots(1, 5, sharex=True, sharey=True)
            target_image_ma = np.ma.array(data=target_image.copy(),
                                          mask=np.stack([target_mask for i in range(band_num)], axis=2))
            axes[0].imshow(linear_pct_stretch_ma(color_composite_ma(target_image_ma, [3, 2, 1]), pct=2))
            axes[0].set_title("Cloudy image")
            axes[1].imshow(target_mask, cmap="gray")
            axes[1].set_title("Cloud mask")
            axes[2].imshow(class_map, cmap="tab20")
            axes[2].set_title("Land-cover classes")
            axes[3].imshow(linear_pct_stretch(color_composite(reg_prediction, [3, 2, 1]), pct=2))
            axes[3].set_title("Regression")
            axes[4].imshow(linear_pct_stretch(color_composite(final_prediction, [3, 2, 1]), pct=2))
            axes[4].set_title("CLEAR")
            plt.show()

        filled_images[:, :, :, image_idx] = final_prediction
        filled_masks[:, :, image_idx] = False
        cloud_covers[image_idx] = 999

    t2 = datetime.now()
    time_span = t2 - t1
    total_minutes = time_span.total_seconds() / 60
    print(f"Finished the gap-filling of time-series. Used {total_minutes:.2f} min.")
    write_raster(filled_images, images_profile, CLEAR_save_path)


