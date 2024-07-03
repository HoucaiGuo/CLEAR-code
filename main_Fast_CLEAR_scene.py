import os

import matplotlib.pyplot as plt
import rasterio
import numpy as np
from os.path import join
from datetime import datetime
from Fast_CLEAR_utils import *

from functions import *

np.set_printoptions(suppress=True, precision=5)

image_folder = r"data/Omaha-Scene/R112_T15TTF_images"
temp_folder = r"data/Omaha-Scene/Temp"
save_folder = r"data/Omaha-Scene/Results"

image_suffix = ".jp2"
mask_suffix = ".tif"

target_DOYs = [164, 184, 209, 229, 234, 259]
scene_num = len(target_DOYs)

ref_cloud_threshold = 5
fill_cloud_threshold = 80

PCA_pct = 95
class_num = 10
sample_num = 2000
similar_num = 20
min_val = 0.0
max_val = 10000.0

image_row_size = 10980
image_col_size = 10980
patch_row_size = 1098
patch_col_size = 1098
overlap_row_size = 0
overlap_col_size = 0


def get_DOY(name):
    if "mask" in name:
        return eval(name.split(sep="-")[0])
    else:
        return eval(name.split(sep=".")[0])


if __name__ == "__main__":
    """
    Calculate number of patches
    """
    patch_row_num = int(np.round(image_row_size / patch_row_size))
    patch_col_num = int(np.round(image_col_size / patch_col_size))
    patch_total_num = patch_row_num * patch_col_num
    print(f"There are total {patch_total_num} ({patch_row_num} * {patch_col_num}) patches.")

    """
    Get image names
    """
    file_names = os.listdir(image_folder)
    image_names = []
    DOYs = []
    for file_idx in range(len(file_names)):
        file_name = file_names[file_idx]
        if file_name.endswith(image_suffix):
            image_names.append(file_name)
            DOY = eval(file_name.split(sep=".")[0])
            DOYs.append(DOY)
    image_names.sort(key=get_DOY)
    DOYs = np.array(DOYs, dtype=np.int32)
    print(DOYs)

    """
    Fast CLEAR image-by-image
    """
    filled_scene_num = 0
    cloud_coverages = np.load(join(temp_folder, "cloud-coverages.npy"))
    for target_DOY in target_DOYs:
        T1 = datetime.now()
        print(f"Start the gap-filling of DOY {target_DOY}.")

        txt_file_path = join(temp_folder, f"{target_DOY}-time.txt")

        scene_idx = (DOYs == target_DOY).nonzero()[0][0]
        scene_cloud_coverages = cloud_coverages[:, :, scene_idx]
        print(scene_cloud_coverages)

        scene_cloud_coverages = np.ma.array(data=scene_cloud_coverages.copy(),
                                            mask=[scene_cloud_coverages == 0.0])
        if np.ma.min(scene_cloud_coverages) > fill_cloud_threshold:
            print("Skip!")
            continue

        cloudy_patch_num = np.count_nonzero(~scene_cloud_coverages.mask)

        """
        Patch-by-Patch
        """
        filled_patch_num = 0
        for cloudy_idx in range(cloudy_patch_num):
            patch_idx = np.ma.argmin(scene_cloud_coverages)
            patch_row_idx, patch_col_idx = np.unravel_index(patch_idx, shape=scene_cloud_coverages.shape)
            patch_cloud_coverage = scene_cloud_coverages[patch_row_idx, patch_col_idx]
            print(f"\tStart the gap-filling of Patch {target_DOY}-{patch_row_idx}-{patch_col_idx}, "
                  f"cloud coverage: {patch_cloud_coverage:.2f}%")
            patch_folder = join(temp_folder, f"{patch_row_idx}-{patch_col_idx}")
            save_path = join(patch_folder, f"{target_DOY}-{patch_row_idx}-{patch_col_idx}-filled" + image_suffix)
            if os.path.exists(save_path):
                filled_patch_num += 1
                print(f"\t\tPatch {target_DOY}-{patch_row_idx}-{patch_col_idx} has been filled. "
                      f"({filled_patch_num} / {cloudy_patch_num})")

                scene_cloud_coverages.data[patch_row_idx, patch_col_idx] = 0.0
                scene_cloud_coverages.mask[patch_row_idx, patch_col_idx] = True
                continue

            patch_cloud_coverages = cloud_coverages[patch_row_idx, patch_col_idx, :]

            # current patch is enough for gap-filling
            if patch_cloud_coverage <= fill_cloud_threshold:
                use_neighbor_patch = False
                images = []
                masks = []
                for image_idx in range(len(image_names)):
                    DOY = DOYs[image_idx]
                    mask_name = f"{DOY}-{patch_row_idx}-{patch_col_idx}-mask" + mask_suffix
                    mask_path = join(patch_folder, mask_name)
                    mask_dataset = rasterio.open(mask_path)
                    mask = mask_dataset.read()
                    mask = np.transpose(mask, (1, 2, 0))

                    filled_name = f"{DOY}-{patch_row_idx}-{patch_col_idx}-filled" + image_suffix
                    filled_path = join(patch_folder, filled_name)
                    # if this patch has been filled as a reference image
                    if os.path.exists(filled_path) and \
                            cloud_coverages[patch_row_idx, patch_col_idx, image_idx] <= ref_cloud_threshold:
                        image_name = filled_name
                        mask[mask == 1] = 0
                        print(f"\t\t\tPatch {DOY}-{patch_row_idx}-{patch_col_idx} has been filled.")
                    else:
                        image_name = f"{DOY}-{patch_row_idx}-{patch_col_idx}" + image_suffix
                    image_path = join(patch_folder, image_name)
                    image_dataset = rasterio.open(image_path)
                    patch_profile = image_dataset.profile
                    image = image_dataset.read()
                    image = np.transpose(image, (1, 2, 0))
                    image = image.astype(np.float32)
                    images.append(image)
                    mask = np.where(mask == 1, True, False)
                    masks.append(mask)
                images = np.stack(images, axis=3)
                cloud_masks = np.concatenate(masks, axis=2)
            # need neighboring filled patch to assist the gap-filling
            else:
                """
                同时读取images和nei images,判断合并在一起的云量，以及是否filled
                """
                use_neighbor_patch = True
                # search for a filled neighboring patch
                expected_mask = scene_cloud_coverages.data == 0.0
                expected_row_indices, expected_col_indices = expected_mask.nonzero()
                distances = np.sqrt(np.square(expected_row_indices - patch_row_idx) +
                                    np.square(expected_col_indices - patch_col_idx))
                nei_idx = np.argmin(distances)
                nei_row_idx = expected_row_indices[nei_idx]
                nei_col_idx = expected_col_indices[nei_idx]
                print(f"\t\tUse neighboring Patch {target_DOY}-{nei_row_idx}-{nei_col_idx} to assist the gap-filling.")
                nei_patch_folder = join(temp_folder, f"{nei_row_idx}-{nei_col_idx}")

                images = []
                masks = []
                nei_images = []
                nei_masks = []
                for image_idx in range(len(image_names)):
                    DOY = DOYs[image_idx]
                    mask_name = f"{DOY}-{patch_row_idx}-{patch_col_idx}-mask" + mask_suffix
                    mask_path = join(patch_folder, mask_name)
                    mask_dataset = rasterio.open(mask_path)
                    mask = mask_dataset.read()
                    mask = np.transpose(mask, (1, 2, 0))
                    nei_mask_name = f"{DOY}-{nei_row_idx}-{nei_col_idx}-mask" + mask_suffix
                    nei_mask_path = join(nei_patch_folder, nei_mask_name)
                    nei_mask_dataset = rasterio.open(nei_mask_path)
                    nei_mask = nei_mask_dataset.read()
                    nei_mask = np.transpose(nei_mask, (1, 2, 0))

                    filled_name = f"{DOY}-{patch_row_idx}-{patch_col_idx}-filled" + image_suffix
                    filled_path = join(patch_folder, filled_name)
                    nei_filled_name = f"{DOY}-{nei_row_idx}-{nei_col_idx}-filled" + image_suffix
                    nei_filled_path = join(nei_patch_folder, nei_filled_name)

                    # neighboring patch is clear
                    if DOY == target_DOY and cloud_coverages[nei_row_idx, nei_col_idx, scene_idx] == 0:
                        image_name = f"{DOY}-{patch_row_idx}-{patch_col_idx}" + image_suffix
                        nei_image_name = f"{DOY}-{nei_row_idx}-{nei_col_idx}" + image_suffix
                        print(f"\t\t\tPatch {DOY}-{nei_row_idx}-{nei_col_idx} is clear.")
                    # neighboring patch has been filled
                    elif DOY == target_DOY and cloud_coverages[nei_row_idx, nei_col_idx, scene_idx] != 0:
                        image_name = f"{DOY}-{patch_row_idx}-{patch_col_idx}" + image_suffix
                        nei_image_name = nei_filled_name
                        nei_mask[nei_mask == 1] = 0
                        print(f"\t\t\tPatch {DOY}-{nei_row_idx}-{nei_col_idx} has been filled.")
                    # both patch have been filled as reference image
                    elif os.path.exists(filled_path) and os.path.exists(nei_filled_path) \
                            and cloud_coverages[patch_row_idx, patch_col_idx, image_idx] <= ref_cloud_threshold \
                            and cloud_coverages[nei_row_idx, nei_col_idx, image_idx] <= ref_cloud_threshold:
                        # image_name = f"{DOY}-{patch_row_idx}-{patch_col_idx}" + image_suffix
                        image_name = filled_name
                        mask[mask == 1] = 0
                        nei_image_name = nei_filled_name
                        nei_mask[nei_mask == 1] = 0
                        print(f"\t\t\tPatch {DOY}-{patch_row_idx}-{patch_col_idx} and neighboring "
                              f"Patch {DOY}-{nei_row_idx}-{nei_col_idx} have been filled.")
                    else:
                        image_name = f"{DOY}-{patch_row_idx}-{patch_col_idx}" + image_suffix
                        nei_image_name = f"{DOY}-{nei_row_idx}-{nei_col_idx}" + image_suffix
                    image_path = join(patch_folder, image_name)
                    image_dataset = rasterio.open(image_path)
                    patch_profile = image_dataset.profile
                    image = image_dataset.read()
                    image = np.transpose(image, (1, 2, 0))
                    image = image.astype(np.float32)
                    images.append(image)
                    nei_image_path = join(nei_patch_folder, nei_image_name)
                    nei_image_dataset = rasterio.open(nei_image_path)
                    nei_image = nei_image_dataset.read()
                    nei_image = np.transpose(nei_image, (1, 2, 0))
                    nei_image = nei_image.astype(np.float32)
                    nei_images.append(nei_image)
                    mask = np.where(mask == 1, True, False)
                    nei_mask = np.where(nei_mask == 1, True, False)
                    masks.append(mask)
                    nei_masks.append(nei_mask)

                images = np.stack(images, axis=3)
                cloud_masks = np.concatenate(masks, axis=2)
                nei_images = np.stack(nei_images, axis=3)
                nei_cloud_masks = np.concatenate(nei_masks, axis=2)
                # concatenate two patches
                if images.shape[0] == nei_images.shape[0]:
                    images = np.concatenate([images, nei_images], axis=1)
                    cloud_masks = np.concatenate([cloud_masks, nei_cloud_masks], axis=1)
                    axis = 1
                elif images.shape[1] == nei_images.shape[1]:
                    images = np.concatenate([images, nei_images], axis=0)
                    cloud_masks = np.concatenate([cloud_masks, nei_cloud_masks], axis=0)
                    axis = 0

            """
            Select reference images
            """
            ref_images = []
            ref_masks = []
            ref_indices = []
            # ref_cloud_coverages = []
            for image_idx in range(images.shape[3]):
                mask = cloud_masks[:, :, image_idx]
                cloud_coverage = np.count_nonzero(mask) / \
                                 (mask.shape[0] * mask.shape[1]) * 100
                if cloud_coverage <= ref_cloud_threshold:
                    ref_image = images[:, :, :, image_idx]
                    ref_images.append(ref_image)
                    ref_masks.append(mask)
                    ref_indices.append(image_idx)
                    # ref_cloud_coverages.append(cloud_coverage)
            ref_images = np.stack(ref_images, axis=3)
            ref_masks = np.stack(ref_masks, axis=2)
            ref_DOYs = DOYs[ref_indices]
            print(f"\t\tSelected {ref_images.shape[3]} reference images. "
                  f"{ref_DOYs}")
            # print(ref_DOYs)
            # print(np.count_nonzero(ref_masks))

            # all reference images have been filled
            if np.count_nonzero(ref_masks) == 0:
                filled_ref_images = ref_images
                print("\t\tAll reference images have been filled.")
            # reference images have not been filled
            else:
                """
                Fill gaps in reference images using linear interpolation.
                """
                interp_images = linear_interpolation(images, cloud_masks,
                                                     ref_images, ref_masks, ref_indices,
                                                     min_val, max_val)
                print(f"\t\tFinished linear interpolation.")

                """
                PCA transformation and MiniBatchKmeans classification.
                """
                ref_PC_images = PCA_transform(ref_images, PCA_pct)
                class_map, class_centers = MiniBatchKMeans_classification(ref_PC_images, class_num)

                """
                Gap-filling of reference images.
                """
                filled_ref_images = interp_images.copy()
                for ref_idx in range(interp_images.shape[3]):
                    ref_image = ref_images[:, :, :, ref_idx]
                    ref_mask = ref_masks[:, :, ref_idx]
                    # current reference image has not been filled
                    if np.count_nonzero(ref_mask) != 0:
                        class_map = check_class_map_validity(class_map, class_centers, ref_mask)
                        filled_ref_image = fill_single_image_fast(ref_PC_images,
                                                                  ref_image, ref_mask,
                                                                  class_map,
                                                                  sample_num, similar_num,
                                                                  min_val, max_val)[1]
                        filled_ref_images[:, :, :, ref_idx] = filled_ref_image
                        print(f"\t\t\tFinished gap-filling of the {ref_idx}th reference image.")
                print(f"\t\tFinished gap-filling of the reference images.")

                # save the gap-filled reference images
                if not use_neighbor_patch:
                    for ref_idx in range(filled_ref_images.shape[3]):
                        filled_ref_image = filled_ref_images[:, :, :, ref_idx]
                        ref_DOY = ref_DOYs[ref_idx]
                        ref_name = f"{ref_DOY}-{patch_row_idx}-{patch_col_idx}-filled" + image_suffix
                        # if use_neighbor_patch:
                        #     row_size = patch_profile["height"]
                        #     col_size = patch_profile["width"]
                        #     if axis == 0:
                        #         filled_ref_image = filled_ref_image[:row_size, :, :]
                        #     elif axis == 1:
                        #         filled_ref_image = filled_ref_image[:, :col_size, :]
                        filled_ref_image = np.transpose(filled_ref_image, (2, 0, 1))
                        ref_save_path = join(patch_folder, ref_name)
                        filled_ref_dataset = rasterio.open(ref_save_path, mode='w', **patch_profile)
                        filled_ref_dataset.write(filled_ref_image)
                        filled_ref_dataset.close()

            if patch_cloud_coverage <= ref_cloud_threshold:
                filled_patch_num += 1
                print(f"\t\tPatch {target_DOY}-{patch_row_idx}-{patch_col_idx} "
                      f"has been filled as a reference image. "
                      f"({filled_patch_num} / {cloudy_patch_num})")
            else:
                cloudy_image = images[:, :, :, scene_idx]
                cloud_mask = cloud_masks[:, :, scene_idx].squeeze()
                t1 = datetime.now()
                ref_PC_images = PCA_transform(filled_ref_images, PCA_pct)
                class_map, class_centers = MiniBatchKMeans_classification(ref_PC_images, class_num)

                class_map = check_class_map_validity(class_map, class_centers, cloud_mask)
                final_prediction = fill_single_image_fast(ref_PC_images,
                                                          cloudy_image, cloud_mask,
                                                          class_map,
                                                          sample_num, similar_num,
                                                          min_val, max_val)[1]
                t2 = datetime.now()
                time_span = t2 - t1
                total_minutes = time_span.total_seconds() / 60
                filled_patch_num += 1
                print(f"\tFinished the gap-filling of Patch {target_DOY}-{patch_row_idx}-{patch_col_idx}. "
                      f"Used {total_minutes:.2f} min. "
                      f"({filled_patch_num} / {cloudy_patch_num})")
                if use_neighbor_patch:
                    row_size = patch_profile["height"]
                    col_size = patch_profile["width"]
                    # print(f"{row_size}, {col_size}")
                    if axis == 0:
                        final_prediction = final_prediction[:row_size, :, :]
                    elif axis == 1:
                        final_prediction = final_prediction[:, :col_size, :]
                final_prediction = np.transpose(final_prediction, (2, 0, 1))
                # save_path = join(patch_folder, f"{target_DOY}-{patch_row_idx}-{patch_col_idx}-filled" + image_suffix)
                filled_dataset = rasterio.open(save_path, mode='w', **patch_profile)
                filled_dataset.write(final_prediction)
                filled_dataset.close()

                if os.path.exists(txt_file_path):
                    txt_file = open(txt_file_path, mode="a", encoding="utf-8")
                else:
                    txt_file = open(txt_file_path, mode="x", encoding="utf-8")
                txt_file.write(f"{patch_row_idx}\t{patch_col_idx}\t{total_minutes:.2f}\n")
                txt_file.close()

            scene_cloud_coverages.data[patch_row_idx, patch_col_idx] = 0.0
            scene_cloud_coverages.mask[patch_row_idx, patch_col_idx] = True

        T2 = datetime.now()
        time_span = T2 - T1
        total_minutes = time_span.total_seconds() / 60
        filled_scene_num += 1
        print(f"Finished the gap-filling of DOY {target_DOY}. "
              f"Used {total_minutes:.2f} min. "
              f"({filled_scene_num} / {scene_num})")

        """
        Mosaic and save the result
        """
        scene_name = f"{target_DOY}" + image_suffix
        scene_path = join(image_folder, scene_name)
        scene_dataset = rasterio.open(scene_path)
        scene_profile = scene_dataset.profile
        scene_filled = np.zeros(shape=(scene_profile["height"], scene_profile["width"], scene_profile["count"]),
                                dtype=np.float32)
        row_start = 0
        for patch_row_idx in range(patch_row_num):
            col_start = 0
            for patch_col_idx in range(patch_col_num):
                patch_folder = join(temp_folder, f"{patch_row_idx}-{patch_col_idx}")
                if cloud_coverages[patch_row_idx, patch_col_idx, scene_idx] == 0:
                    patch_name = f"{target_DOY}-{patch_row_idx}-{patch_col_idx}" + image_suffix
                else:
                    patch_name = f"{target_DOY}-{patch_row_idx}-{patch_col_idx}-filled" + image_suffix
                # print(patch_name)
                patch_path = join(patch_folder, patch_name)
                patch_dataset = rasterio.open(patch_path)
                patch = patch_dataset.read()
                patch = np.transpose(patch, (1, 2, 0))
                patch = patch.astype(np.float32)
                scene_filled[row_start:row_start + patch.shape[0],
                             col_start:col_start + patch.shape[1], :] = patch
                col_start += patch.shape[1]
            row_start += patch.shape[0]
        scene_filled = np.transpose(scene_filled, (2, 0, 1))
        save_path = join(save_folder, f"{target_DOY}-filled" + image_suffix)
        filled_dataset = rasterio.open(save_path, mode='w', **scene_profile)
        filled_dataset.write(scene_filled)
        filled_dataset.close()
