import os
import numpy as np
np.set_printoptions(suppress=True, precision=5)
import rasterio
from rasterio.windows import Window
from os.path import join
# from functions import *

image_folder = r"data/Omaha-Scene/R112_T15TTF_images"
mask_folder = r"data/Omaha-Scene/R112_T15TTF_masks"
temp_folder = r"data/Omaha-Scene/Temp"

image_suffix = ".jp2"
mask_suffix = ".tif"

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
    patch_row_num = np.round(image_row_size / patch_row_size).astype(int)
    patch_col_num = np.round(image_col_size / patch_col_size).astype(int)
    patch_total_num = patch_row_num * patch_col_num
    print(f"There are total {patch_total_num} ({patch_row_num} * {patch_col_num}) patches.")

    for patch_row_idx in range(patch_row_num):
        for patch_col_idx in range(patch_col_num):
            patch_folder = join(temp_folder, f"{patch_row_idx}-{patch_col_idx}")
            os.makedirs(patch_folder)

    """
    Get image names
    """
    file_names = os.listdir(image_folder)
    image_names = []
    for file_name in file_names:
        if file_name.endswith(image_suffix):
            image_names.append(file_name)
    image_names.sort(key=get_DOY)

    """
    Divide images to patches
    """
    print(f"Start to divide the images to patches.")
    patch_cloud_coverages = np.empty(shape=(patch_row_num, patch_col_num, len(image_names)), dtype=np.float32)
    for image_idx in range(len(image_names)):
        print(f"\tDivide the {image_idx + 1}th image.")
        image_name = image_names[image_idx]
        DOY = image_name.split(sep=".")[0]
        mask_name = f"{DOY}-mask" + mask_suffix
        image_path = join(image_folder, image_name)
        image_dataset = rasterio.open(image_path)
        image_profile = image_dataset.profile
        print(image_profile)
        mask_path = join(mask_folder, mask_name)
        mask_dataset = rasterio.open(mask_path)
        mask_profile = mask_dataset.profile

        for patch_row_idx in range(patch_row_num):
            if patch_row_idx == 0:
                row_start_idx = 0
                row_end_idx = 0 + patch_row_size + overlap_row_size
            elif patch_row_idx == patch_row_num - 1:
                row_start_idx = patch_row_idx * patch_row_size - overlap_row_size
                row_end_idx = image_row_size
            else:
                row_start_idx = patch_row_idx * patch_row_size - overlap_row_size
                row_end_idx = patch_row_idx * patch_row_size + patch_row_size + overlap_row_size
            for patch_col_idx in range(patch_col_num):
                if patch_col_idx == 0:
                    col_start_idx = 0
                    col_end_idx = 0 + patch_col_size + overlap_col_size
                elif patch_col_idx == patch_col_num - 1:
                    col_start_idx = patch_col_idx * patch_col_size - overlap_col_size
                    col_end_idx = image_col_size
                else:
                    col_start_idx = patch_col_idx * patch_col_size - overlap_col_size
                    col_end_idx = patch_col_idx * patch_col_size + patch_col_size + overlap_col_size

                window = Window.from_slices((row_start_idx, row_end_idx), (col_start_idx, col_end_idx))
                image_transform = image_dataset.window_transform(window)
                image_patch = image_dataset.read(window=window)
                image_patch_profile = image_dataset.profile.copy()
                image_patch_profile.update({'width': image_patch.shape[2],
                                            'height': image_patch.shape[1],
                                            'transform': image_transform})
                patch_folder = join(temp_folder, f"{patch_row_idx}-{patch_col_idx}")
                save_path = join(patch_folder, f"{DOY}-{patch_row_idx}-{patch_col_idx}" + image_suffix)
                image_patch_dataset = rasterio.open(save_path, mode='w', **image_patch_profile)
                image_patch_dataset.write(image_patch)
                image_patch_dataset.close()

                mask_transform = mask_dataset.window_transform(window)
                mask_patch = mask_dataset.read(window=window)
                patch_cloud_coverage = (np.count_nonzero(mask_patch) /
                                        (mask_patch.shape[1] * mask_patch.shape[2]) * 100)
                patch_cloud_coverages[patch_row_idx, patch_col_idx, image_idx] = patch_cloud_coverage
                mask_patch_profile = mask_dataset.profile.copy()
                mask_patch_profile.update({'width': mask_patch.shape[2],
                                           'height': mask_patch.shape[1],
                                           'transform': mask_transform})
                save_path = join(patch_folder, f"{DOY}-{patch_row_idx}-{patch_col_idx}-mask" + mask_suffix)
                mask_patch_dataset = rasterio.open(save_path, mode='w', **mask_patch_profile)
                mask_patch_dataset.write(mask_patch)
                mask_patch_dataset.close()
    np.save(join(temp_folder, "cloud-coverages.npy"), patch_cloud_coverages)

