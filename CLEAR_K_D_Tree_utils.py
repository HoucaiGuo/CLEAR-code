import numpy as np
from tqdm import trange, tqdm
from scipy.interpolate import interp1d
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KDTree


def linear_interpolation(images, masks, ref_images, ref_masks, ref_indices):
    """
    Fill gaps in reference images using linear interpolation.
    """
    interp_images = ref_images.copy()
    # the value is 1 if the pixel located at (x, y) is cloudy in at least one reference image
    any_cloudy_mask = np.any(ref_masks, axis=2)
    any_cloudy_row_indices, any_cloudy_col_indices = any_cloudy_mask.nonzero()
    any_cloudy_num = any_cloudy_row_indices.shape[0]
    # process for each any-cloudy location
    for any_cloudy_idx in trange(any_cloudy_num):
        any_cloudy_row_idx = any_cloudy_row_indices[any_cloudy_idx]
        any_cloudy_col_idx = any_cloudy_col_indices[any_cloudy_idx]
        # process for each band
        for band_idx in range(ref_images.shape[2]):
            # time-series reflectances of band band_idx at location (any_cloudy_row_idx, any_cloudy_col_idx)
            values = images[any_cloudy_row_idx, any_cloudy_col_idx, band_idx, :]
            # 1, cloudy; 0, clear
            cloud_flag = masks[any_cloudy_row_idx, any_cloudy_col_idx, :]
            clear_indices = (cloud_flag == 0).nonzero()[0]
            clear_values = values[clear_indices]
            # only one clear value, in which case the linear interpolation
            # cannot be performed (it needs at least two)
            if clear_values.shape[0] == 1:
                interp_values = np.array([clear_values[0] for i in range(images.shape[3])])
            else:
                # linear interpolation to fill the cloudy values
                linear_interp = interp1d(clear_indices, clear_values, kind='linear', fill_value="extrapolate")
                image_indices = np.arange(0, cloud_flag.shape[0], 1)
                interp_values = linear_interp(image_indices)
            interp_images[any_cloudy_row_idx, any_cloudy_col_idx, band_idx, :] = interp_values[ref_indices]

    return interp_images


def classify_reference_images_fast(ref_images, class_num):
    """
    Classify the stacked reference images using MiniBatchKMeans.
    """
    X = np.reshape(ref_images,
                   (ref_images.shape[0] * ref_images.shape[1],
                    ref_images.shape[2] * ref_images.shape[3]))
    kmeans = MiniBatchKMeans(n_clusters=class_num, max_iter=1000, random_state=42)
    kmeans.fit(X)
    class_map = kmeans.labels_.reshape((ref_images.shape[0], ref_images.shape[1]))

    return class_map, kmeans.cluster_centers_


def check_class_map_validity(class_map, class_centers, cloud_mask):
    """
    Check the validity of classification map. If there is no clear pixel within a certain class,
    merge it with the nearest class.
    """
    class_indices = np.unique(class_map)

    deleted_classes = []
    for class_idx in class_indices:
        class_mask = class_map == class_idx
        common_mask = np.all(np.stack([class_mask, ~cloud_mask], axis=2), axis=2)
        if np.count_nonzero(common_mask) == 0:
            deleted_classes.append(class_idx)

            class_center = class_centers[class_idx, :]
            center_distances = np.sum(np.abs(class_centers - class_center), axis=1)

            sorted_indices = np.argsort(center_distances)
            for idx in sorted_indices:
                if idx not in deleted_classes:
                    dst_class = idx
                    class_map[class_mask] = dst_class
                    print(f"{class_idx} ---> {dst_class}")
                    break

    return class_map


def fill_single_image_kd_tree(ref_images,
                              cloudy_image, cloud_mask,
                              class_map, class_num,
                              common_num, similar_num):
    """
    Fill a cloudy image using CLEAR.
    """
    final_prediction = cloudy_image.copy()
    cloud_mask_1 = cloud_mask.copy()
    residuals = np.zeros(shape=cloudy_image.shape, dtype=np.float32)

    """
    Step 1. Class-based linear regression
    """
    # print(f"\tStart class-based linear regression.")
    for class_idx in range(class_num):
        class_mask = class_map == class_idx
        class_cloudy_mask = np.all(np.stack([class_mask, cloud_mask_1], axis=2), axis=2)
        # no cloudy pixel in this class
        if np.count_nonzero(class_cloudy_mask) == 0:
            continue
        # row and column indices of the common pixels
        common_mask = np.all(np.stack([class_mask, ~cloud_mask_1], axis=2), axis=2)
        for band_idx in range(cloudy_image.shape[2]):
            ref_bands = ref_images[:, :, band_idx, :]
            cloudy_band = cloudy_image[:, :, band_idx]

            reg = LinearRegression()

            # shape = (common_num, ref_num)
            X_train = ref_bands[common_mask, :]
            # shape = (common_num, )
            y_train = cloudy_band[common_mask]

            # fit, predict, and calculate the residuals
            reg.fit(X_train, y_train)
            # shape = (class_cloudy_num, ref_num)
            X_pred = ref_bands[class_cloudy_mask, :]
            final_prediction[class_cloudy_mask, band_idx] = reg.predict(X_pred)
            residuals[common_mask, band_idx] = y_train - reg.predict(X_train)

    reg_prediction = final_prediction.copy()

    """
    Step 2. Iterative residual compensation
    """
    # print(f"\tStart iterative residual compensation.")
    for class_idx in range(class_num):
        class_mask = class_map == class_idx
        class_cloudy_mask = np.all(np.stack([class_mask, cloud_mask_1], axis=2), axis=2)
        # no cloudy pixel in this class
        if np.count_nonzero(class_cloudy_mask) == 0:
            continue
        # row and column indices of the common pixels
        common_mask = np.all(np.stack([class_mask, ~cloud_mask_1], axis=2), axis=2)
        common_row_indices, common_col_indices = common_mask.nonzero()
        common_pixels = ref_images[common_mask, :, :]
        common_pixels = np.reshape(common_pixels, (common_pixels.shape[0],
                                                   common_pixels.shape[1] * common_pixels.shape[2]))
        common_coordinates = np.stack([common_row_indices, common_col_indices], axis=1)
        # print(common_coordinates.shape)
        kd_tree = KDTree(common_coordinates, leaf_size=40, metric="euclidean")
        # print(f"The K-D Tree for class {class_idx} has been constructed!")

        k = common_num if common_num <= common_coordinates.shape[0] else common_coordinates.shape[0]
        # print(k)

        common_residuals = residuals[common_mask, :]

        cloudy_num = np.count_nonzero(class_cloudy_mask)
        cloudy_row_indices, cloudy_col_indices = class_cloudy_mask.nonzero()
        for cloudy_idx in trange(cloudy_num, position=0, leave=True):
            target_row_idx = cloudy_row_indices[cloudy_idx]
            target_col_idx = cloudy_col_indices[cloudy_idx]
            # cloudy_vals = np.expand_dims(cloudy_pixels[cloudy_idx, :], axis=0)
            cloudy_coordinates = np.expand_dims(np.stack([target_row_idx, target_col_idx], axis=0), axis=0)
            # print(cloudy_coordinates.shape)
            indices = kd_tree.query(cloudy_coordinates, k=k, return_distance=False)
            indices = indices.squeeze()

            """
            consider spectral difference
            """
            common_values = common_pixels[indices, :]
            target_values = ref_images[target_row_idx, target_col_idx, :, :]
            # shape = (, band_num * ref_num)
            target_values = target_values.reshape(-1, target_values.shape[0] * target_values.shape[1])
            spectral_diffs = np.sum(np.abs(common_values - target_values), axis=1)

            similar_indices = np.argsort(spectral_diffs)[:similar_num]
            similar_spectral_diffs = spectral_diffs[similar_indices]
            similar_spectral_diffs_norm = ((similar_spectral_diffs - similar_spectral_diffs.min()) /
                                           (similar_spectral_diffs.max() - similar_spectral_diffs.min() + 0.00001) + 1)

            selected_row_indices = common_row_indices[indices]
            selected_col_indices = common_col_indices[indices]

            similar_row_indices = selected_row_indices[similar_indices]
            similar_col_indices = selected_col_indices[similar_indices]

            similar_distances = np.sqrt(np.square(similar_row_indices - target_row_idx) +
                                        np.square(similar_col_indices - target_col_idx))
            similar_distances_norm = ((similar_distances - similar_distances.min()) /
                                      (similar_distances.max() - similar_distances.min() + 0.00001) + 1)

            similar_weights = ((1 / (similar_distances_norm * similar_spectral_diffs_norm)) /
                               np.sum((1 / (similar_distances_norm * similar_spectral_diffs_norm))))

            # similar_residuals = common_residuals[indices, :]
            selected_residuals = common_residuals[indices, :]
            similar_residuals = selected_residuals[similar_indices, :]
            residual = np.sum(np.stack([similar_weights for i in range(cloudy_image.shape[2])],
                                       axis=1) * similar_residuals,
                              axis=0)
            final_prediction[target_row_idx, target_col_idx, :] += residual

    return reg_prediction, final_prediction


