import numpy as np
from tqdm import trange
from scipy.interpolate import interp1d
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.linear_model import LinearRegression
from skimage.morphology import disk, binary_erosion


def linear_interpolation(images, masks, ref_images, ref_masks, ref_indices):
    """
    Fill gaps in reference images using linear interpolation.

    Parameters
    ----------
    images: All available images.
    masks: Masks of all available images.
    ref_images: Reference images.
    ref_masks: Masks of reference images.
    ref_indices: Indices of reference images in all available images.

    Returns
    -------
    interp_images: Interpolated reference images.
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


def classify_reference_images(ref_images, class_num):
    """
    Classify the stacked reference images using KMeans.

    Parameters
    ----------
    ref_images: Reference images.
    class_num: Number of land-cover classes.

    Returns
    -------
    class_map: Classification map.
    class_centers: Vectors of class centers.
    """
    X = np.reshape(ref_images,
                   (ref_images.shape[0] * ref_images.shape[1],
                    ref_images.shape[2] * ref_images.shape[3]))
    kmeans = KMeans(n_clusters=class_num, max_iter=1000)
    kmeans.fit(X)
    class_map = kmeans.labels_.reshape((ref_images.shape[0], ref_images.shape[1]))
    class_centers = kmeans.cluster_centers_

    return class_map, class_centers


def check_class_map_validity(class_map, class_centers, cloud_mask):
    """
    Check the validity of classification map. If there is no clear pixel within a certain class,
    merge it with the nearest class.

    Parameters
    ----------
    class_map
    class_centers
    cloud_mask

    Returns
    -------

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


def select_similar_pixels(target_row_idx, target_col_idx,
                          common_row_indices, common_col_indices,
                          target_values, common_values,
                          similar_num):
    """
    Select similar pixels for the target pixel.

    Parameters
    ----------
    target_row_idx
    target_col_idx
    common_row_indices
    common_col_indices
    target_values
    common_values
    similar_num

    Returns
    -------

    """
    # shape = (common_num, band_num * ref_num)
    common_values = common_values.reshape(-1, common_values.shape[1] * common_values.shape[2])
    # shape = (, band_num * ref_num)
    target_values = target_values.reshape(-1, target_values.shape[0] * target_values.shape[1])

    spectral_diffs = np.sum(np.abs(common_values - target_values), axis=1)

    similar_indices = np.argsort(spectral_diffs)[:similar_num]
    similar_spectral_diffs = spectral_diffs[similar_indices]
    similar_spectral_diffs_norm = ((similar_spectral_diffs - similar_spectral_diffs.min()) /
                                   (similar_spectral_diffs.max() - similar_spectral_diffs.min()) + 1)
    similar_row_indices = common_row_indices[similar_indices]
    similar_col_indices = common_col_indices[similar_indices]

    similar_distances = np.sqrt(np.square(similar_row_indices - target_row_idx) +
                                np.square(similar_col_indices - target_col_idx))
    similar_distances_norm = ((similar_distances - similar_distances.min()) /
                              (similar_distances.max() - similar_distances.min()) + 1)

    similar_weights = ((1 / (similar_distances_norm * similar_spectral_diffs_norm)) /
                       np.sum((1 / (similar_distances_norm * similar_spectral_diffs_norm))))

    return similar_row_indices, similar_col_indices, similar_weights


def calculate_window_extent(center_row_idx, center_col_idx, win_size, image_shape):
    """
    Given an expected window size, calculate the valid window extent for the target pixel.
    The window shall not exceed the image boundary.

    Parameters
    ----------
    center_row_idx
    center_col_idx
    win_size
    image_shape

    Returns
    -------

    """
    half_win_size = win_size // 2

    row_start = center_row_idx - half_win_size
    if row_start < 0:
        row_start = 0
    row_end = center_row_idx + half_win_size + 1
    if row_end > image_shape[0]:
        row_end = image_shape[0]
    col_start = center_col_idx - half_win_size
    if col_start < 0:
        col_start = 0
    col_end = center_col_idx + half_win_size + 1
    if col_end > image_shape[1]:
        col_end = image_shape[1]

    return row_start, row_end, col_start, col_end


def calculate_window_common_num(target_row_idx, target_col_idx,
                                win_size, cloud_mask, class_map):
    row_start, row_end, col_start, col_end = calculate_window_extent(target_row_idx, target_col_idx,
                                                                     win_size, cloud_mask.shape)
    target_class = class_map[target_row_idx, target_col_idx]
    win_clear_mask = ~cloud_mask[row_start:row_end, col_start:col_end]
    win_class_mask = class_map[row_start:row_end, col_start:col_end] == target_class
    win_common_mask = np.all(np.stack([win_clear_mask, win_class_mask], axis=2), axis=2)
    win_common_num = np.count_nonzero(win_common_mask)

    return win_common_num


def expand_window_size(initial_win_size, expand_size,
                       target_row_idx, target_col_idx,
                       target_win_common_num, similar_num,
                       target_class, class_map,
                       cloud_mask):
    """
    Expand the window size for searching common pixels.

    Parameters
    ----------
    initial_win_size
    expand_size
    target_row_idx
    target_col_idx
    target_win_common_num
    similar_num
    target_class
    class_map
    cloud_mask

    Returns
    -------

    """
    final_win_size = initial_win_size
    final_win_common_num = target_win_common_num
    while final_win_common_num < similar_num:
        final_win_size += expand_size

        row_start, row_end, col_start, col_end = (
            calculate_window_extent(target_row_idx, target_col_idx, final_win_size, cloud_mask.shape))

        win_clear_mask = ~cloud_mask[row_start:row_end, col_start:col_end]
        win_class_mask = class_map[row_start:row_end, col_start:col_end] == target_class
        win_common_mask = np.all(np.stack([win_clear_mask, win_class_mask], axis=2), axis=2)
        final_win_common_num = np.count_nonzero(win_common_mask)

        final_row_win_size = row_end - row_start
        final_col_win_size = col_end - col_start

        # if the window size is larger than the whole image, do not expand
        if final_row_win_size >= cloud_mask.shape[0] and final_col_win_size >= cloud_mask.shape[1]:
            break

    return row_start, row_end, col_start, col_end, final_row_win_size, final_col_win_size


def classify_reference_images(ref_images, class_num):
    """
    Classify the stacked reference images using MiniBatchKMeans.

    Parameters
    ----------
    ref_images: Reference images.
    class_num: Number of land-cover classes.

    Returns
    -------
    class_map: Classification map.
    class_centers: Vectors of class centers.
    """
    X = np.reshape(ref_images,
                   (ref_images.shape[0] * ref_images.shape[1],
                    ref_images.shape[2] * ref_images.shape[3]))
    kmeans = MiniBatchKMeans(n_clusters=class_num, max_iter=1000)
    kmeans.fit(X)
    class_map = kmeans.labels_.reshape((ref_images.shape[0], ref_images.shape[1]))

    return class_map, kmeans.cluster_centers_


def fill_single_image(ref_images,
                      cloudy_image, cloud_mask,
                      class_map, class_num,
                      win_size, expand_size,
                      similar_num):
    """
    Fill a cloudy image using Fast CLEAR.

    Parameters
    ----------
    ref_images
    cloudy_image
    cloud_mask
    class_map
    class_num
    win_size
    expand_size
    similar_num

    Returns
    -------

    """
    final_prediction = cloudy_image.copy()
    cloud_mask_1 = cloud_mask.copy()
    residuals = np.zeros(shape=cloudy_image.shape, dtype=np.float32)

    """
    Step 1. Class-based linear regression
    """
    print(f"\tStart class-based linear regression.")
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
    print(f"\tStart iterative residual compensation.")
    cloudy_num = np.count_nonzero(cloud_mask_1)

    half_row_win_size = win_size // 2
    half_col_win_size = win_size // 2
    footprint = disk(1)
    while cloudy_num > 0:
        # calculate edge mask
        mask_eroded = binary_erosion(cloud_mask_1, footprint)
        edge_mask = np.bitwise_xor(cloud_mask_1, mask_eroded)
        edge_row_indices, edge_col_indices = edge_mask.nonzero()
        edge_num = edge_row_indices.shape[0]

        # process for each edge pixel
        for edge_idx in trange(edge_num):
            target_row_idx = edge_row_indices[edge_idx]
            target_col_idx = edge_col_indices[edge_idx]
            target_class = class_map[target_row_idx, target_col_idx]
            # number of common pixel inside the local window
            target_win_common_num = calculate_window_common_num(target_row_idx, target_col_idx,
                                                                win_size, cloud_mask_1, class_map)

            # expand the window
            if target_win_common_num < similar_num:
                row_start, row_end, col_start, col_end, \
                final_row_win_size, final_col_win_size = (
                    expand_window_size(win_size, expand_size,
                                       target_row_idx, target_col_idx,
                                       target_win_common_num, similar_num,
                                       target_class, class_map,
                                       cloud_mask_1))
            else:
                row_start, row_end, col_start, col_end = calculate_window_extent(target_row_idx, target_col_idx,
                                                                                 win_size, cloud_mask_1.shape)

            # subset the local window
            win_clear_mask = ~cloud_mask_1[row_start:row_end, col_start:col_end]
            win_class_mask = class_map[row_start:row_end, col_start:col_end] == target_class
            win_common_mask = np.all(np.stack([win_clear_mask, win_class_mask], axis=2), axis=2)
            win_ref_images = ref_images[row_start:row_end, col_start:col_end, :, :]
            common_row_indices_w, common_col_indices_w = win_common_mask.nonzero()

            target_row_idx_w = target_row_idx - row_start
            target_col_idx_w = target_col_idx - col_start

            # select similar pixels for the target cloudy pixel
            target_values = win_ref_images[target_row_idx_w, target_col_idx_w, :, :]
            common_values = win_ref_images[common_row_indices_w, common_col_indices_w, :, :]

            similar_row_indices_w, similar_col_indices_w, similar_weights = \
                select_similar_pixels(target_row_idx_w, target_col_idx_w,
                                      common_row_indices_w, common_col_indices_w,
                                      target_values, common_values,
                                      similar_num)

            # calculate and assign the residual of the target cloudy pixel
            win_residuals = residuals[row_start:row_end, col_start:col_end, :]
            similar_residuals = win_residuals[similar_row_indices_w, similar_col_indices_w, :]
            residual = np.sum(np.stack([similar_weights for i in range(cloudy_image.shape[2])],
                                       axis=1) * similar_residuals, axis=0)
            final_prediction[target_row_idx, target_col_idx, :] += residual

            """
            Post-processing
            """
            residuals[target_row_idx, target_col_idx, :] = residual
            # set the processed target pixel as clear
            cloud_mask_1[target_row_idx, target_col_idx] = 0
            cloudy_num -= 1

    return reg_prediction, final_prediction
