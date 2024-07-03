import numpy as np
from tqdm import trange
from scipy.interpolate import interp1d
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from skimage.measure import label


def linear_interpolation(images, masks, ref_images, ref_masks, ref_indices):
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
    X = np.reshape(ref_images,
                   (ref_images.shape[0] * ref_images.shape[1],
                    ref_images.shape[2] * ref_images.shape[3]))
    kmeans = KMeans(n_clusters=class_num, max_iter=1000)
    kmeans.fit(X)
    class_map = kmeans.labels_.reshape((ref_images.shape[0], ref_images.shape[1]))

    return class_map


def select_similar_pixels(target_row_idx, target_col_idx,
                          common_row_indices, common_col_indices,
                          target_values, common_values,
                          similar_num):
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


def fill_single_image(ref_images,
                      cloudy_image, cloud_mask,
                      class_map, class_num,
                      win_size, similar_num):
    final_prediction = cloudy_image.copy()
    residuals = np.zeros(shape=cloudy_image.shape, dtype=np.float32)

    print(f"\tStart class-based linear regression.")
    for class_idx in range(class_num):
        class_mask = class_map == class_idx
        class_cloudy_mask = np.all(np.stack([class_mask, cloud_mask], axis=2), axis=2)
        # no cloudy pixel in this class
        if np.count_nonzero(class_cloudy_mask) == 0:
            continue
        # row and column indices of the common pixels
        common_mask = np.all(np.stack([class_mask, ~cloud_mask], axis=2), axis=2)
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

    print(f"\tStart iterative residual compensation.")
    # pad the cloud label, cloud mask, class map, reference images, and residuals,
    # because the residual compensation is a window-based process
    pad_width = win_size // 2
    cloud_labels = label(cloud_mask, background=0)
    cloud_labels_pad = np.pad(cloud_labels, pad_width=((pad_width, pad_width), (pad_width, pad_width)),
                              mode="constant",
                              constant_values=((0, 0), (0, 0)))
    cloud_mask_pad = np.pad(cloud_mask,
                            pad_width=((pad_width, pad_width),
                                       (pad_width, pad_width)), mode="reflect")
    class_map_pad = np.pad(class_map,
                           pad_width=((pad_width, pad_width),
                                      (pad_width, pad_width)), mode="reflect")
    ref_images_pad = np.pad(ref_images,
                            pad_width=((pad_width, pad_width),
                                       (pad_width, pad_width),
                                       (0, 0),
                                       (0, 0)), mode="reflect")
    residuals_pad = np.pad(residuals,
                           pad_width=((pad_width, pad_width),
                                      (pad_width, pad_width),
                                      (0, 0)), mode="reflect")

    # numpy.unique() returns the sorted unique values, the first element is 0,
    # which represents the background (clear pixels)
    unique_cloud_labels = np.unique(cloud_labels)[1:]
    # number of cloud patches
    cloud_count = unique_cloud_labels.shape[0]

    # process for each cloud
    for cloud_label in unique_cloud_labels:
        print(f"\tStart to predict cloud {cloud_label} (total {cloud_count}).")
        """
        Extract local cloud patch
        """
        # calculate the extent of current cloud, taking the padding into consideration
        cloud_label_mask = cloud_labels_pad == cloud_label
        cloud_row_indices_g, cloud_col_indices_g = cloud_label_mask.nonzero()
        row_start_g = cloud_row_indices_g.min() - pad_width
        row_end_g = cloud_row_indices_g.max() + pad_width + 1
        col_start_g = cloud_col_indices_g.min() - pad_width
        col_end_g = cloud_col_indices_g.max() + pad_width + 1

        # subset the cloud label mask, class map, cloud mask, reference images,
        # and cloudy image from global to local patch, this can accelerate the computation
        local_cloud_label_mask = cloud_label_mask[row_start_g:row_end_g, col_start_g:col_end_g]
        local_class_map = class_map_pad[row_start_g:row_end_g, col_start_g:col_end_g]
        local_cloud_mask = cloud_mask_pad[row_start_g:row_end_g, col_start_g:col_end_g]
        local_ref_images = ref_images_pad[row_start_g:row_end_g, col_start_g:col_end_g, :, :]
        local_residuals = residuals_pad[row_start_g:row_end_g, col_start_g:col_end_g, :]

        """
        Calculate the number of common pixels inside the local window for all cloudy pixels
        """
        # for simplicity, the 2D array window_common_num has the same shape as local_cloud_mask,
        # the clear pixels will be masked as nodata
        window_common_num = np.empty(shape=(local_cloud_label_mask.shape[0] *
                                            local_cloud_label_mask.shape[1]), dtype=np.int32)
        for cloudy_idx in range(window_common_num.shape[0]):
            # 1d index to 2d row-col indices
            cloudy_row_idx_l, cloudy_col_idx_l = np.unravel_index(cloudy_idx, shape=local_cloud_label_mask.shape)

            # if it is a cloudy pixel
            if local_cloud_label_mask[cloudy_row_idx_l, cloudy_col_idx_l]:
                cloudy_class = local_class_map[cloudy_row_idx_l, cloudy_col_idx_l]

                # subset the two masks from local to window (centered by the target cloudy pixel)
                window_clear_mask = ~local_cloud_mask[cloudy_row_idx_l - pad_width:cloudy_row_idx_l + pad_width + 1,
                                     cloudy_col_idx_l - pad_width:cloudy_col_idx_l + pad_width + 1]
                window_class_mask = local_class_map[cloudy_row_idx_l - pad_width:cloudy_row_idx_l + pad_width + 1,
                                    cloudy_col_idx_l - pad_width:cloudy_col_idx_l + pad_width + 1] \
                                    == cloudy_class

                # count the number of clear pixels that belong to the same land-cover class
                # as the target cloudy pixel
                window_common_mask = np.all(np.stack([window_clear_mask, window_class_mask], axis=2), axis=2)
                window_common_num[cloudy_idx] = np.count_nonzero(window_common_mask)
            # if it is a clear pixel
            else:
                window_common_num[cloudy_idx] = -1
        # mask the clear pixels (-1 values)
        window_common_num = np.ma.array(data=window_common_num,
                                        mask=[window_common_num == -1])
        # number of actual cloudy pixels of current cloud patch
        cloudy_num = np.count_nonzero(~window_common_num.mask)

        """
        Iterative residual compensation for each target cloudy pixel
        """
        for idx in trange(cloudy_num):
            # find a cloudy pixel that has the most clear-class pixels inside the window,
            # np.ma.argmax() only returns the index of cloudy pixel since the clear pixels were masked
            target_idx = np.ma.argmax(window_common_num)
            # 1d index to 2d row-col indices
            target_row_idx_l, target_col_idx_l = np.unravel_index(target_idx, shape=local_cloud_label_mask.shape)
            # global row-col indices of the target cloudy pixel in the whole image
            target_row_idx_g = row_start_g + target_row_idx_l - pad_width
            target_col_idx_g = col_start_g + target_col_idx_l - pad_width

            # land-cover class of the target cloudy pixel
            target_class = local_class_map[target_row_idx_l, target_col_idx_l]

            # Condition 1: residual compensation using window pixels,
            # enough similar pixels can be searched inside the window (centered by the target cloudy pixel)
            if window_common_num[target_idx] >= similar_num:
                # subset from local to window
                window_clear_mask = ~local_cloud_mask[target_row_idx_l - pad_width:target_row_idx_l + pad_width + 1,
                                     target_col_idx_l - pad_width:target_col_idx_l + pad_width + 1]
                window_class_mask = local_class_map[target_row_idx_l - pad_width:target_row_idx_l + pad_width + 1,
                                    target_col_idx_l - pad_width:target_col_idx_l + pad_width + 1] \
                                    == target_class
                window_reference_images = local_ref_images[
                                          target_row_idx_l - pad_width:target_row_idx_l + pad_width + 1,
                                          target_col_idx_l - pad_width:target_col_idx_l + pad_width + 1, :, :]

                # row-col indices of the common pixels inside the window
                window_common_mask = np.all(np.stack([window_clear_mask, window_class_mask], axis=2), axis=2)
                common_row_indices_w, common_col_indices_w = window_common_mask.nonzero()

                # select similar pixels for the target cloudy pixel
                target_values = window_reference_images[pad_width, pad_width, :, :]
                common_values = window_reference_images[common_row_indices_w, common_col_indices_w, :, :]
                similar_row_indices_w, similar_col_indices_w, similar_weights = \
                    select_similar_pixels(pad_width, pad_width,
                                          common_row_indices_w, common_col_indices_w,
                                          target_values, common_values,
                                          similar_num)

                # calculate and assign the residual of the target cloudy pixel
                window_residuals = local_residuals[target_row_idx_l - pad_width:target_row_idx_l + pad_width + 1,
                                   target_col_idx_l - pad_width:target_col_idx_l + pad_width + 1, :]
                similar_residuals = window_residuals[similar_row_indices_w, similar_col_indices_w, :]
                residual = np.sum(np.stack([similar_weights for i in range(cloudy_image.shape[2])],
                                           axis=1) * similar_residuals, axis=0)
                final_prediction[target_row_idx_g, target_col_idx_g, :] += residual

            # Condition 1 is not satisfied,
            # enough similar pixels can not be searched inside the window
            else:
                local_clear_mask = ~local_cloud_mask
                local_class_mask = local_class_map == target_class
                # row-col indices of the common pixels in the local patch
                local_common_mask = np.all(np.stack([local_clear_mask, local_class_mask], axis=2), axis=2)
                common_row_indices_l, common_col_indices_l = local_common_mask.nonzero()

                # Condition 2: residual compensation using local patch pixels,
                # enough similar pixels can be searched inside the local patch
                if common_row_indices_l.shape[0] >= similar_num:
                    # select similar pixels for the target cloudy pixel
                    target_values = local_ref_images[target_row_idx_l, target_col_idx_l, :, :]
                    common_values = local_ref_images[common_row_indices_l, common_col_indices_l, :, :]
                    similar_row_indices_l, similar_col_indices_l, similar_weights = \
                        select_similar_pixels(target_row_idx_l, target_col_idx_l,
                                              common_row_indices_l, common_col_indices_l,
                                              target_values, common_values,
                                              similar_num)

                    # calculate and assign the residual of the target cloudy pixel
                    similar_residuals = local_residuals[similar_row_indices_l, similar_col_indices_l, :]
                    residual = np.sum(np.stack([similar_weights for i in range(cloudy_image.shape[2])],
                                               axis=1) * similar_residuals, axis=0)
                    final_prediction[target_row_idx_g, target_col_idx_g, :] += residual

                # Condition 3: residual compensation using global pixels,
                # enough similar pixels can only be searched using the whole image
                else:
                    # global_clear_mask = ~self.cloud_mask
                    global_clear_mask = ~cloud_mask_pad[pad_width:-pad_width, pad_width:-pad_width]
                    global_class_mask = class_map == target_class
                    # row-col indices of the common pixels in the whole image
                    global_common_mask = np.all(np.stack([global_clear_mask, global_class_mask], axis=2),
                                                axis=2)
                    common_row_indices_g, common_col_indices_g = global_common_mask.nonzero()

                    # select similar pixels for the target cloudy pixel
                    target_values = ref_images[target_row_idx_g, target_col_idx_g, :, :]
                    common_values = ref_images[common_row_indices_g, common_col_indices_g, :, :]
                    similar_row_indices_g, similar_col_indices_g, similar_weights = \
                        select_similar_pixels(target_row_idx_g, target_col_idx_g,
                                              common_row_indices_g, common_col_indices_g,
                                              target_values, common_values,
                                              similar_num)

                    # calculate and assign the residual of target cloudy pixel
                    similar_residuals = residuals[similar_row_indices_g, similar_col_indices_g, :]
                    residual = np.sum(np.stack([similar_weights for i in range(cloudy_image.shape[2])],
                                               axis=1) * similar_residuals, axis=0)
                    final_prediction[target_row_idx_g, target_col_idx_g, :] += residual

            """
            Post-processing
            """
            # the predicted residual can be used in the following iterations
            local_residuals[target_row_idx_l, target_col_idx_l, :] = residual
            residuals_pad[target_row_idx_g + pad_width, target_col_idx_g + pad_width, :] = residual

            # the window_common_num of all cloudy-class pixels
            # in the processing window (centered by the target pixel) should plus 1
            window_cloud_label_mask = local_cloud_label_mask[
                                      target_row_idx_l - pad_width:target_row_idx_l + pad_width + 1,
                                      target_col_idx_l - pad_width:target_col_idx_l + pad_width + 1]
            window_class_mask = local_class_map[
                                target_row_idx_l - pad_width:target_row_idx_l + pad_width + 1,
                                target_col_idx_l - pad_width:target_col_idx_l + pad_width + 1] \
                                == target_class
            window_cloud_label_class_mask = np.all(
                np.stack([window_cloud_label_mask, window_class_mask], axis=2), axis=2)
            update_row_indices_w, update_col_indices_w = window_cloud_label_class_mask.nonzero()
            update_indices_2d = (update_row_indices_w + target_row_idx_l - pad_width,
                                 update_col_indices_w + target_col_idx_l - pad_width)
            update_indices_1d = np.ravel_multi_index(update_indices_2d,
                                                     dims=local_cloud_label_mask.shape,
                                                     order="C")
            window_common_num[update_indices_1d] += 1

            # set the processed target pixel as clear
            window_common_num.mask[target_idx] = True
            local_cloud_mask[target_row_idx_l, target_col_idx_l] = 0
            cloud_mask_pad[target_row_idx_g + pad_width, target_col_idx_g + pad_width] = 0

    return reg_prediction, final_prediction
