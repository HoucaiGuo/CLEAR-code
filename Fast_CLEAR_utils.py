import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LinearRegression


def linear_interpolation(images, masks,
                         ref_images, ref_masks, ref_indices):
    interp_images = ref_images.copy()
    # the value is 1 if the pixel located at (x, y) is cloudy in at least one reference image
    any_cloudy_mask = np.any(ref_masks, axis=2)
    any_cloudy_row_indices, any_cloudy_col_indices = any_cloudy_mask.nonzero()
    any_cloudy_num = any_cloudy_row_indices.shape[0]
    # process for each any-cloudy location
    for any_cloudy_idx in trange(any_cloudy_num):
    # for any_cloudy_idx in range(any_cloudy_num):
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


def PCA_transform(ref_images, PCA_pct):
    X = np.reshape(ref_images, (ref_images.shape[0] * ref_images.shape[1],
                                ref_images.shape[2] * ref_images.shape[3]))
    pca = PCA(n_components=X.shape[1])
    X_transformed = pca.fit_transform(X)
    idx = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= (PCA_pct / 100))
    X_reduced = X_transformed[:, :idx + 1]
    X_reduced = (X_reduced - X_reduced.min()) / (X_reduced.max() - X_reduced.min())
    ref_PC_images = X_reduced.reshape(ref_images.shape[0], ref_images.shape[1], X_reduced.shape[1])

    return ref_PC_images


def MiniBatchKMeans_classification(ref_PC_images, class_num):
    X = np.reshape(ref_PC_images, (ref_PC_images.shape[0] * ref_PC_images.shape[1],
                                   ref_PC_images.shape[2]))
    kmeans = MiniBatchKMeans(n_clusters=class_num, max_iter=100, batch_size=1024)
    kmeans.fit(X)
    class_map = kmeans.labels_.reshape((ref_PC_images.shape[0], ref_PC_images.shape[1]))

    return class_map, kmeans.cluster_centers_


def check_class_map_validity(class_map, class_centers, cloud_mask):
    class_indices = np.unique(class_map)

    # print(class_indices)
    # fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    # axes[0].imshow(class_map, cmap="Paired")
    # axes[1].imshow(cloud_mask, cmap="gray")
    # plt.show()

    deleted_classes = []
    for class_idx in class_indices:
        class_mask = class_map == class_idx
        common_mask = np.all(np.stack([class_mask, ~cloud_mask], axis=2), axis=2)
        if np.count_nonzero(common_mask) == 0:
            deleted_classes.append(class_idx)

            class_center = class_centers[class_idx, :]
            center_distances = np.sum(np.abs(class_centers - class_center), axis=1)

            sorted_indices = np.argsort(center_distances)
            # print(deleted_classes)
            # print(sorted_indices)
            # print(center_distances)
            for idx in sorted_indices:
                if idx not in deleted_classes:
                    dst_class = idx
                    class_map[class_mask] = dst_class
                    print(f"{class_idx} ---> {dst_class}")
                    break

    return class_map


def fill_single_image_fast(ref_PC_images,
                           cloudy_image, cloud_mask,
                           class_map,
                           sample_num, similar_num):
    reg_prediction = cloudy_image.copy()
    final_prediction = cloudy_image.copy()

    """
    Perform Fast-Sen2Fill class-by-class.
    """
    class_indices = np.unique(class_map)
    for class_idx in class_indices:
        # print(f"\tStart to predict the {class_idx + 1}th land-cover class.")
        class_mask = class_map == class_idx
        class_cloudy_mask = np.all(np.stack([class_mask, cloud_mask], axis=2), axis=2)
        # no cloudy pixel in this class
        if np.count_nonzero(class_cloudy_mask) == 0:
            continue
        # row and column indices of the cloudy pixels in current class
        cloudy_row_indices, cloudy_col_indices = class_cloudy_mask.nonzero()
        cloudy_num = cloudy_row_indices.shape[0]

        # row and column indices of the common pixels
        common_mask = np.all(np.stack([class_mask, ~cloud_mask], axis=2), axis=2)
        common_row_indices, common_col_indices = common_mask.nonzero()

        # sometimes we can not find enough common pixels, especially when the cloud coverage is high
        common_num = common_row_indices.shape[0]
        real_sample_num = sample_num if common_num >= sample_num else common_num

        # randomly select the samples
        sample_indices = np.random.randint(low=0, high=common_num, size=real_sample_num)
        sample_row_indices = common_row_indices[sample_indices]
        sample_col_indices = common_col_indices[sample_indices]

        """
        Linear regression
        """
        common_residuals = np.empty(shape=(common_num, cloudy_image.shape[2]), dtype=np.float32)
        # process for each band
        for band_idx in range(cloudy_image.shape[2]):
            X_train = ref_PC_images[common_mask, :]
            y_train = cloudy_image[common_mask, band_idx]

            reg = LinearRegression()
            # fit the linear regression model
            reg.fit(X_train, y_train)
            # predict
            X_pred = ref_PC_images[class_cloudy_mask, :]
            final_prediction[class_cloudy_mask, band_idx] = reg.predict(X_pred)
            reg_prediction[class_cloudy_mask, band_idx] = final_prediction[class_cloudy_mask, band_idx]

            # residual calculation
            common_residuals[:, band_idx] = y_train - reg.predict(X_train)
        # print("\tFinished linear regression.")

        """
        Residual compensation
        """
        ref_PC_samples = ref_PC_images[sample_row_indices, sample_col_indices, :]
        sample_residuals = common_residuals[sample_indices, :]
        # process for each cloudy pixel
        for cloudy_idx in trange(cloudy_num):
        # for cloudy_idx in range(cloudy_num):
            cloudy_row_idx = cloudy_row_indices[cloudy_idx]
            cloudy_col_idx = cloudy_col_indices[cloudy_idx]
            cloudy_PC_pixel = ref_PC_images[cloudy_row_idx, cloudy_col_idx, :]
            # calculate spectral difference
            spectral_diffs = np.sum(np.abs(ref_PC_samples - cloudy_PC_pixel), axis=1)
            # select similar pixels
            similar_indices = np.argsort(spectral_diffs)[:similar_num]
            # normalize the spectral difference of similar pixels
            similar_spectral_diffs = spectral_diffs[similar_indices]
            similar_spectral_diffs_norm = ((similar_spectral_diffs - similar_spectral_diffs.min()) /
                                           (similar_spectral_diffs.max() - similar_spectral_diffs.min()) + 1)
            # calculate the normalized spatial distance between the target pixel and its similar pixels
            similar_row_indices = sample_row_indices[similar_indices]
            similar_col_indices = sample_col_indices[similar_indices]
            similar_distances = np.sqrt(np.square(similar_row_indices - cloudy_row_idx) +
                                        np.square(similar_col_indices - cloudy_col_idx))
            similar_distances_norm = ((similar_distances - similar_distances.min()) /
                                      (similar_distances.max() - similar_distances.min()) + 1)
            # calculate the weight of similar pixels
            similar_weights = ((1 / (similar_distances_norm * similar_spectral_diffs_norm)) /
                               np.sum((1 / (similar_distances_norm * similar_spectral_diffs_norm))))
            similar_residuals = sample_residuals[similar_indices, :]
            residuals = np.sum(np.stack([similar_weights for i in range(cloudy_image.shape[2])],
                                        axis=1) * similar_residuals, axis=0)
            final_prediction[cloudy_row_idx, cloudy_col_idx, :] += residuals
        # print("\tFinished residual compensation.")

    return reg_prediction, final_prediction
