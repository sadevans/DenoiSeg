import numpy as np
from denoiseg.utils.misc_utils import shuffle_train_data


def generate_patches_from_list(data,
                               masks,
                               axes,
                               num_patches_per_img=None,
                               shape=(256, 256),
                               augment=True,
                               shuffle=False,
                               seed=1):
    """
    Extracts patches from 'list_data', which is a list of images, and returns them in a 'numpy-array'. The images
    can have different dimensionality.
    Parameters
    ----------
    seed
    data                : list(array(float))
                          List of images

    masks               : list(array(float))
                          List of masks

    axes                : str
                          Possible dimensions include S(number of samples), ZYX(dimesions of a single sample),
                          C(channel, can be singleton dimension). E.g., SYXC in case of 2D data with S samples of shape YX
    num_patches_per_img : int, optional(default=None)
                          Number of patches to extract per image. If 'None', as many patches as fit i nto the
                          dimensions are extracted.
    shape               : tuple(int), optional(default=(256, 256))
                          Shape of the extracted patches.

    augment             : bool, optional(default=True)
                          Rotate the patches in XY-Plane and flip them along X-Axis. This only works if the patches are square in XY.
    shuffle             : bool, optional(default=False)
                          Shuffles extracted patches across all given images (data).
    Returns
    -------
    patches : array(float)
              Numpy-Array with the patches. The dimensions are 'SZYXC' or 'SYXC'
    """
    image_patches, mask_patches = [], []
    assert len(data) == len(masks)

    for img, mask in zip(data, masks):
        for s in range(img.shape[0]):
            p = generate_patches(img[s][np.newaxis], axes, num_patches=num_patches_per_img, shape=shape,
                                 augment=augment)
            m = generate_patches(mask[s][np.newaxis], axes, num_patches=num_patches_per_img, shape=shape,
                                 augment=augment)

            image_patches.append(p)
            mask_patches.append(m)

    train_images = np.concatenate(image_patches, axis=0)
    train_masks = np.concatenate(mask_patches, axis=0)

    if shuffle:
        train_images, train_masks = shuffle_train_data(train_images, train_masks, random_seed=seed)

    return train_images, train_masks


def generate_patches(data, axes, num_patches=None, shape=(256, 256), augment=True, shuffle=False):
    """
    Extracts patches from 'data'. The patches can be augmented, which means they get rotated three times
    in XY-Plane and flipped along the X-Axis. Augmentation leads to an eight-fold increase in training data.
    Parameters
    ----------
    shuffle
    data        : list(array(float))
                  List of images with dimensions 'SZYX' or 'SYX' with optional C
    axes        : str
                  Possible dimesions include S(number of samples), ZYX(dimesions of a single sample),
                  C(channel, can be singleton dimension). E.g., SYXC in case of 2D data with S samples of shape YX
    num_patches : int, optional(default=None)
                  Number of patches to extract per image. If 'None', as many patches as fit i nto the
                  dimensions are extracted.
    shape       : tuple(int), optional(default=(256, 256))
                  Shape of the extracted patches.
    augment     : bool, optional(default=True)
                  Rotate the patches in XY-Plane and flip them along X-Axis. This only works if the patches are square in XY.
    Returns
    -------
    patches : array(float)
              Numpy-Array containing all patches (randomly shuffled along S-dimension).
              The dimensions are 'SZYXC' or 'SYXC'
    """

    patches = extract_patches(data, axes=axes, num_patches=num_patches, shape=shape)
    if shape[-2] == shape[-1]:
        if augment:
            patches = augment_patches(patches=patches, axes=axes)
    else:
        if augment:
            print("XY-Plane is not square. Omit augmentation!")
    if shuffle:
        np.random.shuffle(patches)

    return patches


def extract_patches(data, axes: str, num_patches=None, shape=(256, 256)):
    """
    Extract patches from numpy array with axes S(X)YX(C), with S a singleton
    dimension.

    Parameters
    ----------
    axes
    data
    num_patches
    shape

    Returns
    -------

    """
    # get indices of X and Y axes
    ind_x = axes.find('X')
    ind_y = axes.find('Y')

    if num_patches is None:
        patches = []
        if 'Z' not in axes:
            if data.shape[ind_y] > shape[0] and data.shape[ind_x] > shape[1]:
                for y in range(0, data.shape[ind_y] - shape[0] + 1, shape[0]):
                    for x in range(0, data.shape[ind_x] - shape[1] + 1, shape[1]):
                        patches.append(data[:, y:y + shape[0], x:x + shape[1]])

                return np.concatenate(patches)
            elif data.shape[ind_y] == shape[0] and data.shape[ind_x] == shape[1]:
                return data
            else:
                print('Incorrect shape')
        else:
            # index of the Z axis
            ind_z = axes.find('Z')

            # pad X, Y and Z
            target = int((max(16, 2 ** np.ceil(np.log2(data.shape[ind_z])))))
            pad = target - data.shape[ind_z]

            padding = (int(np.ceil(pad / 2)), int(np.floor(pad / 2)))
            if 'C' in axes:
                padded_data_list = []
                for c in range(data.shape[-1]):
                    # assume c is the last dimension
                    padded_c = np.pad(data[..., c], padding, 'constant')[..., np.newaxis]
                    padded_data_list.append(padded_c)

                # concatenate along the C dimension
                padded_data = np.concatenate(padded_data_list, axis=-1)
            else:
                padded_data = np.pad(data, padding, 'constant')

            if padded_data.shape[ind_z] >= shape[0] and \
                    padded_data.shape[ind_y] >= shape[1] and \
                    padded_data.shape[ind_x] >= shape[2]:
                for z in range(0, padded_data.shape[ind_z] - shape[0] + 1, shape[0]):
                    for y in range(0, padded_data.shape[ind_y] - shape[1] + 1, shape[1]):
                        for x in range(0, padded_data.shape[3] - shape[2] + 1, shape[2]):
                            patches.append(padded_data[:, z:z + shape[0], y:y + shape[1], x:x + shape[2], ...])

                return np.concatenate(patches)
            elif padded_data.shape[ind_z] == shape[0] and \
                    padded_data.shape[ind_y] == shape[1] and \
                    padded_data.shape[ind_x] == shape[2]:
                return padded_data
            else:
                print('Incorrect shape')
    else:
        patches = []
        if 'Z' not in axes:
            for i in range(num_patches):
                y = np.random.randint(0, data.shape[ind_y] - shape[0] + 1)
                x = np.random.randint(0, data.shape[ind_x] - shape[1] + 1)
                patches.append(data[0, y:y + shape[0], x:x + shape[1], ...])

            if len(patches) > 1:
                return np.stack(patches)
            else:
                return np.array(patches)[np.newaxis]
        else:
            # index of the Z axis
            ind_z = axes.find('Z')

            for i in range(num_patches):
                z = np.random.randint(0, data.shape[ind_z] - shape[0] + 1)
                y = np.random.randint(0, data.shape[ind_y] - shape[1] + 1)
                x = np.random.randint(0, data.shape[ind_z] - shape[2] + 1)
                patches.append(data[0, z:z + shape[0], y:y + shape[1], x:x + shape[2], ...])

            if len(patches) > 1:
                return np.stack(patches)
            else:
                return np.array(patches)[np.newaxis]


def augment_patches(patches, axes: str):
    """
    Performs an 8-fold augmentation (3 rotations in (XY), 1 flip) of an array. The array should have axes S(Z)YX(C).

    Parameters
    ----------
    patches: Patches along the 1st dimension of a numpy array
    axes: S, (Z), X, Y and (C)

    Returns
    -------
    Augmented patches with axes S(Z)YX(C).
    """
    if 'S' not in axes:
        raise ValueError('S should be in the axes.')

    ind_S = axes.find('S')
    ind_x = axes.find('X')
    ind_y = axes.find('Y')

    # rotations
    X_rot = [np.rot90(patches, i, (ind_y, ind_x)) for i in range(4)]
    X_rot = np.concatenate(X_rot, axis=ind_S)

    # flip
    X_flip = np.flip(X_rot, axis=ind_y)

    # return concatenated augmentations along S axis
    return np.concatenate([X_rot, X_flip], axis=ind_S)
