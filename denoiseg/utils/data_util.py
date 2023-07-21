import numpy as np
from tifffile import imread, imsave, imwrite
from glob import glob
import random
import tqdm
from matplotlib import pyplot as plt
import matplotlib
from sklearn.feature_extraction import image
import cv2
import os


def change_from_rgb(patch, rgb_to_index):
    mask = np.zeros_like(patch)

    for i in rgb_to_index.keys():
        mask[patch == i] = rgb_to_index[i]

    return mask


def generate_patches(imgs, lbls, rgb_to_index, patch_size, max_patches, random_state=42):
    imgs_patches = []
    labels_patches = []

    for lbl in lbls:
        lbl_ = change_from_rgb(cv2.imread(lbl, 0), rgb_to_index)
        patchesmasks = image.extract_patches_2d(lbl_, patch_size=patch_size, max_patches=max_patches, random_state=random_state)

        for j in range(patchesmasks.shape[0]):
            labels_patches.append(patchesmasks[j])

    for img in imgs:
        img_ = cv2.imread(img, 0)
        patchesimgs = image.extract_patches_2d(img_, patch_size=patch_size, max_patches=max_patches, random_state=random_state)
        
        for j in range(patchesimgs.shape[0]):
            imgs_patches.append(patchesimgs[j])

    return imgs_patches, labels_patches


def split_data(imgs_patches, labels_patches, len_val):
    ind_val = sorted(random.sample(range(len(labels_patches)), len_val))

    X_val = [imgs_patches[i] for i in ind_val]
    Y_val = [labels_patches[i] for i in ind_val[:5]]

    ind_x_train = np.setdiff1d(np.arange(len(imgs_patches)), ind_val)
    ind_y_train = np.setdiff1d(np.arange(len(labels_patches)), ind_val)


    X_train = [imgs_patches[i] for i in sorted(ind_x_train)]
    Y_train = [labels_patches[i] for i in sorted(ind_y_train)]

    print('- training:       %3d' % len(X_train))
    print('- validation:     %3d' % len(X_val))

    return X_train, Y_train, X_val, Y_val


def make_dataset(imgs_dir, lbls_dir, ext_img, ext_lbl, rgb_to_index, patch_size, max_patches, len_val):
    imgs = sorted(glob(os.path.join(imgs_dir, '*' + ext_img)))
    lbls = sorted(glob(os.path.join(lbls_dir, '*' + ext_lbl)))

    imgs_patches, labels_patches = generate_patches(imgs, lbls, rgb_to_index, patch_size, max_patches, random_state=42)

    X_train, Y_train, X_val, Y_val = split_data(imgs_patches, labels_patches, len_val)

    return X_train, Y_train, X_val, Y_val