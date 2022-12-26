import os
import math
import multiprocessing
from os.path import join
from copy import copy

import numpy as np
from PIL import Image
import math

import visual_words


def get_feature_from_wordmap(opts, wordmap):
    """
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    """

    K = opts.K
    L = opts.L
    # ----- TODO -----
    hist, _ = np.histogram(wordmap, bins=np.arange(K + 1))
    hist = hist/np.sum(hist)
    return hist


def get_feature_from_wordmap_SPM(opts, wordmap):
    """
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape K*(4^(L+1) - 1) / 3
    """

    K = opts.K
    L = opts.L
    # ----- TODO -----
    row, col = wordmap.shape

    num_cell = pow(2, L)
    cell_row = math.floor(row / num_cell)
    cell_col = math.floor(col / num_cell)

    if L <= 1:
        weight = pow(2, -L)
    else:
        weight = 1 / 2

    layer1 = np.zeros((num_cell, num_cell, K))
    for row in range(num_cell):
        for col in range(num_cell):
            new_row1 = cell_row * row
            new_row2 = cell_row * (row + 1)
            new_col1 = cell_col * col
            new_col2 = cell_col * (col + 1)
            part_wordmap = wordmap[new_row1:new_row2, new_col1:new_col2]
            hist = get_feature_from_wordmap(opts, part_wordmap)
            layer1[row, col, :] = hist

    layer1 = layer1.reshape(1, -1)[0]*weight
    hist_all = np.array([], dtype=np.float32)
    hist_all = hist_all.reshape(1, 0)
    hist_all = np.append(layer1, hist_all)
    hist_all = hist_all / np.sum(hist_all)
    return hist_all


def get_image_feature(opts, img_path, dictionary):
    """
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K)
    """

    # ----- TODO -----

    img = np.array(Image.open(img_path)).astype(np.float32) / 255
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.tile(img, (1, 1, 3))
    elif img.shape[2] > 3:
        img = img[:, :, :3]
    wordmap = visual_words.get_visual_words(opts, img, dictionary)
    feature = get_feature_from_wordmap_SPM(opts, wordmap)
    return feature


def build_recognition_system(opts, n_worker=1):
    """
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    """

    data_dir = opts.data_dir
    train_files = open(join(data_dir, "train_files.txt")).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, "train_labels.txt"), np.int32)

    out_dir = opts.out_dir
    dictionary = np.load(join(out_dir, "dictionary.npy"))

    img_path = [join(data_dir, img_name) for img_name in train_files]
    args = zip([opts] * len(train_files), img_path, [dictionary] * len(train_files))
    features = multiprocessing.Pool(n_worker).starmap(get_image_feature, args)

    np.savez_compressed(join(out_dir, 'trained_system.npz'), features=features,
                        labels=train_labels, dictionary=dictionary, SPM_layer_num=opts.L)


def similarity_to_set(word_hist, histograms):
    """
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    """

    # ----- TODO -----
    num_features, _ = histograms.shape
    sim = np.full(num_features, 1) - np.sum(np.minimum(word_hist, histograms), axis=1)
    return sim


def evaluate_recognition_system(opts, n_worker=1):
    """
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    """

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, "trained_system.npz"))
    dictionary = trained_system["dictionary"]

    trained_features = trained_system['features']
    trained_labels = trained_system['labels']

    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system["SPM_layer_num"]

    test_files = open(join(data_dir, "test_files.txt")).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, "test_labels.txt"), np.int32)

    # ----- TODO -----
    test_img_len = len(test_files)
    img_path = [join(data_dir, img_name) for img_name in test_files]
    args = zip([opts] * test_img_len, img_path, [dictionary] * test_img_len)
    test_features = np.asarray(multiprocessing.Pool(n_worker).starmap(get_image_feature, args))
    np.savez_compressed(join(out_dir, 'test_system.npz'), features=test_features)

    labels_predicted = list()
    for i in range(test_img_len):
        labels_predicted.append(trained_labels[np.argmin(similarity_to_set(test_features[i, :], trained_features))])

    np.savetxt(join(opts.out_dir, 'q26.txt'), [np.asarray(labels_predicted)], fmt='%d')
    conf = np.zeros((8, 8))
    for i, j in zip(test_labels, np.asarray(labels_predicted)):
        conf[i][j] = conf[i][j] + 1
    accuracy = np.sum(np.diag(conf)) / np.sum(conf)
    return conf, accuracy

