import os
import multiprocessing
from os.path import join, isfile
from skimage import io
import numpy as np
import scipy.ndimage
import skimage.color
from PIL import Image
from sklearn.cluster import KMeans
from opts import get_opts


def image_float(img):
    return img.astype(np.float32) / 255


def extract_filter_responses(opts, img):
    """
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    """

    # ----- TODO -----
    check_type = type(img[0, 0, 0])
    if check_type != np.float32:
        img = image_float(img)
    if np.amax(img) > 1.0:
        img = image_float(img)
    if np.amin(img) < 0.0:
        img = image_float(img)

    img_shape = img.shape
    if len(img_shape) < 3:
        img = img[:, :, np.newaxis]
        img = np.tile(img, (1, 1, 3))
    elif img_shape[2] > 3:
        img = img[:, :, :3]

    lab_color = skimage.color.rgb2lab(img)
    filter_scales = opts.filter_scales
    height = img_shape[0]
    width = img_shape[1]
    tup = (height, width, 3*4*len(filter_scales))
    filter_responses = np.zeros(tup)

    for i in range(len(filter_scales)):
        for j in range(3):
            filter1 = scipy.ndimage.gaussian_filter(lab_color[:, :, j], filter_scales[i])
            filter_responses[:, :, i*4*3+j] = filter1
            filter2 = scipy.ndimage.gaussian_laplace(lab_color[:, :, j], filter_scales[i])
            filter_responses[:, :, i*4*3+j+3] = filter2
            filter3 = scipy.ndimage.gaussian_filter(lab_color[:, :, j], filter_scales[i], order=[1, 0])
            filter_responses[:, :, i*4*3+j+6] = filter3
            filter4 = scipy.ndimage.gaussian_filter(lab_color[:, :, j], filter_scales[i], order=[0, 1])
            filter_responses[:, :, i*4*3+j+9] = filter4

    return filter_responses
    pass


def compute_dictionary_one_image(args):
    """
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    """

    # ----- TODO -----
    path = args[2]
    filter_response = extract_filter_responses(get_opts(), io.imread(path).astype(np.float32) / 255)
    alpha = int(args[1])
    rand_x = np.random.choice(filter_response.shape[1], alpha)
    rand_y = np.random.choice(filter_response.shape[0], alpha)
    index = args[0]
    opts = get_opts()
    np.save(os.path.join(opts.feat_dir, str(index) + '.npy'), filter_response[rand_y, rand_x, :])


def compute_dictionary(opts, n_worker=1):
    """
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel

    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    """
    train_file = open(join(opts.data_dir, 'train_files.txt')).read().splitlines()
    img_path = [join(opts.data_dir, img_name) for img_name in train_file]

    args = zip(range(1, len(img_path) + 1), [opts.alpha] * len(img_path), img_path)
    multiprocessing.Pool(n_worker).map(compute_dictionary_one_image, args)
    filter_responses = np.array([], dtype=np.float32)
    filter_responses = filter_responses.reshape(0, 3 * 4 * len(opts.filter_scales))

    for i in os.listdir(opts.feat_dir):
        filter_responses = np.append(filter_responses, np.load(join(opts.feat_dir, i)), axis=0)

    K = opts.K
    kmeans = KMeans(n_clusters=K).fit(filter_responses)
    dictionary = kmeans.cluster_centers_

    np.save(join(opts.out_dir, 'dictionary.npy'), dictionary)


def get_visual_words(opts, img, dictionary):
    """
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)

    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    """

    # ----- TODO -----
    height = img.shape[0]
    width = img.shape[1]
    filterImg = extract_filter_responses(opts, img).reshape(height * width, dictionary.shape[1])
    # print(dictionary.shape)
    euclidean_distance = scipy.spatial.distance.cdist(filterImg, dictionary, 'euclidean')
    wordmap = np.argmin(euclidean_distance, axis=1).reshape(height, width)
    return wordmap
