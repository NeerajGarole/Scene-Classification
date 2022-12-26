from os.path import join

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import util
import visual_words
import visual_recog
from opts import get_opts


def main():
    opts = get_opts()

    # # Q1.1
    # img_path = join(opts.data_dir, 'aquarium/sun_aztvjgubyrgvirup.jpg')
    #
    # # img_path = img_path.replace("\\", "/")     # I need to add this in Windows 11 to get exact path
    #
    # img = Image.open(img_path)
    # img = np.array(img).astype(np.float32) / 255
    # print(img)
    # print(opts)
    # filter_responses = visual_words.extract_filter_responses(opts, img)
    # # util.display_filter_responses(opts, filter_responses)
    #
    # # Q1.2
    n_cpu = util.get_num_CPU()
    visual_words.compute_dictionary(opts, n_worker=n_cpu)
    #
    # # Q1.3
    # img_path = join(opts.data_dir, 'aquarium/sun_aairflxfskjrkepm.jpg')
    # img_path = img_path.replace("\\", "/")
    # img = Image.open(img_path)
    # img = np.array(img).astype(np.float32)/255
    # dictionary = np.load(join(opts.out_dir, 'dictionary.npy'))
    # wordmap = visual_words.get_visual_words(opts, img, dictionary)
    # util.visualize_wordmap(wordmap)
    #
    #
    # img_path = join(opts.data_dir, 'desert/sun_adpbjcrpyetqykvt.jpg')
    # img_path = img_path.replace("\\", "/")
    # img = Image.open(img_path)
    # img = np.array(img).astype(np.float32) / 255
    # dictionary = np.load(join(opts.out_dir, 'dictionary.npy'))
    # wordmap = visual_words.get_visual_words(opts, img, dictionary)
    # util.visualize_wordmap(wordmap)
    #
    # img_path = join(opts.data_dir, 'laundromat/sun_aaprcnhpdrhlnhji.jpg')
    # img_path = img_path.replace("\\", "/")
    # img = Image.open(img_path)
    # img = np.array(img).astype(np.float32) / 255
    # dictionary = np.load(join(opts.out_dir, 'dictionary.npy'))
    # wordmap = visual_words.get_visual_words(opts, img, dictionary)
    # util.visualize_wordmap(wordmap)
    #
    #
    # Q2.1-2.4
    n_cpu = util.get_num_CPU()
    visual_recog.build_recognition_system(opts, n_worker=n_cpu)

    # Q2.5
    n_cpu = util.get_num_CPU()
    conf, accuracy = visual_recog.evaluate_recognition_system(opts, n_worker=n_cpu)

    print(conf)
    print("Accuracy = ", accuracy)
    np.savetxt(join(opts.out_dir, 'confmat.csv'), conf, fmt='%d', delimiter=',')
    np.savetxt(join(opts.out_dir, 'accuracy.txt'), [accuracy], fmt='%g')

    q26 = open(join(opts.out_dir, "q26.txt")).read().splitlines()
    print(q26)
    for i in q26:
        j = i + "\n"
    j=j.replace(" ","\n")
    np.savetxt(join(opts.out_dir, 'q26a.txt'), [j], fmt='%s')


if __name__ == '__main__':
    main()
