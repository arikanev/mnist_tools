import numpy as np
import os
import pickle as pkl
import errno
import argparse
from PIL import Image
from resizeimage import resizeimage


# argparse
parser = argparse.ArgumentParser()
parser.add_argument('--imgnet_path', '--ip', type=int, default=128, help='path to imagenet data')
parser.add_argument('--save_path', '--sp', type=str, default=os.getcwd(), help='path to save')
parser.add_argument('--mnist_path', '--mp', type=str, default=os.path.join(os.getcwd(),'mnist'), help='path to mnist')
args = parser.parse_args()

mnist_data_path = args.mnist_path

imgnet_data_path = args.imgnet_path

mnistm_data_path = args.save_path

for i in range(3):  # train, test, val

    if i == 0:  # if train
        set_type = 'train'
    elif i == 1:  # if test
        set_type = 'test'
    else:  # if val
        set_type = 'val'

    curr_mnist_set_path = os.listdir(os.path.join(mnist_data_path, set_type + '_masks'))
    curr_mnist_set_length = len(curr_mnist_set_path)

    # create list of imagenet images, resize to 28x28

    hashable_imgnet_list = []

    imgnet_count = 0

    size = [28, 28]

    for image in os.listdir(imgnet_data_path):

        # load only as many image net images as there are mnist masks

        if imgnet_count < curr_mnist_set_length:

            # open current image net image

            im = Image.open(os.path.join(imgnet_data_path, image))

            # resize

            im = resizeimage.resize_thumbnail(im, size)

            # append current image net image to list
            hashable_imgnet_list.append(str(np.array(im).reshape(28, 28, 3).data))
            imgnet_count += 1

    print('{} imagenet images loaded'.format(imgnet_count))

    # create list of mnist masks

    hashable_mnist_list = []

    mnist_count = 0

    for image in curr_mnist_set_path:

        im = np.array(Image.open(os.path.join(os.path.join(mnist_data_path, set_type + '_masks', image))))

        # convert greyscale to RGB

        rgb_im = np.empty((28, 28, 3), dtype=np.uint8)
        rgb_im[:, :, 2] = rgb_im[:, :, 1] = rgb_im[:, :, 0] = im

        # append current mnist rgb mask to list
        hashable_mnist_list.append(str(rgb_im.data))
        mnist_count += 1

    print('{} mnist masks loaded'.format(mnist_count))

    # generate mnistm

    filenum = 0

    # convert lists to dicts for index comparisons

    mnist_mask_dict = dict((mnist_mask, index) for index, mnist_mask in enumerate(hashable_mnist_list))
    imgnet_im_dict = dict((imgnet_im, index) for index, imgnet_im in enumerate(hashable_imgnet_list))

    for mnist_mask, index in mnist_mask_dict.iteritems():

        for imgnet_im, index in imgnet_im_dict.iteritems():

            # ensure each mnist mask gets assigned a different image net image

            if imgnet_im_dict[imgnet_im] == mnist_mask_dict[mnist_mask]:
                # copy mnist_im to not override mnist image

                mnistm = np.copy(np.fromstring(mnist_mask, dtype=np.uint8).reshape(28, 28, 3))

                # generate mnistm from mnist im and imagenet im broadcasting

                mnistm[mnistm == 0] = np.fromstring(imgnet_im, dtype=np.uint8).reshape(28, 28, 3)[mnistm == 0]
                mnistm[mnistm == 255] = 255 - np.fromstring(imgnet_im, dtype=np.uint8).reshape(28, 28, 3)[mnistm == 255]

                # create train,test,val directories for images and masks

                curr_image_path = os.path.join(mnistm_data_path, set_type + '_images')
                curr_mask_path = os.path.join(mnistm_data_path, set_type + '_masks')

                if not os.path.exists(curr_image_path):
                    try:
                        os.makedirs(curr_image_path)
                        os.makedirs(curr_mask_path)
                    except OSError as exc:  # Guard against race condition
                        if exc.errno != errno.EEXIST:
                            raise

                # save curr mnist_mask as greyscale to curr mask dir
                # save curr mnistm to curr image dir

                curr_mnistm_mask = Image.fromarray(np.fromstring(mnist_mask, dtype=np.uint8).reshape(28, 28, 3)[:, :, 0])
                curr_mnistm_mask.save(os.path.join(curr_mask_path, str(filenum).zfill(5) + '.png'))

                curr_mnistm_im = Image.fromarray(mnistm)
                curr_mnistm_im.save(os.path.join(curr_image_path, str(filenum).zfill(5) + '.png'))

                print('saved im and mask number {} in {} set'.format(str(filenum).zfill(5), set_type))

                filenum += 1
