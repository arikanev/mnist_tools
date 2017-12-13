import numpy as np
import os
import pickle as pkl
import errno
import argparse
from PIL import Image
from timeit import default_timer as timer

start = timer()  # init timer

#argparse
parser = argparse.ArgumentParser()
parser.add_argument('--threshold', '--t', type=int, default=128, help='threshold for masks')
parser.add_argument('--save_path', '--s', type=string, default=os.getcwd(), help='save path')
parser.add_argument('--pkl_path', '--p', type=string, default=os.path.join(os.getcwd(),'mnist.pkl'), help='path to mnist pkl file'
args = parser.parse_args()

# load mnist data

pkl_path = args.pkl_path
save_path = args.save_path

mnist_data = pkl.load(open(pkl_path, 'rb'))

train_im_path = os.path.join(save_path, 'train_images')
train_mask_path = os.path.join(save_path, 'train_masks')

test_im_path = os.path.join(save_path, 'test_images')
test_mask_path = os.path.join(save_path, 'test_masks')

val_im_path = os.path.join(save_path, 'val_images')
val_mask_path = os.path.join(save_path, 'val_masks')

set_type = ''

# generate images and masks for training set

print('generating images and masks for training set')

for i in range(len(mnist_data)):

    curr_filenames = [] # filenames for train, test, or val
    curr_data = mnist_data[i][0] # [i] = train, test, or val, [0] because we only want images, not labels

    for j in range(len(curr_data)):

        curr_filenames.append(str(j).zfill(5) + '.png')

        if i==0: # if train
            curr_im_path = os.path.join(train_im_path, curr_filenames[j])
            curr_mask_path = os.path.join(train_mask_path, curr_filenames[j])
            set_type = 'train'
        elif i==1: # if test
            curr_im_path = os.path.join(test_im_path, curr_filenames[j])
            curr_mask_path = os.path.join(test_mask_path, curr_filenames[j])
            set_type = 'test'
        else: # if val
            curr_im_path = os.path.join(val_im_path, curr_filenames[j])
            curr_mask_path = os.path.join(val_mask_path, curr_filenames[j])
            set_type = 'val'
        # reshape array, normalize between [0,255], convert to uint8
        curr_data_normalized = np.uint8(curr_data[j].reshape(28, 28) *
                             255.0 / curr_data[j].max())

        curr_image = Image.fromarray(curr_data_normalized)  # create Image obj
        
        #save image
        if not os.path.exists(os.path.dirname(curr_im_path)):
            try:
                os.makedirs(os.path.dirname(curr_im_path))
                os.makedirs(os.path.dirname(curr_mask_path))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        curr_image.save(curr_im_path)

        #create mask
        black_mask = curr_data_normalized < args.threshold
        white_mask = curr_data_normalized > args.threshold - 1
        curr_data_normalized[black_mask] = 0
        curr_data_normalized[white_mask] = 255
        curr_mask = Image.fromarray(curr_data_normalized)
        
        #save mask
        curr_mask.save(curr_mask_path)

        print('saved im and mask number {} in {} set'.format(str(j).zfill(5), set_type))

end = timer()  # end timer

print(end - start)
