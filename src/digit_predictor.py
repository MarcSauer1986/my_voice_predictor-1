import sys
import torch

import numpy as np


import imageio
from skimage.transform import rescale, resize

import preprocessing as prep

if __name__ == '__main__':
    # Get the total number of args passed to the prediction.py
    total_args = len(sys.argv)
    if total_args == 3:
        # Get the arguments list
        cmdargs = sys.argv
        model_file = cmdargs[1]
        image_file = cmdargs[2]

    else:
        print('ERROR: incorrect arguments!')
        print('predict.py <model_file> <data>')
        sys.exit(-1)

    dtype = torch.float
    device = torch.device("cpu")

    im = imageio.imread(image_file)

    print(im.shape)

    scaled_image = rescale(image=im, scale=16 / 512)

    scaled_image *= -1
    scaled_image += 0.5
    scaled_image /= np.abs(scaled_image).max()

    x = torch.tensor(scaled_image, device=device, dtype=dtype).reshape((1,256))

    model = torch.load(model_file)

    outputs = model(x)
    y_pred = outputs.data.argmax().item()
    print(y_pred)

    softmax = torch.nn.Softmax()
    y_prob = softmax(model(x)).max().item()
    print(y_prob)
