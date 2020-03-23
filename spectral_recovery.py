#
# ...    Imports
#

import os

# misc
import h5py
import cv2 as cv
from mspec.demunic.models.bmodel import load_model
import custom_networks

#
# ...    Settings
#

# all images within this folder will be reconstructed
path_model = r'model_clean'
path_data = r'E:\DataSets\NTIRE2020\test_clean'
path_out = 'out'     # output path to store the computed spectral images in

if not os.path.isdir(path_out):
    os.mkdir(path_out)

model = load_model(path_model, custom_networks)
model.initialize_from_checkpoint()


# load images
filenames = [f for f in os.listdir(path_data)]

#
# ...    Evaluation
#
for ind_img in range(len(filenames)):
    fn = filenames[ind_img]
    print('Running spectral recovery on image {}.'.format(fn))

    # load the rgb image
    filename = os.path.join(path_data, fn)
    img_rgb = cv.imread(filename)
    img_rgb = cv.cvtColor(img_rgb, cv.COLOR_BGR2RGB)
    img_rgb = img_rgb.transpose(2,0,1)

    # run spectral recovery
    spec_pred = model.execute(img_rgb)

    # save spectral reconstruction
    print('Storing spectral reconstruction.')
    name_out = os.path.join(path_out, filenames[ind_img][:12] + '.mat')     # assuming ntire2020 naming convention
    hf = h5py.File(name_out, 'w')
    hf.create_dataset('cube', data=spec_pred.transpose(1,2,0))
    hf.close()