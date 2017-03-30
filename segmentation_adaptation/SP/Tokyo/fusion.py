import numpy as np
import os
import pdb
import cv2

dm_list = [x for x in sorted(os.listdir('.')) if x.endswith('dm.jpg')]
SP_list = [x for x in sorted(os.listdir('.')) if x.endswith('nC=200.jpg')]

for dm_img in dm_list:
    img_name = dm_img.split('_dm')[0]
    SP_img = img_name + '_SP_nC=200.jpg'
    DM = cv2.resize(cv2.imread(dm_img), (1126,563), interpolation=cv2.INTER_NEAREST)
    SP = cv2.resize(cv2.imread(SP_img), (1126,563), interpolation=cv2.INTER_NEAREST)

    output = (0.5 * DM + 0.5 * SP).astype(np.uint8)
    cv2.imwrite(img_name + '_fuse.jpg', output)


