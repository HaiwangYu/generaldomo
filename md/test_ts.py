
import sys
import numpy as np
import torch
import h5py

import h5_utils as h5u

def main():
    verbose = '-v' in sys.argv

    # load ts model
    model = "ts-model/unet1024-l23-cosmic500-e50-gpu.ts"
    net = torch.jit.load(model)
    
    # input file
    infile = "test/data-0.h5"
    # input tags
    im_tags = ['frame_loose_lf0', 'frame_mp2_roi0', 'frame_mp3_roi0']    # l23

    # output file
    out = h5py.File("out-ref.h5", "w")
    
    requests = 10
    for i in range(requests):
        img = h5u.get_hwc_img(infile, i%10, im_tags, [1, 10], [800, 1600], [0, 6000], 4000) # V

        # numpy -> numpy
        img_tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])) # hwc_to_chw
        img_tensor = img_tensor.cuda()
        with torch.no_grad():
            input = img_tensor.unsqueeze(0)
            print ("input.shape: ", input.shape)
            mask = net.forward(input).squeeze().cpu().numpy()
            print('mask.shape:\n', mask.shape)

        dset = out.create_dataset(name="/%d/mask"%i, shape=mask.shape, dtype='f', data=mask)
        print('mask.shape:\n', mask.shape)



if __name__ == '__main__':
    main()