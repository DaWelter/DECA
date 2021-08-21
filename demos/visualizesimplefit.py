from decalib.deca import DECA
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg
from decalib.datasets import detectors

import os
import h5py
import argparse
import tqdm
from skimage.io import imread
from skimage.transform import estimate_transform, warp
import numpy as np
import torch
from pathlib import Path
import cv2

crop_size = 256


def h52pytorchf32(ds,i, device):
    return torch.from_numpy(np.array(ds[i:i+1,...], dtype=np.float32)).to(device)


def bbox2point(left, right, top, bottom, type='bbox'):
    ''' bbox from detector and landmarks are different
    '''
    if type=='kpt68':
        old_size = (right - left + bottom - top)/2*1.1
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
    elif type=='bbox':
        old_size = (right - left + bottom - top)/2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0  + old_size*0.12])
    else:
        raise NotImplementedError
    return old_size, center


def crop_image(imagefolder, imagename, bbox, crop):
    image = np.array(imread(Path(imagefolder) / imagename))
    if len(image.shape) == 2:
        image = image[:,:,None].repeat(1,1,3)

    center, size = crop[:2], crop[2]

    src_pts = np.array([
        [center[0]-size/2, center[1]-size/2], 
        [center[0] - size/2, center[1]+size/2], 
        [center[0]+size/2, center[1]-size/2]])

    dst_pts = np.array([[0,0], [0,crop_size - 1], [crop_size - 1, 0]])
    tform = estimate_transform('similarity', src_pts, dst_pts)

    image = image/255.

    dst_image = warp(image, tform.inverse, output_shape=(crop_size, crop_size))
    dst_image = dst_image.transpose(2,0,1)
    return dst_image


def main():
    parser = argparse.ArgumentParser(description='DECA: Visualize store code')
    parser.add_argument('savefile', type=str, help="Filename of the parameter file")
    parser.add_argument("imagefolder", type=str, help="Root folder where images are")
    parser.add_argument("savefolder", type=str, help="Where to save the images")
    parser.add_argument('--device', default='cuda', type=str,
                        help='set device, cpu for using cpu' )
    parser.add_argument('-s', type=int, help="Start index", default=0)
    parser.add_argument('-e', type=int, help="End index", default=0)
    args = parser.parse_args()
    device = args.device

    deca_cfg.model.use_tex = False
    deca = DECA(config = deca_cfg, device=device)

    with h5py.File(args.savefile, 'r') as f:
        fitgroup = f['fits']
        N = len(fitgroup['filename'])
        if args.e <= 0:
            args.e = N
        for i in tqdm.trange(args.s, args.e):
            codedict = { k:h52pytorchf32(fitgroup[k],i,device) for k in 'cam detail exp shape pose light tex'.split() }
            imagefilename = fitgroup['filename'][i].decode('ascii')
            
            image = crop_image(args.imagefolder, imagefilename, fitgroup['box'][i][...], fitgroup['crop'][i][...])
            codedict['images'] = torch.from_numpy(image)[None,...].float().to(device)
            opdict, visdict = deca.decode(codedict)
            
            outname = os.path.join(args.savefolder, os.path.splitext(imagefilename)[0])
            os.makedirs(os.path.dirname(outname), exist_ok=True)
            
            cv2.imwrite(outname+'_vis.jpg', deca.visualize(visdict))


if __name__ == '__main__':
    main()