from decalib.deca import DECA
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg
from decalib.datasets import detectors
from decalib.head_indices import head_indices

import h5py
import argparse
import tqdm
from skimage.io import imread
from skimage.transform import estimate_transform, warp
import numpy as np
import torch
from pathlib import Path


crop_size = 224


def iter_batched(iterable, batchsize):
    it = iter(iterable)
    while True:
        ret = [*zip(range(batchsize),it)]
        ret = [ x for _,x in ret ]
        if not ret:
            break
        yield ret

def soa2aos(d):
    keys = d.keys()
    return [ dict(zip(keys,v)) for v in zip(*d.values()) ]

def aos2soa(structs):
    assert len(structs)
    return { k:[ s[k] for s in structs ] for k in structs[0].keys() }


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


def crop_policy_bbox_scale(image, bbox, bbox_type):
    left = bbox[0]; right=bbox[2]
    top = bbox[1]; bottom=bbox[3]

    old_size, center = bbox2point(left, right, top, bottom, type=bbox_type)
    if old_size < 64:
        return None, None, None

    size = int(old_size*1.25)
    src_pts = np.array([
        [center[0]-size/2, center[1]-size/2], 
        [center[0]-size/2, center[1]+size/2], 
        [center[0]+size/2, center[1]-size/2]
    ])
    
    dst_pts = np.array([[0,0], [0,crop_size - 1], [crop_size - 1, 0]])
    tform = estimate_transform('similarity', src_pts, dst_pts)

    dst_image = warp(image, tform.inverse, output_shape=(crop_size, crop_size))
    return dst_image, size, center


def crop_policy_original(image, bbox, bbox_type):
    h, w, _ = image.shape
    assert h == crop_size
    assert w == crop_size
    size = float(h)
    center = np.array([crop_size/2., crop_size/2.])
    return image, size, center


def compute_input(face_detector, imagefolder, imagename, crop_policy):
    image = np.array(imread(Path(imagefolder) / imagename))
    if len(image.shape) == 2:
        image = image[:,:,None].repeat(1,1,3)
    h, w, _ = image.shape
    if h < 64 or w < 64:
        return None

    bbox, bbox_type = face_detector.run(image)
    if len(bbox) < 4:
        return None

    image = image/255.

    dst_image, size, center = crop_policy(image, bbox, bbox_type)
    if dst_image is None:
        return None

    dst_image = dst_image.transpose(2,0,1)

    # FIXME: transform bounding box to the crop region?!

    return {
        'image': dst_image,
        'filename': imagename,
        'box' : bbox,
        'crop' : np.array([ center[0], center[1], size ], dtype=np.float32)
    }


class TransfromAndStoreBadItems(object):
    def __init__(self, seq, mapfunc):
        self.badlist = []
        self.seq = seq
        self.mapfunc = mapfunc
    def __iter__(self):
        for item in self.seq:
            out = self.mapfunc(item)
            if out is None:
                self.badlist.append(item)
            else:
                yield out


def removed_already_processed_files(hdf5cachefilename, filenames):
    if not Path(hdf5cachefilename).is_file():
        return filenames
    with h5py.File(hdf5cachefilename, 'r') as hdf5cache:
        exclude = frozenset()
        if 'fits/filename' in hdf5cache:
            exclude |= frozenset(fn.decode('ascii') for fn in hdf5cache['fits/filename'][...])
        if 'excluded' in hdf5cache:
            exclude |= frozenset(fn.decode('ascii') for fn in hdf5cache['excluded'][...])
        if exclude:
            print (f"Excluding {len(exclude)} images which are already processed")
            filenames = list(frozenset(filenames) - exclude)
    return filenames



def append_or_create_dataset(f, name, data, max_size = 4000000):
    data = np.array(data)
    if data.dtype in (np.uint8, np.int32):
        v = data
    elif data.dtype==np.float32:
        v = data.astype(np.float16)
    elif isinstance(data[0], str):
        assert all(len(s)<32 for s in data)
        v = np.array([x.encode('ascii') for x in data], dtype = 'S32')
    else:
        assert False, "Bad data "+str(data)
    if name not in f:
        f.create_dataset(name, shape=v.shape, dtype=v.dtype, chunks=True, maxshape=(max_size,)+v.shape[1:], compression="gzip", data = v)
    else:
        f[name].resize((f[name].shape[0] + v.shape[0]), axis=0)
        f[name][-v.shape[0]:] = v


def save_batch_to_hdf5(f, batch):
    for k,v in batch.items():
        append_or_create_dataset(f, k, v)
        N = f[k].shape[0]
    assert all(ds.shape[0]==N for ds in f.values())
    f.file.flush()


def run_deca_on_list_of_inputs(batch, deca):
    batch = aos2soa(batch)
    batch = { k:np.array(v) for (k,v) in batch.items() }
    images = torch.from_numpy(batch['image']).float().to(deca.device)
    with torch.no_grad():
        codedict = deca.encode(images)
        opdict = deca.decode(codedict,
            rendering=True, vis_lmk=True, return_vis=False, use_detail=False)
        #deca.save_obj('/tmp/bad.obj', opdict)
        v = opdict['trans_verts'][:,head_indices,:2]
        box = torch.cat([torch.amin(v, dim=1), torch.amax(v, dim=1)], dim=-1)
    result = {
        **{ k:v.cpu().numpy() for (k,v) in codedict.items() },
        **{k:batch[k] for k in 'filename box crop'.split()},
        **{k:opdict[k].cpu().numpy() for k in 'landmarks2d landmarks3d joint_positions'.split() },
        'head_box' : box.cpu().numpy()
    }
    del result['images']
    return result

def compute_filenames_from_file(args):
    return [ s.strip() for s in open(args.listfile,'r').readlines()]

def compute_filenames_from_glob(args):
    root = Path(args.imagefolder)
    result = list(root.glob('**/*.png')) + list(root.glob('**/*.jpg'))
    result = [ str(p.relative_to(root)) for p in result ]
    return result


def main():
    parser = argparse.ArgumentParser(description='DECA: Fit and store code')
    parser.add_argument("listfile", type=str, help="File with list of images")
    parser.add_argument("imagefolder", type=str, help="Root folder where images are")
    parser.add_argument('savefile', type=str, help="Output filename")
    parser.add_argument('--device', default='cuda', type=str,
                        help='set device, cpu for using cpu' )
    parser.add_argument('-n','--num', type=int, help="How many images to process", default=0)
    parser.add_argument('-b','--batchsize', type=int, help="Batch size", default=32)
    parser.add_argument('--no-crop', dest='crop', action='store_false', default=True)
    args = parser.parse_args()

    compute_filenames = compute_filenames_from_glob if args.listfile=='-' else compute_filenames_from_file
    crop_policy = crop_policy_bbox_scale if args.crop else crop_policy_original

    filenames = compute_filenames(args)
    filenames = removed_already_processed_files(args.savefile, filenames)
    filenames = sorted(filenames)
    if args.num:
        filenames = filenames[:args.num]

    device = args.device

    face_detector = detectors.FAN()
    deca_cfg.model.use_tex = False
    deca = DECA(config = deca_cfg, device=device)

    with h5py.File(args.savefile, 'a') as f:
        fitsgroup = f.require_group('fits')
        inputgenerator = TransfromAndStoreBadItems(
            tqdm.tqdm(filenames),
            lambda filename: compute_input(face_detector, args.imagefolder, filename, crop_policy)
        )
        for batch in iter_batched(inputgenerator, args.batchsize):
            outputs = run_deca_on_list_of_inputs(batch, deca)
            save_batch_to_hdf5(fitsgroup, outputs)
            if inputgenerator.badlist:
                append_or_create_dataset(f, 'excluded', inputgenerator.badlist)
                inputgenerator.badlist.clear()


def test1():
    face_detector = detectors.MTCNN()
    out = compute_input(face_detector,
        '/mnt/BigData/head-tracking-datasets/VGG-Face2/train/imgs',
        'n001200/0001_02.jpg')
    print(out)

def test2():
    face_detector = detectors.MTCNN()
    deca_cfg.model.use_tex = False
    deca = DECA(config = deca_cfg, device='cpu')
    out1 = compute_input(face_detector,
        '/mnt/BigData/head-tracking-datasets/VGG-Face2/train/imgs',
        'n001200/0001_02.jpg')
    out2 = compute_input(face_detector,
        '/mnt/BigData/head-tracking-datasets/VGG-Face2/train/imgs',
        'n001200/0001_02.jpg')
    batch = [ out1, out2 ]
    outputs = run_deca_on_list_of_inputs(batch, deca)
    print (outputs)
    # with open('exampleoutput.pkl','wb') as f:
    #     pickle.dump(outputs, f)


def test3():
    thingy = TransfromAndStoreBadItems([1,2,-3,-4,5,6], lambda x: x if x>0 else None)
    out = [ *thingy ]
    assert out == [1,2,5,6]
    assert thingy.badlist == [-3,-4]


if __name__ == '__main__':
    #test1()
    #test2()
    #test3()
    main()