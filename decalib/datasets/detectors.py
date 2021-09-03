# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import numpy as np
import torch

class FAN(object):
    def __init__(self):
        import face_alignment
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda')
        self.fa_cpu = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cpu')

    def run(self, image):
        '''
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box list
        '''
        assert isinstance(image, np.ndarray)
        try:
            lms, _, boxes = self.fa.get_landmarks_from_image(image, return_bboxes=True)
        except Exception as e:
            print (f'GPU face detection failed with error: {str(e)}')
            try:
                lms, _, boxes = self.fa_cpu.get_landmarks_from_image(image, return_bboxes=True)
            except Exception as e:
                print (f'CPU face detection fallback also failed with error: {str(e)}')
                lms, boxes = None, None
        out = lms
        if out is None:
            return [0], 'kpt68'
        else:
            kpt = out[0].squeeze()
            left = np.min(kpt[:,0]); right = np.max(kpt[:,0]); 
            top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])
            bbox = [left,top, right, bottom]
            return bbox, 'kpt68'

class MTCNN(object):
    def __init__(self, device = 'cpu'):
        '''
        https://github.com/timesler/facenet-pytorch/blob/master/examples/infer.ipynb
        '''
        from facenet_pytorch import MTCNN as mtcnn
        self.device = device
        self.model = mtcnn(keep_all=True)
    def run(self, input):
        '''
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box
        '''
        boxes, probs, landmarks = self.model.detect(input, landmarks=True)
        boxes, _, _ = self.model.select_boxes(boxes, probs, landmarks, input)
        if boxes is None or boxes[0] is None:
            return [0], 'bbox'
        else:
            return boxes[0], 'bbox'



