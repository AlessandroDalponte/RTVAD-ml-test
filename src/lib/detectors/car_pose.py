from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch

try:
    from external.nms import soft_nms_39
except:
    print('NMS not imported! If you need it,'
          ' do \n cd $CenterNet_ROOT/src/lib/external \n make')
from models.decode import multi_pose_decode
from models.decode import car_pose_decode,car_pose_decode_faster
from models.utils import flip_tensor, flip_lr_off, flip_lr
from utils.image import get_affine_transform
from utils.post_process import multi_pose_post_process
from utils.post_process import car_pose_post_process
from utils.debugger import Debugger

from .base_detector import BaseDetector


class CarPoseDetector(BaseDetector):
    def __init__(self, opt):
        super(CarPoseDetector, self).__init__(opt)
        self.flip_idx = opt.flip_idx

    def process(self, images,meta, return_time=False):
        with torch.no_grad():
            torch.cuda.synchronize()
            output = self.model(images)[-1]
            output['hm'] = output['hm'].sigmoid_()
            if self.opt.hm_hp and not self.opt.mse_loss:
                output['hm_hp'] = output['hm_hp'].sigmoid_()

            reg = output['reg'] if self.opt.reg_offset else None
            hm_hp = output['hm_hp'] if self.opt.hm_hp else None
            hp_offset = output['hp_offset'] if self.opt.reg_hp_offset else None
            torch.cuda.synchronize()
            forward_time = time.time()

            if self.opt.flip_test:
                output['hm'] = (output['hm'][0:1] + flip_tensor(output['hm'][1:2])) / 2
                output['wh'] = (output['wh'][0:1] + flip_tensor(output['wh'][1:2])) / 2
                output['hps'] = (output['hps'][0:1] +
                                 flip_lr_off(output['hps'][1:2], self.flip_idx)) / 2
                hm_hp = (hm_hp[0:1] + flip_lr(hm_hp[1:2], self.flip_idx)) / 2 \
                    if hm_hp is not None else None
                reg = reg[0:1] if reg is not None else None
                hp_offset = hp_offset[0:1] if hp_offset is not None else None
            if self.opt.faster==True:
                dets = car_pose_decode_faster(
                    output['hm'], output['hps'], output['dim'], output['rot'], prob=output['prob'],K=self.opt.K, meta=meta, const=self.const)
            else:
                dets = car_pose_decode(
                    output['hm'], output['wh'], output['hps'],output['dim'],output['rot'],prob=output['prob'],
                    reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=self.opt.K,meta=meta,const=self.const)

        if return_time:
            return output, dets, forward_time
        else:
            return output, dets

    def post_process(self, dets, meta, scale=1):
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets = car_pose_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'])
        for j in range(1,2):#, self.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 41)
            # import pdb; pdb.set_trace()
            dets[0][j][:, :4] /= scale
            dets[0][j][:, 5:23] /= scale
        return dets[0]

    def merge_outputs(self, detections):
        results = {}
        results[1] = np.concatenate(
            [detection[1] for detection in detections], axis=0).astype(np.float32)
        if self.opt.nms or len(self.opt.test_scales) > 1:
            soft_nms_39(results[1], Nt=0.5, method=2)
        results[1] = results[1].tolist()
        return results

    def debug(self, debugger, images, dets, output, scale=1):
        dets = dets.detach().cpu().numpy().copy()
        dets[:, :, :4] *= self.opt.down_ratio
        dets[:, :, 5:39] *= self.opt.down_ratio
        img = images[0].detach().cpu().numpy().transpose(1, 2, 0)
        img = np.clip(((
                               img * self.std + self.mean) * 255.), 0, 255).astype(np.uint8)
        pred = debugger.gen_colormap(output['hm'][0].detach().cpu().numpy())
        debugger.add_blend_img(img, pred, 'pred_hm')
        if self.opt.hm_hp:
            pred = debugger.gen_colormap_hp(
                output['hm_hp'][0].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hmhp')

    def show_results(self, debugger, image, results, calib, counter):
        bbox_track = []
        confidence_track = []
        class_ids_track = []
        
        debugger.add_img(image, img_id='car_pose')
        for bbox in results[1]:
            if bbox[4] > self.opt.vis_thresh:
                x1 = bbox[0]
                y1 = bbox[1]
                width = bbox[2]-bbox[0]
                height = bbox[3]-bbox[1]
                
                bbox_track.append([x1, y1, width, height])
                confidence_track.append(bbox[4])
                class_ids_track.append(2)

                debugger.add_coco_bbox(bbox[:4], bbox[40], bbox[4], img_id='car_pose')
                debugger.add_kitti_hp(bbox[5:23], img_id='car_pose')
                debugger.add_bev(bbox, img_id='car_pose',is_faster=self.opt.faster)
                final_frame = debugger.add_3d_detection(bbox, calib, img_id='car_pose')
                debugger.save_kitti_format(bbox,self.image_path,self.opt,img_id='car_pose',is_faster=self.opt.faster, counter=counter)
        
        bbox_track = np.array(bbox_track)
        confidence_track = np.array(confidence_track)
        class_ids_track = np.array(class_ids_track)

        if self.opt.vis:
            debugger.show_all_imgs(pause=self.pause)
        
        return [bbox_track, confidence_track, class_ids_track, final_frame]
