import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from lib.model.utils.config import cfg
from model.rpn.rpn import _RPN

from model.roi_layers import ROIAlign, ROIPool

# from model.roi_pooling.modules.roi_pool import _RoIPooling
# from model.roi_align.modules.roi_align import RoIAlignAvg

from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta
import cv2 as cv2

class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)

        # self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        # self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)
        self.RCNN_roi_align = ROIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0, 0)

    def forward(self, im_data, im_info, gt_boxes, num_boxes, tgt_im_data, tgt_im_info, tgt_gt_boxes, tgt_num_boxes):
        batch_size, channel, width, height = im_data.size()

        # img_size = im_data.size()

        # print('gt boxes size, ', gt_boxes.size(), tgt_gt_boxes.size())

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        # gt_boxes = None
        num_boxes = num_boxes.data


        # feed image data to base model to obtain base feature map
        base_feat_1 = self.RCNN_base1(im_data)

        base_feat = self.RCNN_base2(base_feat_1)

        ds_feat_src = self.ds_enc_s(base_feat_1)





        # print('base feat ds, ', ds_feat_src.size())
        # print('base feat di, ', base_feat.size())

        # heat_di = base_feat
        # heat_ds = ds_feat_src


        # cv2.imwrite('demo-feat-vis-di.png', heat_di)
        # cv2.imwrite('demo-feat-vis-ds.png', heat_ds)

        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            # print('gt boxes shape, ', gt_boxes.size())
            # roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            # rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data
            #
            # rois_label = Variable(rois_label.view(-1).long())

            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))

        # feed pooled features to top model
        # print('pooled feat shape', pooled_feat.size())
        pooled_feat = self._head_to_tail(pooled_feat)
        # print('pooled feat shape', pooled_feat.size())

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        # cls_score = self.cosine_prototype_classifier(pooled_feat, self.RCNN_cls_score)
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)


        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)


        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)



        """ =================== for target =========================="""
        tgt_im_info = tgt_im_info.data  # (size1,size2, image ratio(new image / source image) )
        tgt_gt_boxes = tgt_gt_boxes.data
        tgt_num_boxes = tgt_num_boxes.data

        src_size = (im_data.size()[2], im_data.size()[3])
        # tgt_im_data = F.interpolate(tgt_im_data, src_size, mode='bilinear')

        # feed image data to base model to obtain base feature map
        tgt_base_feat_1 = self.RCNN_base1(tgt_im_data)
        tgt_base_feat = self.RCNN_base2(tgt_base_feat_1)

        ds_feat_tgt = self.ds_enc_s(tgt_base_feat_1)

        tgt_rois, _, _ = \
            self.RCNN_rpn(tgt_base_feat, tgt_im_info, tgt_gt_boxes, tgt_num_boxes)

        tgt_rois = Variable(tgt_rois)

        # print('src roi size', rois.size())
        # print('tgt roi size', tgt_rois.size())

        if cfg.POOLING_MODE == 'align':
            tgt_pooled_feat_tensor = self.RCNN_roi_align(tgt_base_feat, tgt_rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            tgt_pooled_feat_tensor = self.RCNN_roi_pool(tgt_base_feat, tgt_rois.view(-1, 5))

        tgt_pooled_feat = self._head_to_tail(tgt_pooled_feat_tensor)

        bbox_pred_tgt = self.RCNN_bbox_pred(tgt_pooled_feat)

        cls_score_tgt = self.RCNN_cls_score(tgt_pooled_feat)
        cls_prob_tgt = F.softmax(cls_score_tgt, 1)



        """ =================== collect features =========================="""

        heat_di = self.get_feat_map_vis2(base_feat, size=(width, height))
        heat_ds = self.get_feat_map_vis2(ds_feat_src, size=(width, height))

        heat_di_tgt = self.get_feat_map_vis2(tgt_base_feat, size=(width, height))
        heat_ds_tgt = self.get_feat_map_vis2(ds_feat_tgt, size=(width, height))






        return tgt_rois,  cls_prob_tgt, bbox_pred_tgt, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, \
                heat_di, heat_ds, heat_di_tgt, heat_ds_tgt

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()


    def cosine_prototype_classifier(self, x, classifier):

        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)

        cosine_scale = 20.0

        # normalize the input x along the `input_size` dimension
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 1e-5)

        # normalize weight
        temp_norm = torch.norm(classifier.weight.data,p=2, dim=1).unsqueeze(1).expand_as(classifier.weight.data)
        classifier.weight.data = classifier.weight.data.div(temp_norm + 1e-5)
        cos_dist = classifier(x_normalized)
        scores = cosine_scale * cos_dist
        return scores



    def get_feat_map_vis(self, feat, size):

        # print('feat size, ', feat.size())

        feat = F.interpolate(feat, size = size, mode = 'bilinear')

        feat_avg = torch.mean(feat, dim=1)

        feat_avg_norm = F.sigmoid(feat_avg).squeeze(0)

        # print('feat avg size, ', feat_avg_norm.size())


        featnp = feat_avg_norm.data.cpu().numpy()

        heatmap = cv2.applyColorMap(np.uint8(255 * featnp), cv2.COLORMAP_JET)

        # cv2.imwrite('demo-feat-vis.png', heatmap)

        return heatmap

    # def show_cam_on_image(self, img, mask):
    #     heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    #     heatmap = np.float32(heatmap) / 255
    #     cam = heatmap + np.float32(img)
    #     cam = cam / np.max(cam)
    #     return np.uint8(255 * cam)


    def append_feat_to_list(self, feat, lis, classes, idx):
        num_obj = len(idx)

        for i in range(num_obj):

            cls_id = classes[idx[i]]

            feat_i = feat[idx[i]].data.cpu().numpy()

            lis[cls_id-1].append(feat_i)


        return lis


    def get_feat_map_vis2(self, feat, size):


        feat = F.interpolate(feat, size = size, mode = 'bilinear').squeeze(0)


        # feat_select, idx = torch.max(feat, dim=0)
        feat_select = torch.mean(feat, dim=0)



        featnp = feat_select.data.cpu().numpy()

        featnp = np.maximum(featnp, 0)

        featnp = featnp - np.min(featnp)
        featnp = featnp / np.max(featnp)


        heatmap = cv2.applyColorMap(np.uint8(255 * featnp), cv2.COLORMAP_JET)

        # cv2.imwrite('demo-feat-vis.png', heatmap)

        return heatmap

