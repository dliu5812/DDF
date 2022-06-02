import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from lib.model.utils.config import cfg
from lib.model.rpn.rpn import _RPN

from lib.model.roi_layers import ROIAlign, ROIPool

# from model.roi_pooling.modules.roi_pool import _RoIPooling
# from model.roi_align.modules.roi_align import RoIAlignAvg

from lib.model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer

from lib.model.utils.net_utils import _smooth_l1_loss


from lib.model.da_faster_rcnn_disent_new.DA import _ImageDADeep, grad_reverse,TripletLossMean
from lib.model.da_faster_rcnn_disent_new.LabelResizeLayer import ImageLabelResizeLayer


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


        self.RCNN_imageDA = _ImageDADeep(self.dout_base_model)


        self.ImageLabelResizeLayer = ImageLabelResizeLayer()

        self.ce_loss = nn.CrossEntropyLoss()

        self.cos_sim_official = nn.CosineSimilarity(dim=1, eps=1e-6)

        self.cos_embed_loss = nn.CosineEmbeddingLoss()

        self.l1loss = torch.nn.L1Loss()
        self.l2loss = torch.nn.MSELoss()

        self.tanh = nn.Tanh()

        self.tp_loss_mean = TripletLossMean()

        self.beta = 0.1

    def forward(self, im_data, im_info, gt_boxes, num_boxes,
                tgt_im_data, tgt_im_info, tgt_gt_boxes, tgt_num_boxes,ins_key):



        need_backprop = Variable(torch.ones(im_data.size(0)).long().cuda())

        tgt_need_backprop = Variable(torch.zeros(im_data.size(0)).long().cuda())


        # assert need_backprop.detach() == 1 and tgt_need_backprop.detach() == 0

        # src label, 1 ; tgt label, 0


        batch_size = im_data.size(0)



        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data
        need_backprop = need_backprop.data

        # feed image data to base model to obtain base feature map
        base_feat_1 = self.RCNN_base1(im_data)
        base_feat = self.RCNN_base2(base_feat_1)

        # feed base feature map tp RPN to obtain rois
        self.RCNN_rpn.train()
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
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        # print('src roi labels, ', rois_label)

        rois = Variable(rois)
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == 'align':
            pooled_feat_tensor = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat_tensor = self.RCNN_roi_pool(base_feat, rois.view(-1,5))

        pooled_feat = self._head_to_tail(pooled_feat_tensor)

        # compute bbox offset
        bbox_pred_raw = self.RCNN_bbox_pred(pooled_feat)



        # print('box loc pre, ', bbox_pred.size())

        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred_raw.view(bbox_pred_raw.size(0), int(bbox_pred_raw.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)





        """ =================== for target =========================="""

        tgt_batch_size = tgt_im_data.size(0)
        tgt_im_info = tgt_im_info.data  # (size1,size2, image ratio(new image / source image) )
        tgt_gt_boxes = tgt_gt_boxes.data
        tgt_num_boxes = tgt_num_boxes.data
        tgt_need_backprop = tgt_need_backprop.data

        src_size = (im_data.size()[2], im_data.size()[3])
        tgt_im_data = F.interpolate(tgt_im_data, src_size, mode='bilinear')

        # feed image data to base model to obtain base feature map
        tgt_base_feat_1 = self.RCNN_base1(tgt_im_data)
        tgt_base_feat = self.RCNN_base2(tgt_base_feat_1)


        """  Disentangle loss   """

        ds_feat_src = self.ds_enc_s(base_feat_1)
        # ds_feat_tgt = self.ds_enc_t(tgt_base_feat_1)
        ds_feat_tgt = self.ds_enc_s(tgt_base_feat_1)

        """  DA loss for disentanglement at the image level   """

        # image-level disent
        domain_score_src_img_di = self.RCNN_imageDA(grad_reverse(base_feat))
        domain_score_src_img_ds = self.RCNN_imageDA(ds_feat_src)

        domain_label_src_img = self.ImageLabelResizeLayer(domain_score_src_img_di, need_backprop)



        domain_score_tgt_img_di = self.RCNN_imageDA(grad_reverse(tgt_base_feat))
        domain_score_tgt_img_ds = self.RCNN_imageDA(ds_feat_tgt)

        domain_label_tgt_img = self.ImageLabelResizeLayer(domain_score_tgt_img_di, tgt_need_backprop)



        if domain_score_src_img_ds.size(-1) != domain_score_src_img_di.size(-1) or domain_score_src_img_ds.size(-2) != domain_score_src_img_di.size(-2):
            rs_size = (domain_score_src_img_di.size(-2), domain_score_src_img_di.size(-1))
            domain_score_src_img_ds = F.interpolate(domain_score_src_img_ds, size=rs_size)
            domain_score_tgt_img_ds = F.interpolate(domain_score_tgt_img_ds, size=rs_size)

        loss_DA_img_dis = self.ce_loss(domain_score_src_img_di, domain_label_src_img)
        loss_DA_img_dss = self.ce_loss(domain_score_src_img_ds, domain_label_src_img)

        loss_DA_img_dit = self.ce_loss(domain_score_tgt_img_di, domain_label_tgt_img)
        loss_DA_img_dst = self.ce_loss(domain_score_tgt_img_ds, domain_label_tgt_img)

        loss_DA_img_disent_di = (loss_DA_img_dis + loss_DA_img_dit) * 0.5
        loss_DA_img_disent_ds = (loss_DA_img_dss + loss_DA_img_dst) * 0.5 * self.beta

        domain_prob_src_img_di = F.softmax(domain_score_src_img_di, dim=1)
        domain_prob_src_img_ds = F.softmax(domain_score_src_img_ds, dim=1)
        domain_prob_tgt_img_di = F.softmax(domain_score_tgt_img_di, dim=1)
        domain_prob_tgt_img_ds = F.softmax(domain_score_tgt_img_ds, dim=1)

        margin_triplet = 0.5

        # print(domain_prob_tgt_img_ds.size())
        loss_triplet_s = F.triplet_margin_loss(domain_prob_src_img_di, domain_prob_tgt_img_di, domain_prob_src_img_ds,
                                               margin=margin_triplet)
        loss_triplet_t = F.triplet_margin_loss(domain_prob_tgt_img_di, domain_prob_src_img_di, domain_prob_tgt_img_ds,
                                               margin=margin_triplet)

        loss_triplet_img_disent = (loss_triplet_s + loss_triplet_t) * 0.5 * self.beta


        """  target rpn, cls, and box det learning   """

        # feed base feature map tp RPN to obtain rois
        self.RCNN_rpn.eval()
        tgt_rois, _, _ = \
            self.RCNN_rpn(tgt_base_feat, tgt_im_info, tgt_gt_boxes, tgt_num_boxes)

        if self.training:
            tgt_roi_data = self.RCNN_proposal_target(tgt_rois, tgt_gt_boxes, tgt_num_boxes)
            tgt_rois, tgt_rois_label, tgt_rois_target, tgt_rois_inside_ws, tgt_rois_outside_ws = tgt_roi_data


        tgt_rois = Variable(tgt_rois)

        if cfg.POOLING_MODE == 'align':
            tgt_pooled_feat_tensor = self.RCNN_roi_align(tgt_base_feat, tgt_rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            tgt_pooled_feat_tensor = self.RCNN_roi_pool(tgt_base_feat, tgt_rois.view(-1, 5))

        tgt_pooled_feat = self._head_to_tail(tgt_pooled_feat_tensor)



        """  instance-level disent cosine   """

        if ins_key:

            if cfg.POOLING_MODE == 'align':
                src_pooled_feat_ds_tensor = self.RCNN_roi_align(ds_feat_src, rois.view(-1, 5))
                tgt_pooled_feat_ds_tensor = self.RCNN_roi_align(ds_feat_tgt, tgt_rois.view(-1, 5))
            elif cfg.POOLING_MODE == 'pool':
                # print('roi pooling is on')
                src_pooled_feat_ds_tensor = self.RCNN_roi_pool(ds_feat_src, rois.view(-1, 5))
                tgt_pooled_feat_ds_tensor = self.RCNN_roi_pool(ds_feat_tgt, tgt_rois.view(-1, 5))

            src_pooled_feat_ds = self._head_to_tail(src_pooled_feat_ds_tensor)
            tgt_pooled_feat_ds = self._head_to_tail(tgt_pooled_feat_ds_tensor)


            loss_DA_ins_disent_triplet_src = self.cos_sim_loss(pooled_feat, src_pooled_feat_ds)

            loss_DA_ins_disent_triplet_tgt = self.cos_sim_loss(tgt_pooled_feat, tgt_pooled_feat_ds)



            loss_DA_ins_disent_triplet = (loss_DA_ins_disent_triplet_src + loss_DA_ins_disent_triplet_tgt) * 0.5 * self.beta


            loss_DA_ins_disent_spec = self.cos_sim_loss(src_pooled_feat_ds, tgt_pooled_feat_ds) * self.beta

        else:
            loss_DA_ins_disent_triplet = torch.zeros_like(loss_triplet_img_disent)

            loss_DA_ins_disent_spec = torch.zeros_like(loss_triplet_img_disent)



        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, \
               loss_DA_img_disent_di, loss_DA_img_disent_ds, loss_triplet_img_disent, loss_DA_ins_disent_triplet, \
               loss_DA_ins_disent_spec


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

    def cos_sim_loss(self,inp1,inp2,temp_fac = 1.0):
        # To compute the cosine similarity score of the input 2 vectors scaled by temparature factor
        # cos_sim_ = (self.cos_sim_official(inp1, inp2).pow(2)) / temp_fac


        cos_sim_ = (self.cos_sim_official(inp1, inp2)) / temp_fac

        loss_cs = torch.mean(cos_sim_)



        return loss_cs





