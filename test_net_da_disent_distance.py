# --------------------------------------------------------
# Pytorch Multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time

import cv2

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
from lib.roi_data_layer.roidb import combined_roidb
from lib.roi_data_layer.roibatchLoader import roibatchLoader
from lib.model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from lib.model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from lib.model.roi_layers import nms
from lib.model.rpn.bbox_transform import bbox_transform_inv
from lib.model.utils.net_utils import save_net, load_net, vis_detections
from lib.model.da_faster_rcnn_disent_new.vgg16_inference_distance import vgg16
from lib.model.da_faster_rcnn_disent_new.resnet_inference import resnet
import torch.nn.functional as F
from evaluate_domain_distance import domain_distance, domain_distance_ins



try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3



def mkdir(out_folder):
    try:
        os.stat(os.path.dirname(out_folder + '/'))
    except:
        os.mkdir(os.path.dirname(out_folder + '/'))





def show_heat_on_image(im, heatmap):
    # heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    # img =
    im = np.float32(im) / 255
    width, height = heatmap.shape[:2]
    # print('img shape, ', im.shape)
    # print('heat shape, ', heatmap.shape)
    img = cv2.resize(im, (height, width))

    # print('img rs shape, ', img.shape)

    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default='cfgs/vgg16.yml', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='res101', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  # parser.add_argument('--load_dir', dest='load_dir',
  #                     help='directory to load models', default="models",
  #                     type=str)
  parser.add_argument('--model_dir', dest='model_dir',
                      help='directory to load models', default="models.pth",
                      type=str)
  parser.add_argument('--save_name', dest='save_name',
                      help='the folder name to save the predictions', default="faster_rcnn_og",
                      type=str)
  parser.add_argument('--part', dest='part',
                      help='test_s or test_t or test_all', default="test_t",
                      type=str)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')
  parser.add_argument('--parallel_type', dest='parallel_type',
                      help='which part of model to parallel, 0: all, 1: model before roi pooling',
                      default=0, type=int)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=10021, type=int)
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      action='store_true')
  parser.add_argument('--vis_dir', dest='vis_dir',
                      help='directory to save visualizations', default="vis-file",
                      type=str)
  args = parser.parse_args()
  return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)

  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  np.random.seed(cfg.RNG_SEED)
  if args.dataset == "pascal_voc":
      args.imdb_name = "voc_2007_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']

  elif args.dataset == "pascal2clipart":
      args.s_imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.t_imdb_name = "clipart_train"
      args.s_imdbtest_name = "voc_2007_test"
      args.t_imdbtest_name = "clipart_test"


      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

      # args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

      args.all_imdbtest_name = "clipart_test"

  elif args.dataset == "sim10k2cityscape":
      args.s_imdb_name = "sim10k_train"
      args.t_imdb_name = "cityscape_car_train"


      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
      args.all_imdbtest_name = "cityscape_car_val"

      args.s_imdbtest_name = "sim10k_test_vis"
      # args.t_imdbtest_name = "cityscape_car_val"
      args.t_imdbtest_name = "cityscape_car_test_vis"

  elif args.dataset == "kitti2cityscape":
      args.s_imdb_name = "kitti_car_trainval"
      args.t_imdb_name = "cityscape_car_train"

      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
      args.all_imdbtest_name = "cityscape_car_val"

      args.s_imdbtest_name = "kitti_car_test_vis"
      args.t_imdbtest_name = "cityscape_car_val"

  elif args.dataset == "cityscape2kitti":
      args.s_imdb_name = "cityscape_car_train"
      args.t_imdb_name = "kitti_car_trainval"

      args.s_imdbtest_name = "kitti_car_trainval"
      args.t_imdbtest_name = "kitti_car_trainval"

      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
      args.all_imdbtest_name = "kitti_car_trainval"

  elif args.dataset == "cityscape":
      print('loading our dataset...........')
      args.s_imdb_name = "cityscape_2007_train_s"
      args.t_imdb_name = "cityscape_2007_train_t"
      #
      args.s_imdbtest_name="cityscape_2007_test_s"
      args.t_imdbtest_name = "cityscape_2007_test_t"


      args.all_imdbtest_name = "cityscape_2007_test_all"
      args.set_cfgs = ['ANCHOR_SCALES', '[4,8,16,32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "pascal_voc_0712":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "coco":
      args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
      args.imdbval_name = "coco_2014_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "imagenet":
      args.imdb_name = "imagenet_train"
      args.imdbval_name = "imagenet_val"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "vg":
      args.imdb_name = "vg_150-50-50_minitrain"
      args.imdbval_name = "vg_150-50-50_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']

  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)

  cfg.TRAIN.USE_FLIPPED = False

  imdb, roidb, ratio_list, ratio_index = combined_roidb(args.s_imdbtest_name, False)

  imdb_t, roidb_t, ratio_list_t, ratio_index_t = combined_roidb(args.t_imdbtest_name, False)




  imdb.competition_mode(on=True)

  imdb_t.competition_mode(on=True)

  print('{:d} roidb entries'.format(len(roidb_t)))


  load_name = args.model_dir

  # initilize the network here.
  if args.net == 'vgg16':
    fasterRCNN = vgg16(imdb.classes, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res101':
    fasterRCNN = resnet(imdb.classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
    fasterRCNN = resnet(imdb.classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res152':
    fasterRCNN = resnet(imdb.classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()

  fasterRCNN.create_architecture()

  print("load checkpoint %s" % (load_name))
  checkpoint = torch.load(load_name)
  # fasterRCNN.load_state_dict(checkpoint['model'])
  fasterRCNN.load_state_dict({k: v for k, v in checkpoint['model'].items() if k in fasterRCNN.state_dict()})
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']


  print('load model successfully!')
  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  im_data_tgt = torch.FloatTensor(1)
  im_info_tgt = torch.FloatTensor(1)
  num_boxes_tgt = torch.LongTensor(1)
  gt_boxes_tgt = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

    im_data_tgt = im_data_tgt.cuda()
    im_info_tgt = im_info_tgt.cuda()
    num_boxes_tgt = num_boxes_tgt.cuda()
    gt_boxes_tgt = gt_boxes_tgt.cuda()

  # make variable
  im_data = Variable(im_data)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)

  im_data_tgt = Variable(im_data_tgt)
  im_info_tgt = Variable(im_info_tgt)
  num_boxes_tgt = Variable(num_boxes_tgt)
  gt_boxes_tgt = Variable(gt_boxes_tgt)

  if args.cuda:
    cfg.CUDA = True

  if args.cuda:
    fasterRCNN.cuda()

  start = time.time()
  max_per_image = 100

  vis = args.vis
  vis_dir = args.vis_dir
  mkdir(vis_dir)

  vis = False


  thresh = 0.0



  num_images = len(imdb_t.image_index)

  all_boxes = [[[] for _ in xrange(num_images)]
               for _ in xrange(imdb_t.num_classes)]

  save_name = args.save_name

  output_dir = get_output_dir(imdb, save_name)

  dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, \
                        imdb.num_classes, training=False, normalize = False)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=0,
                            pin_memory=True)



  dataset_t = roibatchLoader(roidb_t, ratio_list_t, ratio_index_t, 1, \
                             imdb_t.num_classes, training=False, normalize = False)
  dataloader_t = torch.utils.data.DataLoader(dataset_t, batch_size=1,
                                             shuffle=False, num_workers=0,
                                             pin_memory=True)

  data_iter = iter(dataloader)
  data_iter_tgt = iter(dataloader_t)

  _t = {'im_detect': time.time(), 'misc': time.time()}
  det_file = os.path.join(output_dir, 'detections.pkl')

  fasterRCNN.eval()
  empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))

  src_global_feat_list = []
  tgt_global_feat_list = []

  src_ins_feat_list = []
  tgt_ins_feat_list = []

  for i in range(imdb.num_classes-1):
      src_ins_feat_list.append([])
      tgt_ins_feat_list.append([])



  for i in range(num_images):


      try:
          data = next(data_iter)
      except:
          data_iter = iter(dataloader)
          data = next(data_iter)
      try:
          data_tgt = next(data_iter_tgt)
      except:
          data_iter_tgt = iter(dataloader_t)
          data_tgt = next(data_iter_tgt)

      with torch.no_grad():
              im_data.resize_(data[0].size()).copy_(data[0])
              im_info.resize_(data[1].size()).copy_(data[1])
              gt_boxes.resize_(data[2].size()).copy_(data[2])
              num_boxes.resize_(data[3].size()).copy_(data[3])

              im_data_tgt.resize_(data_tgt[0].size()).copy_(data_tgt[0])
              im_info_tgt.resize_(data_tgt[1].size()).copy_(data_tgt[1])
              gt_boxes_tgt.resize_(data_tgt[2].size()).copy_(data_tgt[2])
              num_boxes_tgt.resize_(data_tgt[3].size()).copy_(data_tgt[3])

      det_tic = time.time()
      rois, cls_prob, bbox_pred, \
      rpn_loss_cls, rpn_loss_box, \
      RCNN_loss_cls, RCNN_loss_bbox, rois_label, \
      src_global_feat, tgt_global_feat, \
      src_ins_feat_list, tgt_ins_feat_list \
          = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, im_data_tgt, im_info_tgt, gt_boxes_tgt, num_boxes_tgt, src_ins_feat_list, tgt_ins_feat_list)

      src_global_feat_list.append(src_global_feat)
      tgt_global_feat_list.append(tgt_global_feat)


      scores = cls_prob.data
      boxes = rois.data[:, :, 1:5]

      # print('scores, ', scores)
      # print('boxes, ', boxes)

      if cfg.TEST.BBOX_REG:
          # Apply bounding-box regression deltas
          box_deltas = bbox_pred.data
          if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
          # Optionally normalize targets by a precomputed mean and stdev
            if args.class_agnostic:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                box_deltas = box_deltas.view(1, -1, 4)
            else:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                box_deltas = box_deltas.view(1, -1, 4 * len(imdb_t.classes))

          pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
          pred_boxes = clip_boxes(pred_boxes, im_info_tgt.data, 1)

          # print('box aug ')
      else:
          # Simply repeat the boxes, once for each class
          pred_boxes = np.tile(boxes, (1, scores.shape[1]))


      pred_boxes /= data_tgt[1][0][2].item()

      # print('pred boxes, ', pred_boxes)

      scores = scores.squeeze()
      pred_boxes = pred_boxes.squeeze()
      det_toc = time.time()
      detect_time = det_toc - det_tic
      misc_tic = time.time()
      if vis:
          im = cv2.imread(imdb_t.image_path_at(i))
          im2show = np.copy(im)
      for j in xrange(1, imdb.num_classes):
          inds = torch.nonzero(scores[:,j]>thresh).view(-1)
          # if there is det
          if inds.numel() > 0:
            cls_scores = scores[:,j][inds]
            _, order = torch.sort(cls_scores, 0, True)
            if args.class_agnostic:
              cls_boxes = pred_boxes[inds, :]
            else:
              cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
            
            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
            cls_dets = cls_dets[order]
            keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
            cls_dets = cls_dets[keep.view(-1).long()]
            if vis:
              im2show = vis_detections(im2show, imdb.classes[j], j, cls_dets.cpu().numpy(), 0.3)
            all_boxes[j][i] = cls_dets.cpu().numpy()
          else:
            all_boxes[j][i] = empty_array



      # Limit to max_per_image detections *over all classes*
      if max_per_image > 0:
          image_scores = np.hstack([all_boxes[j][i][:, -1]
                                    for j in xrange(1, imdb.num_classes)])
          if len(image_scores) > max_per_image:
              image_thresh = np.sort(image_scores)[-max_per_image]
              for j in xrange(1, imdb.num_classes):
                  keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                  all_boxes[j][i] = all_boxes[j][i][keep, :]

      misc_toc = time.time()
      nms_time = misc_toc - misc_tic

      sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
          .format(i + 1, num_images, detect_time, nms_time))
      sys.stdout.flush()

      if vis:
          img_path = imdb_t.image_path_at(i)
          img_name = os.path.basename(img_path)

          out_path = os.path.join(vis_dir,img_name)
          cv2.imwrite(out_path, im2show)


  src_global_feat_array = np.asarray(src_global_feat_list)
  tgt_global_feat_array = np.asarray(tgt_global_feat_list)

  wass_dist, proxy_dist = domain_distance(src_global_feat_array, tgt_global_feat_array)


  wass_dist_ins_list, proxy_dist_ins_list = domain_distance_ins(src_ins_feat_list, tgt_ins_feat_list, samples=100)

  wass_dist_ins = np.mean(wass_dist_ins_list)
  proxy_dist_ins = np.mean(proxy_dist_ins_list)


  print('earth move distance, ', wass_dist)
  print('proxy a distance, ', proxy_dist)

  print('ins earth move distance, ', wass_dist_ins, 'emd list is, ', wass_dist_ins_list)
  print('ins proxy a distance, ', proxy_dist_ins,'pad list is, ', proxy_dist_ins_list)

  with open(det_file, 'wb') as f:
      pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

  print('Evaluating detections')
  # imdb_t.evaluate_detections(all_boxes, output_dir)
  imdb_t.evaluate_detections(all_boxes, output_dir)

  end = time.time()
  print("test time: %0.4fs" % (end - start))


