import _init_paths
import tensorflow as tf
from fast_rcnn.config import cfg, get_output_dir
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os, sys, cv2
import argparse
from networks.factory import get_network
from datasets.factory import get_imdb
import cPickle
import xml.etree.ElementTree as ET
import ipdb


CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'pottedplant', 'sheep', 
           'sofa', 'train', 'tvmonitor','face')


def vis_detections(im, class_name, dets, bbox, gt_box, thresh=0.9):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(im, aspect='equal')
        
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                            bbox[2] - bbox[0],
                            bbox[3] - bbox[1], fill=False,
                            edgecolor='red', linewidth=3.5))
        # ax.text(bbox[0], bbox[1] - 2,
        #         '{:s} {:.3f}'.format(class_name, score),
        #         bbox=dict(facecolor='blue', alpha=0.5),
        #         fontsize=14, color='white')

    num_gt = gt_box.shape[0]
    for i in range(num_gt):
        x_cen = (gt_box[i, 0] + gt_box[i, 1]) / 2.0
        y_cen = (gt_box[i, 2] + gt_box[i, 3]) / 2.0
        ax.add_patch(
            plt.Rectangle((gt_box[i, 0], gt_box[i, 2]),
                            gt_box[i, 1] - gt_box[i, 0],
                            gt_box[i, 3] - gt_box[i, 2], fill=False,
                            edgecolor='green', linewidth=3.5))

    ax.set_title(('{} detections with p({} | box) >= {:.1f}').format(class_name, class_name, thresh), fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    # plt.draw()
    # plt.pause(.1)
    plt.show()
    plt.close()
        

def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    #im_file = os.path.join('/home/corgi/Lab/label/pos_frame/ACCV/training/000001/',image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')

    CONF_THRESH = 0.9
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, ax, thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        default='VGGnet_test')
    parser.add_argument('--model', dest='model', help='Model path',
                        default=' ')
    parser.add_argument('--imdb', dest='imdb', default='voc_2007_test')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    if args.model == ' ':
        raise IOError(('Error: Model not found.\n'))
        
    # init session
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # load network
    net = get_network(args.demo_net)
    # load model
    saver = tf.train.Saver()
    saver.restore(sess, args.model)
    #sess.run(tf.initialize_all_variables())

    print '\n\nLoaded network {:s}'.format(args.model)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(sess, net, im)

    # Specify imdb
    imdb = get_imdb(args.imdb)
    num_images = len(imdb.image_index)

    weights_filename = 'VGGnet_fast_rcnn_iter_70000'
    output_dir = get_output_dir(imdb, weights_filename)
    
    all_boxes = [[[] for _ in xrange(num_images)]
                  for _ in xrange(imdb.num_classes)]
       
    # Visualize stroke and facial areas classification result
    for i in xrange(num_images):
        im_name = imdb.image_path_at(i)
        xml_name = '/Users/yuliangzou/Documents/EECS442/Project/Faster-RCNN_TF/data/VOCdevkit2007/VOC2007/Annotations' + im_name[96:-3] + 'xml'
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for {}'.format(im_name)
        im = cv2.imread(im_name)
        scores, boxes = im_detect(sess, net, im)
        # scores, boxes, strokes, areas = im_detect_ori(sess, net, im, None)

        # Get ground truth labels
        tree = ET.parse(xml_name)
        objs = tree.findall('object')
        # Get ground truth bounding box
        num_objs = len(objs)
        gt_box = np.zeros((num_objs, 4), dtype=np.uint16)
        for ix, obj in enumerate(objs):
            b = obj.find('bndbox')
            gt_box[ix, 0] = int(b.find('xmin').text) - 1
            gt_box[ix, 1] = int(b.find('xmax').text) - 1
            gt_box[ix, 2] = int(b.find('ymin').text) - 1
            gt_box[ix, 3] = int(b.find('ymax').text) - 1

        # skip j = 0, because it's the background class
        CONF_THRESH = 0.9
        NMS_THRESH = 0.3
        for cls_ind, cls in enumerate(CLASSES[20:]):
            cls_ind += 20 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind+1)]
            cls_scores = scores[:, cls_ind]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = nms(cls_dets, NMS_THRESH)
            cls_dets = cls_dets[keep, :]
            # Choose the largest prob. one
            # inds = [np.argmax(cls_dets[:, -1])]
            vis_detections(im, CLASSES[cls_ind], cls_dets, cls_boxes, gt_box, thresh=0.9)

    ipdb.set_trace()

    # (Yuliang) Test classification mAP
    """
    for i in xrange(num_images):
        im_name = imdb.image_path_at(i)
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for {}'.format(im_name)
        im = cv2.imread(im_name)
        # demo(sess, net, im_name)
        scores, boxes = im_detect(sess, net, im, None)

        # skip j = 0, because it's the background class
        thresh = 0.05
        max_per_image = 300
        for j in xrange(1, imdb.num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j*4:(j+1)*4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = nms(cls_dets, cfg.TEST.NMS)
            cls_dets = cls_dets[keep, :]
            all_boxes[j][i] = cls_dets

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1] for j in xrange(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)
    print 'Evaluationg detections'
    imdb.evaluate_detections(all_boxes, output_dir)
    ipdb.set_trace()
    """
