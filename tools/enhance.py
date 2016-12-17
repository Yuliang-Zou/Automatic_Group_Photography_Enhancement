# Main function to do group photo enhancement
import _init_paths
import tensorflow as tf
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect, im_detect_ori
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import os, sys, cv2
import argparse
from networks.factory import get_network
import ipdb

from facegroup import obtainSimilarityScore
from faceswap import *

# (Yuliang) Background + voc(w/o person) + face
CLASSES = ('__background__', 
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'pottedplant', 'sheep',
           'sofa', 'train', 'tvmonitor', 'face')

def face_detect(sess, net, image_name):
	"""Give bounding boxes and quality scores of one given image"""

	# Load the demo image
	im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
	im = cv2.imread(im_file)

	# Detect all object classes and regress object bounds
	timer = Timer()
	timer.tic()
	# scores, boxes = im_detect(sess, net, im)
	scores, boxes, eyes, smiles = im_detect_ori(sess, net, im)
	timer.toc()
	print ('Detection took {:.3f}s for '
			'{:d} object proposals').format(timer.total_time, boxes.shape[0])

	# Visualize detections for each class
	# im = im[:, :, (2, 1, 0)]
	# fig, ax = plt.subplots(figsize=(8, 8))
	# ax.imshow(im, aspect='equal')

	CONF_THRESH = 0.9
	NMS_THRESH = 0.3
	for cls_ind, cls in enumerate(CLASSES[20:]):
		cls_ind += 20 # because we skipped everything except face
		cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
		cls_scores = scores[:, cls_ind]
		dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
		keep = nms(dets, NMS_THRESH)
		dets = dets[keep, :]
		eye  = eyes[keep, :]
		smile= smiles[keep, :]

	inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
	face_num = len(inds)
	print '{} faces detected!'.format(face_num)
	dets = dets[inds, :]
	eye = eye[inds, 1]
	smile = smile[inds, 1]

	return dets, eye, smile

    
def parse_args():
	"""Parse input arguments."""
	parser = argparse.ArgumentParser(description='Faster R-CNN demo')
	parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]', default=0, type=int)
	parser.add_argument('--cpu', dest='cpu_mode', help='Use CPU mode (overrides --gpu)', action='store_true')
	parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]', default='VGGnet_test')
	parser.add_argument('--model', dest='model', help='Model path', default='model/VGGnet_fast_rcnn_full_eye_smile_1e-4_iter_70000.ckpt')

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

    
	im_names = ['f2_1.png', 'f2_2.png', 'f2_3.png']
	# im_names = ['11.jpg', '22.jpg', '33.jpg']
	im_num = len(im_names)
	im_info = {}
	best_score = 0
	best_im = ''
	face_num = 0

	for im_name in im_names:
		print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
		print 'Detection for data/demo/{}'.format(im_name)
		dets, eyes, smiles = face_detect(sess, net, im_name)
		im_info[im_name] = {}
		im_info[im_name]['dets'] = dets
		im_info[im_name]['eyes'] = eyes
		im_info[im_name]['smiles'] = smiles

		overall_score = eyes.sum() + smiles.sum()
		if overall_score > best_score:
			best_score = overall_score
			best_im = im_name
			face_num = len(eyes)

	print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'     	
	print 'The candidate image is: {}'.format(best_im)
	print 'The face number of the group is: {}'.format(face_num)

	# Use faces from candidate as default best faces
	best_faces = [best_im] * face_num
	best_scores = []
	best_face_box = []
	for i in range(face_num):
		best_scores.append(im_info[best_im]['eyes'][i] + im_info[best_im]['smiles'][i])
		best_face_box.append(im_info[best_im]['dets'][i])

	cand_img = cv2.imread('./data/demo/' + best_im)
	# for i in range(face_num):
	# 	xmin, ymin, xmax, ymax, _ = im_info[best_im]['dets'][i]
	# 	face = cand_img[ymin:ymax, xmin:xmax, :]
	# 	face = face[:, :, (2, 1, 0)]    # BGR -> RGB
	# 	plt.imshow(face)
	# 	plt.show()

	# Grouping and find best faces
	for im_name in im_names:
		if im_name == best_im:
			continue

		temp_num = len(im_info[im_name]['eyes'])
		for i in range(temp_num):
			temp_score = im_info[im_name]['eyes'][i] + im_info[im_name]['smiles'][i]
			# If the score is too small, just ignore it
			if temp_score < min(best_scores):
				continue

			img = cv2.imread('./data/demo/' + im_name)
			xmin, ymin, xmax, ymax, _ = im_info[im_name]['dets'][i]
			face_data = img[ymin:ymax, xmin:xmax, :]
			largest_sim = 60
			match_id = -1    # assume no matching
			# Compare with each face in candidate image
			for j in range(face_num):
				xmin, ymin, xmax, ymax, _ = im_info[best_im]['dets'][j]

				temp_face = cand_img[ymin:ymax, xmin:xmax, :]
				# plt.imshow(face_data)
				# plt.show()
				# plt.imshow(temp_face)
				# plt.show()
				sim = obtainSimilarityScore(face_data, temp_face)
				# if j == 3:
				# 	print sim
				# 	plt.imshow(face_data)
				# 	plt.show()

				if sim > largest_sim:
					largest_sim = sim
					match_id = j
			# print largest_sim
    		
			# No matching
			if match_id == -1:
				continue

			if temp_score > best_scores[match_id]:
				best_faces[match_id] = im_name
				best_scores[match_id] = temp_score
				best_face_box[match_id] = im_info[im_name]['dets'][j]

	ipdb.set_trace()
	for i, im_name in enumerate(best_faces):
		img = cv2.imread('./data/demo/' + im_name)
		xmin, ymin, xmax, ymax, _ = best_face_box[i]
		img = img[ymin:ymax, xmin:xmax, :]
		im = im[:, :, (2, 1, 0)]    # BGR -> RGB
		# plt.imshow(img)
		# plt.show()

	# print best_faces
	# print best_scores
	# print best_face_box


	# Face swapping
	cand_img = cv2.imread('./data/demo/' + best_im)
	for i in range(face_num):
		# Don't need to change
		if best_faces[i] == best_im:
			continue

		# target - good, source - bad
		target_img = cv2.imread('./data/demo/' + best_faces[i])
		xmin, ymin, xmax, ymax, _ = best_face_box[i]
		target_face = target_img[ymin:ymax, xmin:xmax, :]

		xmin, ymin, xmax, ymax, _  = im_info[best_im]['dets'][i]
		source_face = cand_img[ymin:ymax, xmin:xmax, :]

		PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
		predictor = dlib.shape_predictor(PREDICTOR_PATH)
		source_landmark = get_landmarks(source_face, predictor)
		target_landmark = get_landmarks(target_face, predictor)

		M = transformation_from_points(source_landmark[ALIGN_POINTS], target_landmark[ALIGN_POINTS])

		mask = get_face_mask(target_face, target_landmark)
		warped_mask = warp_im(mask, M, source_face.shape)
		combined_mask = numpy.max([get_face_mask(source_face, source_landmark), warped_mask], axis=0)

		warped_im2 = warp_im(target_face, M, source_face.shape)
		warped_corrected_im2 = correct_colours(source_face, warped_im2, source_landmark)

		output_im = source_face * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask

		cand_img[ymin:ymax, xmin:xmax, :] = output_im

	cv2.imwrite('output.jpg', cand_img)

		

    




