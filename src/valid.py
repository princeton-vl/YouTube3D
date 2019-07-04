import argparse
import os
import torch
import numpy as np
import cv2


from torch.utils import data
from torch.autograd import Variable



#from DummyDataset import DummyDataset
from RelativeLoss import RelativeLoss
from RelativeDepthDataset import RelativeDepthDataset, relative_depth_collate_fn


def vis_depth(depths, colors, i, target_res):
	out = depths[0,:]
	out = out - np.min(out[0, :, :])
	out = out / np.max(out[0, :, :]) * 255.0
	

	# img_height = 240
	# img_width = 320
	# f = open('./%d_depth.obj' % i, 'w')
	# for y in range(img_height):
	# 	for x in range(img_width):
	# 		f.write("v %f -%f -%f\n" % (x, y, out[0,y,x]+10.0))

	# for y in range(img_height-1):
	# 	for x in range(img_width-1):
	# 		this_index = y * img_width + x
	# 		f.write("f %d %d %d\n" % (this_index, this_index + img_width, this_index + 1))
	# 		f.write("f %d %d %d\n" % (this_index + img_width, this_index + img_width + 1, this_index + 1))
	




	out = out.transpose(1, 2, 0)  
	
	out_color = colors[0,:]
	out_color = out_color.transpose(1, 2, 0)
	out_color[:,:,0] = (out_color[:,:,0] * 0.229 + 0.485 ) *255.0 
	out_color[:,:,1] = (out_color[:,:,1] * 0.224 + 0.456 ) *255.0 
	out_color[:,:,2] = (out_color[:,:,2] * 0.225 + 0.406 ) *255.0 
	img = np.zeros((out_color.shape[0],out_color.shape[1]*2,3), np.uint8)
	img[:,:out_color.shape[1], :] = out_color
	img[:,out_color.shape[1] : out_color.shape[1]*2, :] = out

	img = cv2.resize(img, (2*target_res[1], target_res[0]))
	cv2.imwrite("./visualize/%d_depth.jpg" % i, img)


#### for debug
def classify(z_A, z_B, ground_truth, thresh):
	n_point = z_A.shape[0]

	eq_correct_count = 0.0
	not_eq_correct_count = 0.0
	eq_count = 0.0
	not_eq_count = 0.0

	for i in range(n_point):
		z_A_z_A = z_A[i] - z_B[i]

		_classify_res = 1.0
		if z_A_z_A > thresh:
			_classify_res = 1
		elif z_A_z_A < -thresh:
			_classify_res = -1
		elif z_A_z_A <= thresh and z_A_z_A >= -thresh:
			_classify_res = 0

		if _classify_res == ground_truth[i]:
			if ground_truth[i] == 0:
				eq_correct_count += 1
			else:
				not_eq_correct_count += 1
		
		if ground_truth[i] == 0:
			eq_count += 1
		else:
			not_eq_count += 1

	print 'classify'
	print 'thresh', thresh
	print 'eq_correct_count', eq_correct_count
	print 'not_eq_correct_count', not_eq_correct_count
	print 'eq_count', eq_count
	print 'not_eq_count', not_eq_count


def evalute_correct_rate_one_img(depth, target, threshes, target_res = None):
	assert len(target) == 1
	# Target is a list of cpu tensor. There is only one element in the list. This element is a 5 x n Tensor.
	# Depth is a cpu tensor. With 4 dimensions.

	depth = depth[0,0,...]
	target = target[0]

	if target_res is not None:
		#print "Resizing network output from (%d, %d) to (%d, %d)" % (depth.shape[0], depth.shape[1], target_res[0], target_res[1])
		depth = resize_tensor(depth, target_res)

	y_A = target[0,:]
	x_A = target[1,:]
	y_B = target[2,:]
	x_B = target[3,:]
	gt_r = target[4,:].float()




	z_A_arr = depth.index_select(1, x_A).gather(0, y_A.view(1,-1))
	z_B_arr = depth.index_select(1, x_B).gather(0, y_B.view(1,-1))
	z_A_z_B = z_A_arr - z_B_arr

	n_pair = target.shape[1]
	eq_count = torch.sum(torch.eq(gt_r, 0.0))
	not_eq_count = torch.sum(torch.eq(gt_r, 1.0)) + torch.sum(torch.eq(gt_r, -1.0))
	assert eq_count + not_eq_count == n_pair

	
	report = {'eq_count':eq_count, 'not_eq_count':not_eq_count} 
	for thresh in threshes:
		_gt_res = torch.gt(z_A_z_B, thresh)
		_lt_res = torch.lt(z_A_z_B, -thresh)

		est_r = _gt_res.float() + _lt_res.float() * -1
		not_eq_correct_count = torch.sum(torch.eq(est_r * gt_r, 1.0))

		eq_mask = 1.0 - torch.abs(gt_r)
		eq_correct_count = torch.sum(eq_mask * (1.0 - torch.abs(est_r)))
	
		report[thresh] = {}
		report[thresh]['not_eq_correct_count'] = not_eq_correct_count
		report[thresh]['eq_correct_count'] = eq_correct_count


		# #### debug
		# print 'eq_correct_count', eq_correct_count
		# print 'not_eq_correct_count', not_eq_correct_count
		# print 'eq_count', eq_count
		# print 'not_eq_count', not_eq_count
		# print "----------------"
		# classify(z_A_arr[0], z_B_arr[0], gt_r, thresh)
		# print '===================================='
		# raw_input()
		

		# eq_res = torch.gt(torch.lt(z_A_z_B, thresh), -thresh)

	return report


def resize_tensor(depth_tensor, target_res):
	# target_res: a tuple, (height, width)
	if depth_tensor.shape[0] == target_res[0] and depth_tensor.shape[1] == target_res[1]:
		return depth_tensor
	else:
		depth = depth_tensor.numpy()
		depth = cv2.resize(depth, (target_res[1], target_res[0]))	#cv2.resize(src, (width, height))
		depth = torch.from_numpy(depth)
		return depth
	
def valid(model, data_loader, criterion, max_iter=1400, verbal=False, b_vis_depth=False, in_thresh = None):
	print "Evaluating..."

	threshes = []
	if in_thresh is None:
		for i in range(140):
			threshes.append(0.1 + 0.01 * i)
	else:
		threshes = [in_thresh]

	assert not model.training

	iter = 0 
	reports = []
	for step, (inputs, target, target_res) in enumerate(data_loader):
		iter += 1
		print iter
		if iter > max_iter:
			break
		input_var = Variable(inputs.cuda())
		output_var = model(input_var)
		if b_vis_depth:
			vis_depth(output_var.data.cpu().numpy(), inputs.numpy(), step, target_res[0])

		report = evalute_correct_rate_one_img(output_var.data.cpu(), target, threshes, target_res[0])
		reports.append(report)

		# this None assignment is necessary to keep the gpu memory clean
		input_var = None
		output_var = None
		inputs = None
		target = None
		target_res = None


	print "%d samples are evaluated.\n" % (iter)
	print "Thresh\tWKDR\tWKDR_neq\tWKDR_eq"	

	highest_correct_rate = 0	
	relative_error = {"thresh":-1.0, "WKDR":100, "WKDR_neq": 100, "WKDR_eq":100}
	for thresh in threshes:
		eq_correct_count = 0.0
		not_eq_correct_count = 0.0
		eq_count = 0.0
		not_eq_count = 0.0

		for report in reports:			
			eq_correct_count += report[thresh]['eq_correct_count']
			not_eq_correct_count += report[thresh]['not_eq_correct_count']
			eq_count += report['eq_count']
			not_eq_count += report['not_eq_count']
			
		correct_rate_neq = not_eq_correct_count / not_eq_count
		if eq_count == 0:
			correct_rate_eq = 0.0
		else:
			correct_rate_eq = eq_correct_count / eq_count
		correct_rate = (not_eq_correct_count + eq_correct_count) / (not_eq_count + eq_count)
		

		if len(threshes) == 1:
			relative_error = {"thresh":thresh, "WKDR":(1.0-correct_rate) * 100, "WKDR_neq": (1.0-correct_rate_neq) * 100, "WKDR_eq":(1.0-correct_rate_eq) * 100}

		if min(correct_rate_neq, correct_rate_eq) > highest_correct_rate:
			highest_correct_rate = min(correct_rate_neq, correct_rate_eq)
			relative_error = {"thresh":thresh, "WKDR":(1.0-correct_rate) * 100, "WKDR_neq": (1.0-correct_rate_neq) * 100, "WKDR_eq":(1.0-correct_rate_eq) * 100}
		
		if verbal:
			print "%.2f\t%.2f%%\t%.2f%%\t%.2f%%" % (thresh, (1.0-correct_rate) * 100, (1.0-correct_rate_neq) * 100, (1.0-correct_rate_eq) * 100)

	
	print "Best:"
	print "%.2f\t%.2f%%\t%.2f%%\t%.2f%%\n" % (relative_error['thresh'], relative_error['WKDR'], relative_error['WKDR_neq'], relative_error['WKDR_eq'])

	return relative_error





