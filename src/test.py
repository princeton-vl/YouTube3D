import argparse
import os
import torch
import numpy as np
import cv2

import valid
from utils import save_obj, load_obj, makedir_if_not_exist
from RelativeLoss import RelativeLoss
from RelativeDepthDataset import RelativeDepthDataset, relative_depth_collate_fn
from DIWDataset import DIWDatasetVal
from YoutubeDataset import YoutubeDatasetVal
from ReDWebNet import ReDWebNet_resnet50
from HourglassNetwork import HourglassNetwork

from torch.utils import data




if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--test_file', '-t', default='NYU_test_50_50_1000.csv')		# or DIW_test.csv	
	parser.add_argument('--valid_file', '-v', default=None)
	parser.add_argument('--num_iters', '-iter', default=100000, type=int)
	parser.add_argument('--model_file', '-model', default=None)
	parser.add_argument('--output_file', '-o', default=None)
	parser.add_argument('--vis_depth', '-vis', action='store_true', default=False)


	args = parser.parse_args()
	training_args = load_obj(os.path.join(os.path.dirname(os.path.dirname(args.model_file)), 'args.pkl'))

	print "#######################################################################"
	print 'Testing args:', args	
	print "Training args:", training_args	
	print "#######################################################################\n\n\n"

	NetworkType = {'ReDWebNet':ReDWebNet_resnet50, 'NIPS':HourglassNetwork}
	
	b_resnet_prep = training_args['model_name'] == 'ReDWebNet'
	model = NetworkType[training_args['model_name']]().cuda()
	if training_args['n_GPUs'] > 1:
		model = torch.nn.parallel.DataParallel(model)
		model.load_state_dict(torch.load(args.model_file))	
		model = model.module
	else:		
		model.load_state_dict(torch.load(args.model_file))	

	criterion = RelativeLoss()

	in_thresh = 0.0
	if args.test_file.find('NYU') >= 0:
		DataSet = RelativeDepthDataset
		in_thresh = None
	elif args.test_file.find('DIW') >= 0:
		DataSet = DIWDatasetVal	
	else:
		DataSet = YoutubeDatasetVal

	model.eval()
	val_rel_error = {'thresh':0.0}
	if args.valid_file is not None:
		v_dataset = DataSet(csv_filename='../data/' + args.valid_file, b_resnet_prep = b_resnet_prep )
		v_data_loader = data.DataLoader(v_dataset, batch_size=1, num_workers=1, shuffle=True, collate_fn = relative_depth_collate_fn)

		print "Validating on %s" % args.valid_file
		val_rel_error = valid.valid(model, v_data_loader, criterion, max_iter = args.num_iters, verbal = True, b_vis_depth=args.vis_depth)



	test_dataset = DataSet(csv_filename='../data/' + args.test_file, b_resnet_prep = b_resnet_prep )
	test_data_loader = data.DataLoader(test_dataset, batch_size=1, num_workers=1, shuffle=False, collate_fn = relative_depth_collate_fn)
	print "Testing on %s" % args.test_file	
	test_rel_error = valid.valid(model, test_data_loader, criterion, max_iter = args.num_iters, in_thresh=in_thresh, b_vis_depth=args.vis_depth, verbal=True)
	# test_rel_error = valid.valid(model, test_data_loader, criterion, max_iter = args.num_iters, b_vis_depth=args.vis_depth, verbal=True)
	model.train()

	if args.output_file is not None:
		makedir_if_not_exist(os.path.dirname(args.output_file))
		save_obj({'val_rel_error':val_rel_error, 'test_rel_error':test_rel_error}, args.output_file)


