import argparse
import os
import cv2

import torch
import torch.nn.parallel
import numpy as np

import valid

import config
import TBLogger

from utils import makedir_if_not_exist, StoreDictKeyPair
from torch import optim
from torch.utils import data
from torch.autograd import Variable

from ReDWebNet import ReDWebNet_resnet50
#from DummyDataset import DummyDataset
from RelativeLoss import RelativeLoss
from RelativeDepthDataset import RelativeDepthDataset, relative_depth_collate_fn, save_obj
from YoutubeDataset import YoutubeDataset, YoutubeDatasetVal
from DIWDataset import DIWDataset, DIWDatasetVal
from ReDWebDataset import ReDWebDataset, ReDWebDatasetVal
from YTmixReDWebDataset import YTmixReDWebDataset, YTmixReDWebDatasetVal

def save_model(optimizer, model, iter, prev_iter, prefix=''):
	makedir_if_not_exist(config.JOBS_MODEL_DIR)
	torch.save(model.state_dict(), os.path.join(config.JOBS_MODEL_DIR, '%smodel_iter_%d.bin' % (prefix, iter + prev_iter) ))
	torch.save(optimizer.state_dict(), os.path.join(config.JOBS_MODEL_DIR, '%sopt_state_iter_%d.bin' % (prefix, iter + prev_iter) ))

def get_prev_iter(pretrained_file):	
	temp = pretrained_file.replace('.bin', '')
	prev_iter = int(temp.split('_')[-1])
	 
	return prev_iter


def train(dataset_name, model_name, loss_name,\
		  n_GPUs, b_oppi, b_data_aug, b_sort, b_diff_lr,\
		  train_file, valid_file,\
		  learning_rate, num_iters, num_epoches,\
		  batch_size, num_loader_workers, pretrained_file,\
		  model_save_interval, model_eval_interval):

	NetworkType = {'ReDWebNet':ReDWebNet_resnet50}
	LossType = {"RelativeLoss":RelativeLoss}
	
	# create (and load) model. Should wrap with torch.nn.parallel.DistributedDataParallel before loading pretraiend model (https://github.com/pytorch/examples/blob/master/imagenet/main.py)
	model = NetworkType[model_name]().cuda()
	b_resnet_prep = model_name == 'ReDWebNet'
	if n_GPUs > 1:
		print "######################################################"
		print "Using %d GPUs, batch_size is %d" % (n_GPUs, batch_size)
		print "######################################################"
		model = torch.nn.parallel.DataParallel(model)

	print 'num_loader_workers:', num_loader_workers

	# resume from a checkpoint model
	prev_iter = 0
	if pretrained_file:
		model.load_state_dict(torch.load(os.path.join( config.JOBS_MODEL_DIR, pretrained_file )))
		prev_iter = get_prev_iter(pretrained_file)
	print "Prev_iter: {}".format(prev_iter)

	# set up criterion and optimizer
	if loss_name == 'L2_loss':
		t_collate_fn = metric_depth_collate_fn
		criterion = torch.nn.MSELoss()
	else:
		t_collate_fn = relative_depth_collate_fn	
		criterion = LossType[loss_name](b_sort = b_sort)

	if b_diff_lr and b_resnet_prep:
		print("==========================================================================")
		print("  Use different learning rates for different part of the model")
		print("    The learning rate for the ResNet encoder is 10x smaller than decoder.")
		print("==========================================================================")
		optimizer = optim.RMSprop([{'params': model.resnet_model.parameters(), 'lr':learning_rate / 10.0},
								   {'params': model.feafu3.parameters() },
								   {'params': model.feafu2.parameters() },
								   {'params': model.feafu1.parameters() },
								   {'params': model.ada_out.parameters()},
									], lr = learning_rate)	
	else:
		optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)	

	try:
		if pretrained_file:
			print pretrained_file
			optimizer.load_state_dict(torch.load(os.path.join( config.JOBS_MODEL_DIR, pretrained_file.replace('model_', 'opt_state_') )))
	except:
		print("Exception happens when trying to load optimizer state, possibility due to different learning rate strategy.")


	# register dataset type 
	DatasetsType = {"YoutubeDataset": {'train_dataset':YoutubeDataset, 'val_dataset':YoutubeDatasetVal, 't_val_dataset':YoutubeDatasetVal},
					"RelativeDepthDataset":{'train_dataset':RelativeDepthDataset, 'val_dataset':RelativeDepthDataset, 't_val_dataset':RelativeDepthDataset},
					"DIWDataset":{'train_dataset':DIWDataset, 'val_dataset':DIWDatasetVal, 't_val_dataset':DIWDatasetVal},
					"YT_DIW":{'train_dataset':YoutubeDataset, 'val_dataset':DIWDatasetVal, 't_val_dataset':YoutubeDatasetVal},
					"ReDWeb_DIW":{'train_dataset':ReDWebDataset, 'val_dataset':DIWDatasetVal, 't_val_dataset':ReDWebDatasetVal},
					"SceneNet_DIW":{'train_dataset':SceneNetDataset, 'val_dataset':DIWDatasetVal, 't_val_dataset':SceneNetDatasetVal},
					"SceneNetMetric_DIW":{'train_dataset':SceneNetDataset_Metric, 'val_dataset':DIWDatasetVal, 't_val_dataset':SceneNetDataset_MetricVal},
					"YTmixReD_DIW":{'train_dataset':YTmixReDWebDataset, 'val_dataset':DIWDatasetVal, 't_val_dataset':YTmixReDWebDatasetVal}
					}

	# create dataset	
	t_dataset = DatasetsType[dataset_name]['train_dataset']( csv_filename= '../data/' + train_file, b_data_aug = b_data_aug, b_resnet_prep = b_resnet_prep, b_oppi = b_oppi )
	v_dataset = DatasetsType[dataset_name]['val_dataset']( csv_filename= '../data/' + valid_file, b_resnet_prep = b_resnet_prep )	
	tv_dataset = DatasetsType[dataset_name]['t_val_dataset']( csv_filename= '../data/' + train_file, b_resnet_prep = b_resnet_prep )

	t_data_loader = data.DataLoader(t_dataset, batch_size=batch_size, num_workers=num_loader_workers, shuffle=True, collate_fn = t_collate_fn)
	tv_data_loader = data.DataLoader(tv_dataset, batch_size=1, num_workers=0, shuffle=False, collate_fn = relative_depth_collate_fn)
	v_data_loader = data.DataLoader(v_dataset, batch_size=1, num_workers=0, shuffle=False, collate_fn = relative_depth_collate_fn)

	# create tensorboard logger
	logger = TBLogger.TBLogger(makedir_if_not_exist(config.JOBS_LOG_DIR))
	logger.create_scalar('Training Loss')
	logger.create_scalar('Train WKDR')
	logger.create_scalar('Val WKDR')
	# logger.create_image('Dummy image')
	# logger.create_histogram('Dummy histogram')
	

	cv2.setNumThreads(0)
	
	iter = 1
	best_v_WKDR = 100000
	for epoch in range(num_epoches):
		print "==============epoch = ", epoch
		for step, (inputs, target, input_res) in enumerate(t_data_loader):
			
			if iter >= num_iters:
				break
			
			###### zero gradient
			optimizer.zero_grad()

			###### read in training data
			input_var = Variable(inputs.cuda())
			if loss_name == 'L2_loss':
				target_var = Variable(target.cuda())
			else:
				target_var = [Variable(a.cuda()) for a in target]

			
			###### forwarding
			output_var = model(input_var)
			
			###### get loss
			loss = criterion(output_var, target_var)
			print iter, loss.data[0]
			
			###### back propagate			
			loss.backward()
			optimizer.step()

			###### save to log
			logger.add_value('Training Loss', loss.data[0], step=(iter + prev_iter) )


			if (iter + prev_iter) % model_save_interval == 0:
				save_model(optimizer, model, iter, prev_iter)

			if (iter + prev_iter) % model_eval_interval == 0:				
				print "Evaluating at iter %d" % iter
				model.eval()
				if n_GPUs > 1:		
					print "========================================validation set"
					v_rel_error = valid.valid(model.module, v_data_loader, criterion, in_thresh=0.0)
					print "========================================training set"
					t_rel_error = valid.valid(model.module, tv_data_loader, criterion, in_thresh=0.0, max_iter=500)
				else:
					print "========================================validation set"
					v_rel_error = valid.valid(model, v_data_loader, criterion, in_thresh=0.0)
					print "========================================training set"
					t_rel_error = valid.valid(model, tv_data_loader, criterion, in_thresh=0.0, max_iter=500)
				logger.add_value('Val WKDR', v_rel_error['WKDR_neq'], step=(iter + prev_iter) )
				logger.add_value('Train WKDR', t_rel_error['WKDR_neq'], step=(iter + prev_iter))
				model.train()
				if best_v_WKDR > v_rel_error['WKDR_neq']:
					best_v_WKDR = v_rel_error['WKDR_neq']
					save_model(optimizer, model, iter, prev_iter, prefix = 'best_')
				else:
					save_model(optimizer, model, iter, prev_iter)
					
			iter += 1

			inputs = None
			target = None
			input_res = None


		if iter >= num_iters:
			break

	

	save_model(optimizer, model, iter, prev_iter)



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# parser.add_argument('--network_name', '-nn', default=config.DEFAULT_NETWORK_NAME)
	parser.add_argument('--train_file', '-t', default='overfit_NYU.csv')
	parser.add_argument('--valid_file', '-v', default='overfit_NYU.csv')
	parser.add_argument('--dataset_name', '-dn', default='RelativeDepthDataset') # can be YoutubeDataset as well
	parser.add_argument('--model_name', '-mn', default='NIPS') # can be SHG as well
	parser.add_argument('--loss_name', default='RelativeLoss') # can be RelativeLossSnap as well
	# parser.add_argument('--optim_name', '-on', default=config.DEFAULT_OPTIM_NAME)
	parser.add_argument('--num_iters', '-iter', default=100000, type=int)
	parser.add_argument('--num_epoches', '-ne', default=100000, type=int)
	parser.add_argument('--batch_size', '-bs', default=4, type=int)
	parser.add_argument('--model_save_interval', '-mt', default=5000, type=int)
	parser.add_argument('--model_eval_interval', '-et', default=3000, type=int)
	parser.add_argument('--learning_rate', '-lr', default=0.001, type=float)
	parser.add_argument('--n_GPUs', '-ngpu', default=1, type=int)	
	parser.add_argument('--num_loader_workers', '-nlw', type=int, default=2)
	parser.add_argument('--pretrained_file', '-pf', default=None)
	parser.add_argument('--b_oppi', '-b_oppi', action='store_true', default=False)
	parser.add_argument('--b_sort', '-b_sort', action='store_true', default=False)
	parser.add_argument('--b_data_aug', '-b_data_aug', action='store_true', default=False)
	parser.add_argument('--b_diff_lr', '-b_diff_lr', action='store_true', default=False)
	# parser.add_argument('--debug', '-d', action='store_true')

	args = parser.parse_args()

	args_dict = vars(args)

	folder = makedir_if_not_exist(config.JOBS_DIR)
	save_obj(args_dict, os.path.join(config.JOBS_DIR, 'args.pkl'))
	
	train(**args_dict)

	print "End of train.py"






