import numpy as np
import random
from utils import save_obj, load_obj

import torch

from torch.utils import data
import cv2
import os
import h5py
from ReDWebNet import resNet_data_preprocess

#######################################################################
##### ATTENTION:
##### When using this dataset, set the number of loading worker to 0.
##### 		There is problem using hdf5 and multi-threading together.
class RelativeDepthDataset(data.Dataset):
	def __init__(self, csv_filename, height=240, width=320,
									 b_data_aug = False, 
									 b_resnet_prep = False):
		"""
			b_data_aug is a dummy var, not used.
		"""
		super(RelativeDepthDataset, self).__init__()
		print("=====================================================")
		print "Using RelativeDepthDataset..."
		self.parse_relative_depth_csv(csv_filename)
		self.height = height
		self.width = width		
		self.n_sample = len(self.img_names)
		self.b_resnet_prep = b_resnet_prep
		print "\t-(width, height): (%d, %d)" % (self.width, self.height)
		print "\t-%s: %d samples" % (csv_filename, self.n_sample)
		print "\t-Resnet data preprocessing:", self.b_resnet_prep		
		print("=====================================================")

	def parse_csv_meta_data(self, csv_filename):
		img_names = []
		n_pairs = []
		with open(csv_filename, 'r') as f:	
			f.readline()		
			while True:
				dummy_info = f.readline()
				if not dummy_info:
					break
				infos = dummy_info.split(',')

				img_name, n_point = infos[0], int(infos[2])
				
				img_names.append(img_name)
				n_pairs.append(n_point)

				for i in range(n_point):
					f.readline()

		n_pairs = np.array(n_pairs)		

		return img_names, n_pairs



	def parse_relative_depth_csv(self, csv_filename):
		hdf5_filename = csv_filename.replace('.csv', '.h5')
		if not os.path.exists(hdf5_filename):
			print "\tError: You need to have a hdf5 version of the csv file!"
		else:
			self.hdf5_handle = h5py.File(hdf5_filename, 'r')
		

		name_filename = csv_filename.replace('.csv', '.meta')
		if not os.path.exists(name_filename):
			self.img_names, self.n_pairs = self.parse_csv_meta_data(csv_filename)
			save_obj({"img_names":self.img_names, "n_pairs": self.n_pairs}, name_filename)
		else:
			temp = load_obj(name_filename)
			self.img_names = temp["img_names"]
			self.n_pairs = temp["n_pairs"]
		

	def __getitem__(self, index):
		# This data reader assumes that the target coordinates are in the 
		# same resolution as the input image, instead of the network input
		# resolution!
		# However, even though it resizes the input image, it does NOT 
		# resize the target accordingly!
		# Therefore, there can only be ONE kind of input when training:
		#	1. target = test = (240,320).
		# When validating / testing: 
		#	1. target = test = (240,320) 
		#	2. target = test = (480,640)
		
		color = cv2.imread(self.img_names[index])
		orig_img_res = color.shape[:2]
		color = cv2.resize(color, (self.width, self.height))
		color = color.transpose(2, 0, 1).astype(np.float32) / 255.0
		if self.b_resnet_prep:
			color = resNet_data_preprocess(color)
		n_pairs = self.n_pairs[index]

		_hdf5_offset = int(5*index) #zero-indexed
		target = self.hdf5_handle['/data'][_hdf5_offset:_hdf5_offset+5,0:n_pairs]
		target[:4,:] = target[:4,:] - 1		# the coordinate in python starts from 0!!!!

		return color, target.astype(np.int64), orig_img_res



	def __len__(self):
		return self.n_sample



def relative_depth_collate_fn(batch):
	return (torch.stack([torch.from_numpy(b[0]) for b in batch], 0),  [torch.from_numpy(b[1]) for b in batch], [b[2] for b in batch]  )


