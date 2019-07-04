import numpy as np
import random
from utils import save_obj, load_obj

import torch

from torch.utils import data
import cv2
import os
from ReDWebNet import resNet_data_preprocess


def draw(img, target, fname):
	img_temp = img.copy()
	
	color_close = (255, 0, 0)	# close is blue
	color_far = (0, 255, 0)		# far is green
	for i in range(target.shape[1]):
		x1 = int(target[1, i]); y1 = int(target[0, i]);
		x2 = int(target[3, i]); y2 = int(target[2, i]);
		
		cv2.circle(img_temp,(x1, y1),2,color_far,-1)
		cv2.circle(img_temp,(x2, y2),2,color_close,-1)
		cv2.arrowedLine(img_temp, (x2, y2), (x1, y1), (0, 255, 255), 1)
	
	cv2.imwrite(fname, img_temp)
	print "Done writing to %s" % fname 


class data_augmenter():
	def __init__(self, width, height):
		"""
			Args:
				width and height are only used to determine the 
				output aspect ratio, not the actual output size
		"""
		self.ops = []
		cv2.setNumThreads(0)
		self.width = float(width)
		self.height = float(height)
		
	def add_rotation(self, probability, max_left_rotation=-10, max_right_rotation=10):
		self.ops.append({'type':'rotation', 'probability':probability, 'max_left_rotation': max_left_rotation, 'max_right_rotation':max_right_rotation})
	def add_zoom(self, probability, min_percentage, max_percentage):
		self.ops.append({'type':'zoom', 'probability':probability, 'min_percentage': min_percentage, 'max_percentage': max_percentage})
	def add_flip_left_right(self, probability):
		self.ops.append({'type':'flip_lr', 'probability':probability})
	def add_crop(self, probability, min_percentage=0.5):
		self.ops.append({'type':'crop', 'probability':probability, 'min_percentage':min_percentage})
	def draw(self, img, target, fname):
		img_temp = img.copy()
		
		color_close = (255, 0, 0)	# close is blue
		color_far = (0, 255, 0)		# far is green
		for i in range(target.shape[1]):
			x1 = int(target[1, i]); y1 = int(target[0, i]);
			x2 = int(target[3, i]); y2 = int(target[2, i]);
			
			cv2.circle(img_temp,(x1, y1),2,color_far,-1)
			cv2.circle(img_temp,(x2, y2),2,color_close,-1)
			cv2.arrowedLine(img_temp, (x2, y2), (x1, y1), (0, 255, 255), 1)
		
		cv2.imwrite(fname, img_temp)
		print "Done writing to %s" % fname 

	def __str__(self):
		out_str = 'Data Augmenter:\n'
		for op in self.ops:
			out_str += '\t'
			for key in op.keys():
				out_str = out_str + str(key) +':'+ str(op[key]) + '\t'
			out_str += '\n'
		return out_str

	def aug(self, img, target):
		orig_img = img.copy()
		orig_target = target.copy()
		
		for op in self.ops:
			if random.uniform(0.0, 1.0) <= op['probability']:
				if op['type'] == 'crop':
					percentage = random.uniform(op['min_percentage'], 1.0)
					# print "Cropping.: Percentage = %f" % percentage
					#################### image
					if img.shape[0] <= img.shape[1]:
						dst_h = int(img.shape[0] * percentage)
						dst_w = min(int(dst_h / self.height * self.width), img.shape[1])
					elif img.shape[0] > img.shape[1]:
						dst_w = int(img.shape[1] * percentage)
						dst_h = min(int(dst_w / self.width * self.height), img.shape[0])
					offset_y = random.randint(0, img.shape[0]- dst_h)
					offset_x = random.randint(0, img.shape[1]- dst_w)
					img = img[offset_y:offset_y+dst_h, offset_x:offset_x+dst_w, :]
					
					#################### target
					target[0,:] = target[0,:] - offset_y
					target[1,:] = target[1,:] - offset_x
					target[2,:] = target[2,:] - offset_y
					target[3,:] = target[3,:] - offset_x
					mask = target[0,:] < dst_h
					mask = np.logical_and(mask, target[1,:] < dst_w)
					mask = np.logical_and(mask, target[2,:] < dst_h)
					mask = np.logical_and(mask, target[3,:] < dst_w)
					mask = np.logical_and(mask, target[0,:] >= 0)
					mask = np.logical_and(mask, target[1,:] >= 0)
					mask = np.logical_and(mask, target[2,:] >= 0)
					mask = np.logical_and(mask, target[3,:] >= 0)

					# self.draw(img, target, '2_crop.png')

					if np.sum(mask) == 0:
						print "Fail at cropping"
						return orig_img, orig_target
					else:
						target = target[:, mask]						

					

				elif op['type'] == 'flip_lr':
					# print "Flipping..................."
					#################### image
					img = cv2.flip(img, 1)

					#################### target
					target[1,:] = img.shape[1] - target[1,:]
					target[3,:] = img.shape[1] - target[3,:]
					# self.draw(img, target, '4_flip.png')

				elif op['type'] == 'zoom':
					# print "Zooming..................."
					#################### image
					percentage = random.uniform(op['min_percentage'], op['max_percentage'])
					img = cv2.resize(img, None, fx = percentage, fy = percentage)

					#################### target
					target[0:4,:] = target[0:4,:] * percentage					
					# self.draw(img, target, '1_zoom.png')

				elif op['type'] == 'rotation':
					# print "Rotating..................."
					#################### image
					angle = random.uniform(-op['max_left_rotation'], op['max_right_rotation'])
					rotation_matrix = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), angle, 1.0)
					img = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))
					
					#################### target
					temp = rotation_matrix[0,:].copy()
					rotation_matrix[0,:] = rotation_matrix[1,:]
					rotation_matrix[1,:] = temp
					temp = rotation_matrix[:,0].copy()
					rotation_matrix[:,0] = rotation_matrix[:,1]
					rotation_matrix[:,1] = temp
					target[0:2,:] = rotation_matrix[:,0:2].dot(target[0:2,:]) + rotation_matrix[:,2:3]
					target[2:4,:] = rotation_matrix[:,0:2].dot(target[2:4,:]) + rotation_matrix[:,2:3]
					mask = target[0,:] < img.shape[0]
					mask = np.logical_and(mask, target[1,:] < img.shape[1])
					mask = np.logical_and(mask, target[2,:] < img.shape[0])
					mask = np.logical_and(mask, target[3,:] < img.shape[1])
					mask = np.logical_and(mask, target[0,:] >= 0)
					mask = np.logical_and(mask, target[1,:] >= 0)
					mask = np.logical_and(mask, target[2,:] >= 0)
					mask = np.logical_and(mask, target[3,:] >= 0)
					if np.sum(mask) == 0:
						print "Fail at rotation"
						return orig_img, orig_target
					else:
						target = target[:, mask]

					# self.draw(img, target, '3_rotation.png')



		return img, target

#######################################################################
##### ATTENTION:
##### This dataset only works with the Integer DIW csv files
class DIWDataset(data.Dataset):
	def __init__(self, csv_filename, 
						height=240, width=320, 
						b_data_aug = False,
						b_resnet_prep = False,
						b_oppi = False
						):
		super(DIWDataset, self).__init__()
		print("=====================================================")
		print "Using DIWDataset..."
		self.parse_relative_depth_csv(csv_filename)
		if b_resnet_prep:
			self.height = 384
			self.width = 384
		else:
			self.height = height
			self.width = width
		self.n_sample = len(self.img_names)
		self.b_resnet_prep = b_resnet_prep
		self.b_data_aug = b_data_aug		
		print "\t-(width, height): (%d, %d)" % (self.width, self.height)
		print "\t-%s: %d samples" % (csv_filename, self.n_sample)
		print "\t-Data augmentation:", self.b_data_aug
		print "\t-Resnet data preprocessing:", self.b_resnet_prep
		print("=====================================================")


		if self.b_data_aug:
			self.da = data_augmenter(width = self.width, height = self.height)
			self.da.add_zoom(0.8, min_percentage = 0.5, max_percentage = 3.0)
			self.da.add_crop(1.1, min_percentage = 0.5)
			self.da.add_rotation(0.8, max_left_rotation = -10.0, max_right_rotation = 10.0)
			self.da.add_flip_left_right(0.5)
			print self.da


	def parse_csv_meta_data(self, csv_filename):
		print "Parsing ", csv_filename
		img_names = []
		y_A_x_A_y_B_x_B_rel = []
		with open(csv_filename, 'r') as f:	
			f.readline()		
			while True:
				dummy_info = f.readline()
				if not dummy_info:
					break
				infos = dummy_info.split(',')

				img_name, n_point = infos[0], int(infos[2])
				
				img_names.append(img_name)				
				assert n_point == 1
				
				# parse coordinates and relation, only one line
				coords = f.readline()
				y_A, x_A, y_B, x_B, rel, _a, _b = coords[:-1].split(',')
				data = np.zeros((5, 1))
				data[0,0] = float(y_A)
				data[1,0] = float(x_A)
				data[2,0] = float(y_B)
				data[3,0] = float(x_B)
				data[4,0] = {'<' : -1, '>' : 1}[rel]				
				y_A_x_A_y_B_x_B_rel.append(data)


		return img_names, y_A_x_A_y_B_x_B_rel



	def parse_relative_depth_csv(self, csv_filename):
		name_filename = csv_filename.replace('.csv', '.meta')
		if not os.path.exists(name_filename):			
			self.img_names, self.y_A_x_A_y_B_x_B_rel = self.parse_csv_meta_data(csv_filename)
			save_obj({"img_names":self.img_names, "y_A_x_A_y_B_x_B_rel":self.y_A_x_A_y_B_x_B_rel}, name_filename, verbal = True)
		else:
			print "Loading ", name_filename
			temp = load_obj(name_filename)
			self.img_names = temp["img_names"]
			self.y_A_x_A_y_B_x_B_rel = temp["y_A_x_A_y_B_x_B_rel"]
		

	def __getitem__(self, index):
		# The input coordinates are defined within the original image size.
		# Need to scale accordingly to fit the network size.

		color = cv2.imread(self.img_names[index])
		orig_img_res = color.shape[:2]


		target = self.y_A_x_A_y_B_x_B_rel[index].copy()
		# draw(color, target, '0_orig.png')
		if self.b_data_aug:
			color, target = self.da.aug(color, target)
			# print target

		target[0,:] = target[0,:] / float(color.shape[0]) * self.height		#y_A
		_dummy = target[0,:]; _dummy[_dummy>self.height] = self.height;	_dummy[_dummy < 1] = 1
		
		target[1,:] = target[1,:] / float(color.shape[1]) * self.width		#x_A
		_dummy = target[1,:]; _dummy[_dummy>self.width] = self.width;	_dummy[_dummy < 1] = 1

		target[2,:] = target[2,:] / float(color.shape[0]) * self.height		#y_B
		_dummy = target[2,:]; _dummy[_dummy>self.height] = self.height;	_dummy[_dummy < 1] = 1
		
		target[3,:] = target[3,:] / float(color.shape[1]) * self.width		#x_B
		_dummy = target[3,:]; _dummy[_dummy>self.width] = self.width;	_dummy[_dummy < 1] = 1

		target[:4,:] = target[:4,:] - 1		# the coordinate in python starts from 0!!!!


		color = cv2.resize(color, (self.width, self.height))
		# draw(color, target, '5_final.png')
		# raw_input()

		color = color.transpose(2, 0, 1).astype(np.float32) / 255.0	
		if self.b_resnet_prep:
			color = resNet_data_preprocess(color)

		return color, target.astype(np.int64), (self.height, self.width)



	def __len__(self):
		return self.n_sample



def relative_depth_collate_fn(batch):
	return (torch.stack([torch.from_numpy(b[0]) for b in batch], 0),  [torch.from_numpy(b[1]) for b in batch], [b[2] for b in batch]  )


class DIWDatasetVal(DIWDataset):
	def __init__(self, csv_filename, height=240, width=320, b_resnet_prep = False):
		print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
		print("\tValidation version of the DIWDataset")
		print("\t\t-It never perform data augmentation")
		DIWDataset.__init__(self, csv_filename, 
								  height, width, 
								  b_data_aug = False,
								  b_resnet_prep = b_resnet_prep,
								  b_oppi = False)
		assert(not self.b_data_aug)


	def __getitem__(self, index):
		# The input coordinates are defined within the original image size.
		# When in validation, you evaluate on the original image resolution.
		#####################################################################
		color = cv2.imread(self.img_names[index])
		orig_img_res = color.shape[:2]
		color = cv2.resize(color, (self.width, self.height))
		color = color.transpose(2, 0, 1).astype(np.float32) / 255.0		
		if self.b_resnet_prep:
			color = resNet_data_preprocess(color)


		#####################################################################
		target = self.y_A_x_A_y_B_x_B_rel[index].copy()
		target[:4,:] = target[:4,:] - 1		# the coordinate in python starts from 0!!!!


		return color, target.astype(np.int64), orig_img_res

