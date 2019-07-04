import numpy as np
import random
from utils import save_obj, load_obj

import torch

from torch.utils import data
import cv2
import os
import random
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

	def __str__(self):
		out_str = 'Data Augmenter:\n'
		for op in self.ops:
			out_str += '\t'
			for key in op.keys():
				out_str = out_str + str(key) +':'+ str(op[key]) + '\t'
			out_str += '\n'
		return out_str

	def aug(self, img, target):
		"""
			img and target are 2D numpy array with the same size
			img should be H x W x 3
			target should be H x W
		"""
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

					#################### crop
					img = img[offset_y:offset_y+dst_h, offset_x:offset_x+dst_w, :]
					target = target[offset_y:offset_y+dst_h, offset_x:offset_x+dst_w]
					

				elif op['type'] == 'flip_lr':
					# print "Flipping..................."
					#################### image
					img = cv2.flip(img, 1)
					#################### target
					target = cv2.flip(target, 1)

				elif op['type'] == 'zoom':
					# print "Zooming..................."
					percentage = random.uniform(op['min_percentage'], op['max_percentage'])

					#################### image
					img = cv2.resize(img, None, fx = percentage, fy = percentage)

					#################### target
					target = cv2.resize(target, None, fx = percentage, fy = percentage)

				elif op['type'] == 'rotation':
					# print "Rotating..................."
					#################### image
					angle = random.uniform(-op['max_left_rotation'], op['max_right_rotation'])
					rotation_matrix = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), angle, 1.0)


					img = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))
					target = cv2.warpAffine(target, rotation_matrix, (img.shape[1], img.shape[0]))
					

					# self.draw(img, target, '3_rotation.png')



		return img, target

class ReDWebDataset(data.Dataset):
	def __init__(self, csv_filename, 
					   height=240, width=320, 
					   b_oppi = False, 
					   b_data_aug = False,
					   b_resnet_prep = False):

		super(ReDWebDataset, self).__init__()
		print("=====================================================")
		print "Using ReDWebDataset..."
		self.parse_ReDWebDataset_txt(csv_filename)
		if b_resnet_prep:
			self.height = 384
			self.width = 384
		else:
			self.height = height
			self.width = width
		
		self.n_sample = len(self.img_names)
		self.b_oppi = b_oppi 	# only take one relative depth pair per image
		self.b_resnet_prep = b_resnet_prep
		self.b_data_aug = b_data_aug
		self.n_pairs = 3000
		print "\t-(width, height): (%d, %d)" % (self.width, self.height)
		print "\t-%s: %d samples" % (csv_filename, self.n_sample)
		print "\t-One relative depth pair per image:", self.b_oppi
		print "\t-Data augmentation:", self.b_data_aug
		print "\t-Resnet data preprocessing:", self.b_resnet_prep
		print "\t-number of randomly sampled pairs:", self.n_pairs
		print("=====================================================")




		if self.b_data_aug:
			self.da = data_augmenter(width = self.width, height = self.height)
			self.da.add_zoom(0.8, min_percentage = 0.5, max_percentage = 3.0)
			self.da.add_crop(1.1, min_percentage = 0.5)
			self.da.add_rotation(0.8, max_left_rotation = -10.0, max_right_rotation = 10.0)
			self.da.add_flip_left_right(0.5)
			print self.da

	def parse_txt_meta_data(self, txt_filename):
		img_names = []
		depth_names = []
		

		with open(txt_filename, 'r') as f:					
			while True:
				line = f.readline().strip()
				if not line:
					break

				img_name = '/home/wfchen/ReDWeb_V1/Imgs/%s.jpg' % line
				depth_name = '/home/wfchen/ReDWeb_V1/RDs/%s.png' % line

				img_names.append(img_name)
				depth_names.append(depth_name)

		return img_names, depth_names



	def parse_ReDWebDataset_txt(self, txt_filename):
		meta_filename = txt_filename.replace('.txt', '.meta')
		if not os.path.exists(meta_filename):
			print meta_filename, "does not exist. Creating..."
			self.img_names, self.depth_names = self.parse_txt_meta_data(txt_filename)
			save_obj({"img_names":self.img_names, "depth_names":self.depth_names}, meta_filename, verbal = True)
		else:
			print "Loading ", meta_filename
			temp = load_obj(meta_filename, verbal = True)
			self.img_names = temp["img_names"]
			self.depth_names = temp["depth_names"]
		

	def __getitem__(self, index):
		# This data reader assumes that the target coordinates are represented 
		# by value in [0, 1.0], i.e., the ratio between the original coordinate
		# and the original image height / image width

		color = cv2.imread(self.img_names[index])		
		orig_img_res = color.shape[:2]
		
		depth = cv2.imread(self.depth_names[index])[:,:,0]
		depth = cv2.resize(depth, (orig_img_res[1], orig_img_res[0]))



		if self.b_data_aug:
			color, depth = self.da.aug(color, depth)
		
		#####################################################
		# resize to network input size
		color = cv2.resize(color, (self.width, self.height))
		depth = cv2.resize(depth, (self.width, self.height))

		#####################################################
		# obtain ground truth relative depth pairs
		y_A = np.random.random_integers(low = 0, high = self.height - 1, size = self.n_pairs)
		x_A = np.random.random_integers(low = 0, high = self.width - 1,  size = self.n_pairs)
		y_B = np.random.random_integers(low = 0, high = self.height - 1, size = self.n_pairs)
		x_B = np.random.random_integers(low = 0, high = self.width - 1,  size = self.n_pairs)
		rand_idx_A = [y * self.width + x for y,x in zip(y_A, x_A)]
		rand_idx_B = [y * self.width + x for y,x in zip(y_B, x_B)]		
		reshape_depth_A = np.reshape(depth, -1)[rand_idx_A]
		reshape_depth_B = np.reshape(depth, -1)[rand_idx_B]
		reshape_depth_A = reshape_depth_A.astype(np.float32) + 0.00000000001
		reshape_depth_B = reshape_depth_B.astype(np.float32) + 0.00000000001
		
		depth_A_div_by_B = np.divide(reshape_depth_A, reshape_depth_B)
		depth_B_div_by_A = np.divide(reshape_depth_B, reshape_depth_A)


		rel_depth = np.zeros(self.n_pairs, dtype =np.int64)
		rel_depth[depth_A_div_by_B >= 1.02] = 1
		rel_depth[depth_B_div_by_A >= 1.02] = -1

		target = np.stack([y_A, x_A, y_B, x_B, rel_depth], axis = 0)


		#####################################################
		# last step of preprocessing
		color = color.transpose(2, 0, 1).astype(np.float32) / 255.0		
		if self.b_resnet_prep:
			color = resNet_data_preprocess(color)

		return color, target.astype(np.int64), (self.height, self.width)



	def __len__(self):
		return self.n_sample


class ReDWebDatasetVal(ReDWebDataset):
	def __init__(self, csv_filename, 
						height=240, width=320, 
						b_oppi = False, 
						b_resnet_prep = False):
		print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
		print("\tValidation version of the ReDWebDataset")
		print("\t\t-It never perform data augmentation")
		ReDWebDataset.__init__(self, csv_filename, 
										height = height, width = width, 
										b_oppi = b_oppi, 
										b_data_aug = False, 
										b_resnet_prep = b_resnet_prep)
		

	def __getitem__(self, index):
		#####################################################
		# read in color 
		color = cv2.imread(self.img_names[index])		
		depth = cv2.imread(self.depth_names[index])[:,:,0]


		#####################################################
		# resize to network input size
		orig_img_res = color.shape[:2]
		color = cv2.resize(color, (self.width, self.height))
		depth = cv2.resize(depth, (orig_img_res[1], orig_img_res[0]))

		# no need to resize the ground truth depth


		#####################################################
		# last step of preprocessing
		color = color.transpose(2, 0, 1).astype(np.float32) / 255.0				
		if self.b_resnet_prep:
			color = resNet_data_preprocess(color)

			
		#####################################################
		# obtain ground truth relative depth pairs
		y_A = np.random.random_integers(low = 0, high = orig_img_res[0] - 1, size = self.n_pairs)
		x_A = np.random.random_integers(low = 0, high = orig_img_res[1] - 1,  size = self.n_pairs)
		y_B = np.random.random_integers(low = 0, high = orig_img_res[0] - 1, size = self.n_pairs)
		x_B = np.random.random_integers(low = 0, high = orig_img_res[1] - 1,  size = self.n_pairs)
		rand_idx_A = [y * orig_img_res[1] + x for y,x in zip(y_A, x_A)]
		rand_idx_B = [y * orig_img_res[1] + x for y,x in zip(y_B, x_B)]		
		reshape_depth_A = np.reshape(depth, -1)[rand_idx_A]
		reshape_depth_B = np.reshape(depth, -1)[rand_idx_B]
		
		reshape_depth_A = reshape_depth_A.astype(np.float32) + 0.00000000001
		reshape_depth_B = reshape_depth_B.astype(np.float32) + 0.00000000001
		
		depth_A_div_by_B = np.divide(reshape_depth_A, reshape_depth_B)
		depth_B_div_by_A = np.divide(reshape_depth_B, reshape_depth_A)


		rel_depth = np.zeros(self.n_pairs, dtype =np.int64)
		rel_depth[depth_A_div_by_B >= 1.02] = 1
		rel_depth[depth_B_div_by_A >= 1.02] = -1
		bool_care = rel_depth!= 0
		y_A = y_A[bool_care]
		x_A = x_A[bool_care]
		y_B = y_B[bool_care]
		x_B = x_B[bool_care]
		rel_depth = rel_depth[bool_care]
		target = np.stack([y_A, x_A, y_B, x_B, rel_depth], axis = 0)


		# target[:4,:] = target[:4,:] - 1		# the coordinate in python starts from 0!!!!





		return color, target.astype(np.int64), orig_img_res
