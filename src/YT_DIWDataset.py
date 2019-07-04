from YoutubeDataset import YoutubeDataset, YoutubeDatasetVal
from DIWDataset import DIWDataset, DIWDatasetVal

from torch.utils import data

import numpy as np
import cv2
from utils import save_obj, load_obj

class YT_DIWDataset(data.Dataset):
	def __init__(self, csv_filename, height=240, width=320):
		print "========================================================================"
		print "YT_DIW Dataset"
		self.DIW_Dataset = DIWDataset(csv_filename = '../data/symm_DIW_release_train.csv')
		self.YT_Dataset = YoutubeDataset(csv_filename)

		self.height = height
		self.width = width

		self.n_DIW = self.DIW_Dataset.__len__()
		self.n_YT = self.YT_Dataset.__len__()

	def __getitem__(self, index):
		if index < self.n_DIW:
			color = cv2.imread(self.DIW_Dataset.img_names[index])
			orig_img_res = color.shape[:2]
			color = cv2.resize(color, (self.width, self.height))
			color = color.transpose(2, 0, 1).astype(np.float32) / 255.0		
			

			target = self.DIW_Dataset.y_A_x_A_y_B_x_B_rel[index].copy()

			target[0,:] = target[0,:] / orig_img_res[0] * self.height		#y_A
			_dummy = target[0,:]; _dummy[_dummy>self.height] = self.height;	_dummy[_dummy < 1] = 1
			
			target[1,:] = target[1,:] / orig_img_res[1] * self.width		#x_A
			_dummy = target[1,:]; _dummy[_dummy>self.width] = self.width;	_dummy[_dummy < 1] = 1

			target[2,:] = target[2,:] / orig_img_res[0] * self.height		#y_B
			_dummy = target[2,:]; _dummy[_dummy>self.height] = self.height;	_dummy[_dummy < 1] = 1
			
			target[3,:] = target[3,:] / orig_img_res[1] * self.width		#x_B
			_dummy = target[3,:]; _dummy[_dummy>self.width] = self.width;	_dummy[_dummy < 1] = 1

			target[:4,:] = target[:4,:] - 1		# the coordinate in python starts from 0!!!!


			return color, target.astype(np.int64), orig_img_res
			
		else:

			index = index - self.n_DIW

			color = cv2.imread(self.YT_Dataset.img_names[index])		
			color = cv2.resize(color, (self.width, self.height))
			color = color.transpose(2, 0, 1).astype(np.float32) / 255.0		


			target = load_obj(self.YT_Dataset.pkl_names[index])
			assert target.shape[0] == 5

			target[0,:] = target[0,:] * self.height		#y_A
			_dummy = target[0,:]; _dummy[_dummy>=self.height] = self.height - 1; _dummy[_dummy < 0] = 0
			
			target[1,:] = target[1,:] * self.width		#x_A
			_dummy = target[1,:]; _dummy[_dummy>=self.width] = self.width - 1; _dummy[_dummy < 0] = 0

			target[2,:] = target[2,:] * self.height		#y_B
			_dummy = target[2,:]; _dummy[_dummy>=self.height] = self.height - 1; _dummy[_dummy < 0] = 0
			
			target[3,:] = target[3,:] * self.width		#x_B
			_dummy = target[3,:]; _dummy[_dummy>=self.width] = self.width - 1; _dummy[_dummy < 0] = 0

			# target[:4,:] = target[:4,:] - 1		# the coordinate in python starts from 0!!!!

			return color, target.astype(np.int64), (self.height, self.width)
			

	def __len__(self):
		return self.n_DIW + self.n_YT


def relative_depth_collate_fn(batch):
	return (torch.stack([torch.from_numpy(b[0]) for b in batch], 0),  [torch.from_numpy(b[1]) for b in batch], [b[2] for b in batch]  )


