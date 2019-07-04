from YoutubeDataset import YoutubeDataset,YoutubeDatasetVal
from ReDWebDataset import ReDWebDataset,ReDWebDatasetVal

from torch.utils import data


YT1M_Train_Val_CSV = '../data/YT8M/t_yteven1000000_0.85.csv'
ReDWeb_Train_Val_CSV = '../data/ReDWeb_V1_data.txt'
ReDWeb_Magnifier = 30

class YTmixReDWebDataset(data.Dataset):
	def __init__(self, csv_filename, 
					   height=240, width=320, 
					   b_oppi = False, 
					   b_data_aug = False,
					   b_resnet_prep = False):
		print "========================================================================"
		print "YTmixReDWebDataset Training Dataset"
		self.ReDWeb_Dataset = ReDWebDataset(ReDWeb_Train_Val_CSV,
										    height, width, 
										    False, 
										    b_data_aug,
										    b_resnet_prep)
		self.YT_Dataset = YoutubeDataset( YT1M_Train_Val_CSV, 
										  height, width, 
										  False, 
										  b_data_aug,
										  b_resnet_prep)

		self.n_ReDWeb = self.ReDWeb_Dataset.__len__()
		self.n_YT = self.YT_Dataset.__len__()

	def __getitem__(self, index):
		if index < self.n_YT:			
			
			return self.YT_Dataset[index]

		else:
			index = index - self.n_YT
			index = index % self.n_ReDWeb
			return self.ReDWeb_Dataset[index]

	def __len__(self):
		return self.n_ReDWeb * ReDWeb_Magnifier + self.n_YT



class YTmixReDWebDatasetVal(data.Dataset):
	def __init__(self, csv_filename, 
					   height = 240, width=320, 
					   b_oppi = False, 
					   b_resnet_prep = False):
		print "========================================================================"
		print "YTmixReDWebDataset Validation Dataset"
		self.ReDWeb_Dataset = ReDWebDatasetVal(ReDWeb_Train_Val_CSV,
										       height, width, 
										       b_oppi, 
										       b_resnet_prep)
		self.YT_Dataset = YoutubeDatasetVal(YT1M_Train_Val_CSV, 
										    height, width, 
										    b_oppi, 										 
										    b_resnet_prep)

		self.n_ReDWeb = self.ReDWeb_Dataset.__len__()
		self.n_YT = self.YT_Dataset.__len__()

	def __getitem__(self, index):
		if index < self.n_YT:			
			
			return self.YT_Dataset[index]

		else:
			index = index - self.n_YT
			index = index % self.n_ReDWeb
			return self.ReDWeb_Dataset[index]

	def __len__(self):
		return self.n_ReDWeb * ReDWeb_Magnifier + self.n_YT
