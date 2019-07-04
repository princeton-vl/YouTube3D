import torch
import math
from torch.autograd import Variable


class RelativeLoss(object):
	def __init__(self, b_sort = False):
		print "Using Relative Loss"
		self.b_sort = b_sort
		if self.b_sort:
			print("\t +++Sort and take the top 75 percent losses")

		

	def loss_func2(self, z_A, z_B, gt_r):
		n_point = z_A.data.shape[1]
		val_sum = 0.0
		# print gt_r.data[0],gt_r.data[1], gt_r.data[2]
		# print z_A.data[0,0], z_A.data[0,1], z_A.data[0,2]
		# print n_point
		# raw_input()
		for i in range(n_point):
			z_A_z_A = z_A.data[0,i] - z_B.data[0,i]
			ground_truth = gt_r.data[i]

			if ground_truth == 0:
				val = max(0, z_A_z_A * z_A_z_A)
			else:
				val = math.log( 1 + math.exp( - ground_truth * z_A_z_A ) )      

			val_sum += val

		return val_sum



	def loss_func(self, z_A, z_B, gt_r):
		mask = torch.abs(gt_r)
		z_A_z_B = z_A - z_B
		# print "z_A", z_A
		# print "z_B", z_B
		# print "mask", mask		
		# print "z_A_z_B", z_A_z_B
		# print "(1 - mask) * z_A_z_B * z_A_z_B", (1 - mask) * z_A_z_B * z_A_z_B
		# print "-1 * z_A_z_B * gt_r", -1 * z_A_z_B * gt_r	
		# print "torch.exp(- z_A_z_B * gt_r)", torch.exp(-1 * z_A_z_B * gt_r)	
		# print "torch.log( 1 + torch.exp(- z_A_z_B * gt_r) )", torch.log( 1 + torch.exp(-1 * z_A_z_B * gt_r) )	
		# print "mask * torch.log( 1 + torch.exp(- z_A_z_B * gt_r) )", mask * torch.log( 1 + torch.exp(-1 * z_A_z_B * gt_r) )

		return mask * torch.log( 1 + torch.exp(-1 * z_A_z_B * gt_r) ) + (1 - mask) * z_A_z_B * z_A_z_B		

	# outputs: nSamples x 1 x Height x Width.
	# targets: a list of nbatch elements, each is a tensor of 5 x n
	def __call__(self, outputs, target, mask=None):
		assert type(target) == list		
		assert len(target) == outputs.data.shape[0]

		# self.output = Variable(torch.Tensor([0])).cuda()
		losses = []
		losses2 = []
		n_point_total = 0.0
		for _idx, _gt in enumerate(target):
			depth = outputs[_idx, 0, ...]

			if self.b_sort:
				n_top_75_percent_pt_pair = int(_gt.data.shape[1] * 0.75)
				n_point_total += n_top_75_percent_pt_pair
			else:
				n_point_total += _gt.data.shape[1]


			y_A = _gt[0,:]
			x_A = _gt[1,:]
			y_B = _gt[2,:]
			x_B = _gt[3,:]
			gt_r = _gt[4,:].float()

			z_A_arr = depth.index_select(1, x_A).gather(0, y_A.view(1,-1))
			z_B_arr = depth.index_select(1, x_B).gather(0, y_B.view(1,-1))

			this_loss = self.loss_func(z_A_arr, z_B_arr, gt_r)

			if self.b_sort:
				sorted_loss, _ = torch.sort(this_loss, descending=True)
				losses.append(torch.sum(sorted_loss[:n_top_75_percent_pt_pair]))
			else:
				losses.append(torch.sum(this_loss))

			# #### debug
			# this_loss2 = self.loss_func2(z_A_arr, z_B_arr, gt_r)
			# losses2.append(this_loss2)

			# print z_A_arr.data.shape
			# print z_A_arr.data.shape
			# print z_A_arr.data[0,0], z_A_arr.data[0,1], z_A_arr.data[0,2], z_A_arr.data[0,3], z_A_arr.data[0,4]
			# print y_A.data.shape
			# for i in range(10):
			# 	print depth[y_A.data[i], x_A.data[i]]

		# return self.output / n_point_total
		# print losses
		return sum(losses) / n_point_total



if __name__ == '__main__':
	# testing
	crit = RelativeLoss()
	print(crit)
	x = Variable(torch.zeros(1,1,6,6).cuda(), requires_grad = True)
	target = torch.Tensor([[0,1,2,3,4,5], [0,1,2,3,4,5], [0,0,0,0,0,0], [5,4,3,2,1,0], [-1,0,1,1,-1,-1]]).long()
	target = [Variable(target.cuda())]
	# target[0]['x_A'] = Variable(torch.Tensor([0,1,2,3,4,5])).cuda()
	# target[0]['y_A'] = Variable(torch.Tensor([0,1,2,3,4,5])).cuda()
	# target[0]['x_B'] = Variable(torch.Tensor([0,0,0,0,0,0])).cuda()
	# target[0]['y_B'] = Variable(torch.Tensor([5,4,3,2,1,0])).cuda()
	# target[0]['ordianl_relation'] = Variable(torch.Tensor([-1,0,1,1,-1,-1])).cuda()
	# target[0]['n_point'] = 6
	loss = crit(x,target)
	print(loss)
	loss.backward()
	# a = crit.backward(1.0)
	# print(a)
	print(x.grad)
	# print(x.creator)