from __future__ import division
import numpy as np
import nn
import scipy.io as sio
import argparse


# Author: Kejun Tang
# Last Revised: Nov 28, 2018


def parse_args():
	desc = "variational principle based on deep nets for steady state diffusion problem"
	parser = argparse.ArgumentParser(description=desc)
	parser.add_argument('--spatial_range', type=list, default=[-1,1], help='the range of spatial domain')
	parser.add_argument('--pde_type', type=str, default='Diffusion', help='the type of PDE')
	parser.add_argument('--net_type', type=str, default='FC', help='the type of neural networks')
	parser.add_argument('--num_hidden', type=int, default=256, help='units of hidden layer')
	parser.add_argument('--num_blocks', type=int, default=10, help='the number of residual blocks')
	parser.add_argument('--batch_size', type=int, default=200, help='batch size')
	parser.add_argument('--num_iters', type=int, default=10000, help='the number of iterations')
	parser.add_argument('--lr_rate', type=float, default=1e-3, help='step size for optimization')
	parser.add_argument('--lr_decay', type=float, default=1.0, help='decay rate after each 10000 iterations')
	parser.add_argument('--output_path', type=str, default='VarNets', help='batch size')

	return parser.parse_args()


def main():
	args = parse_args()
	if args is None:
		exit()

	# parse argument
	spatial_range = args.spatial_range
	pde_type = args.pde_type
	print('----pde_type-----', pde_type)
	net_type = args.net_type
	num_hidden = args.num_hidden
	num_blocks = args.num_blocks
	batch_size = args.batch_size
	num_iters = args.num_iters
	lr_rate = args.lr_rate
	lr_decay = args.lr_decay
	output_path = args.output_path

	# construct model
	VarPDE = nn.VarNets(spatial_range, pde_type, net_type, num_hidden, num_blocks, batch_size, num_iters, lr_rate, lr_decay, output_path)
	# train model 
	VarPDE.train()
	# record variational loss
	varloss_list = VarPDE.variational_loss_history
	np.save('varloss_list.npy', varloss_list)

	# test
	femsol_path = '/home/tkj/rap_prog/pyfile/t3f_UQ/data/square_diffsol.mat'
	grid_path = '/home/tkj/rap_prog/pyfile/t3f_UQ/data/squarediff_grid.mat'
	fem_sol = sio.loadmat(femsol_path)['x_gal']
	grid = sio.loadmat(grid_path)['xy']
	dnn_sol = VarPDE.test(grid)

	rel_error = np.linalg.norm(fem_sol-dnn_sol)/np.linalg.norm(fem_sol)
	print('Relative error: {:.6}' .format(rel_error))


if __name__ == '__main__':
	main()




