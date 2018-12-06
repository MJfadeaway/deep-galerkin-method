from __future__ import division
import numpy as np
import tensorflow as tf
import nn
import scipy.io as sio
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Author: Kejun Tang
# Last Revised: Dec 06, 2018


def parse_args():
	desc = "variational principle based on deep nets for steady state diffusion problem"
	parser = argparse.ArgumentParser(description=desc)
	parser.add_argument('--spatial_range', type=list, default=[0,1], help='the range of spatial domain')
	parser.add_argument('--pde_type', type=str, default='Stokes', help='the type of PDE')
	parser.add_argument('--net_type', type=str, default='Res', help='the type of neural networks')
	parser.add_argument('--num_hidden', type=int, default=256, help='units of hidden layer')
	parser.add_argument('--num_blocks', type=int, default=10, help='the number of residual blocks')
	parser.add_argument('--batch_size', type=int, default=200, help='batch size')
	parser.add_argument('--num_iters', type=int, default=100, help='the number of iterations')
	parser.add_argument('--lr_rate', type=float, default=1e-3, help='step size for optimization')
	parser.add_argument('--lr_decay', type=float, default=1.0, help='decay rate after each 10000 iterations')
	parser.add_argument('--output_path', type=str, default='VarNets', help='output path for model')

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

	# test
	if pde_type == 'Diffusion':
		# record variational loss
		varloss_list = VarPDE.variational_loss_history
		np.save('varloss_list.npy', varloss_list)
		femsol_path = '/home/tkj/rap_prog/pyfile/t3f_UQ/data/square_diffq2sol.mat'
		grid_path = '/home/tkj/rap_prog/pyfile/t3f_UQ/data/squarediff_grid.mat'
		fem_sol = sio.loadmat(femsol_path)['x_gal']
		grid = sio.loadmat(grid_path)['xy']
		grid = tf.convert_to_tensor(grid, dtype=tf.float32)
		dnn_sol, problem_type = VarPDE.test(grid)
		rel_error = np.linalg.norm(fem_sol-dnn_sol)/np.linalg.norm(fem_sol)
		print('Relative error: {:.6}' .format(rel_error))
		print(problem_type)

		nodes = np.linspace(-1, 1, 33)
		Y_nodes, X_nodes = np.meshgrid(nodes, nodes) # FEM mesh structure

		f_value = np.reshape(dnn_sol, (33,33), order='F')
		fig = plt.figure(1)
		ax = Axes3D(fig)
		ax.plot_surface(X_nodes, Y_nodes, f_value, rstride=1, cstride=1, cmap=plt.cm.coolwarm)
		ax.set_xlabel('x label', color='r')
		ax.set_ylabel('y label', color='g')
		ax.set_zlabel('z label', color='b')

		f_value1 = np.reshape(fem_sol, (33,33), order='F')
		fig = plt.figure(2)
		ax = Axes3D(fig)
		ax.plot_surface(X_nodes, Y_nodes, f_value1, rstride=1, cstride=1, cmap=plt.cm.coolwarm)
		ax.set_xlabel('x label', color='r')
		ax.set_ylabel('y label', color='g')
		ax.set_zlabel('z label', color='b')
		plt.show()
	elif pde_type == 'Stokes':
		# record variational loss
		velocity_varloss_list = VarPDE.velocity_varloss_history
		pressure_varloss_list = VarPDE.pressure_varloss_history
		np.save('velocity_varloss_list.npy', velocity_varloss_list)
		np.save('pressure_varloss_list.npy', pressure_varloss_list)
		#femsol_path = '/home/tkj/rap_prog/pyfile/t3f_UQ/data/stokes_driven_cavity.mat'
		gridv_path = '/home/tkj/rap_prog/pyfile/t3f_UQ/data/stokes_grid.mat'
		gridp_path = '/home/tkj/rap_prog/pyfile/t3f_UQ/data/stokes_pressure_grid.mat'
		#fem_sol = sio.loadmat(femsol_path)['u']
		gridv = sio.loadmat(gridv_path)['xy']
		gridv = tf.convert_to_tensor(gridv, dtype=tf.float32)
		gridp = sio.loadmat(gridp_path)['xyp']
		gridp = tf.convert_to_tensor(gridp, dtype=tf.float32)
		grid = (gridv, gridp)
		dnn_sol, problem_type = VarPDE.test(grid)
		sio.savemat('dnn_stokessol.mat', {'stokes_sol': dnn_sol})
		print(problem_type)
	else:
		raise ValueError('not implementation for Helmholtz')




if __name__ == '__main__':
	main()




