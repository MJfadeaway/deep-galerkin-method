from __future__ import division
import numpy as np
import tensorflow as tf
import os
import shutil

# Author: Kejun Tang
# Last Revised: Nov 28, 2018

class VarNets():
	"""VarNets for two dimensional PDEs"""

	"""
	
	"""

	def __init__(self, spatial_range, pde_type, net_type, num_hidden=256, num_blocks=10, batch_size=200, num_iters=100000, lr_rate=1e-3, lr_decay=1.0, output_path='VarNets'):
		self.spatial_range = spatial_range # for example, spatial range = [-1,1]
		self.pde_type = pde_type 
		self.net_type = net_type # FC or Res
		self.num_hidden = num_hidden
		self.num_blocks = num_blocks
		self.batch_size = batch_size
		self.num_iters = num_iters
		self.lr_rate = lr_rate
		self.lr_decay = lr_decay
		self.output_path = output_path
		self.variational_loss_history = []


	def fcnet(self, x_input):
		num_hidden = self.num_hidden
		with tf.variable_scope("fcnet", reuse=tf.AUTO_REUSE):
				h1 = tf.nn.relu(tf.layers.dense(x_input, num_hidden))
				h2 = tf.nn.relu(tf.layers.dense(h1, num_hidden))
				h3 = tf.nn.relu(tf.layers.dense(h2, num_hidden))
				h4 = tf.nn.relu(tf.layers.dense(h3, num_hidden))
				h5 = tf.nn.relu(tf.layers.dense(h4, num_hidden))
				h6 = tf.nn.relu(tf.layers.dense(h5, num_hidden))
				output = tf.layers.dense(h6, 1)
		return output

	def residual_block(self, y):
		num_hidden = self.num_hidden
		with tf.variable_scope("resblock", reuse=tf.AUTO_REUSE):
			output = tf.nn.relu(tf.layers.dense(y, num_hidden))
			output = tf.layers.dense(output, num_hidden)
			#resblock_outdim = boutput.get_shape().as_list()[-1]
			output = tf.nn.relu(output + tf.layers.dense(y, num_hidden, use_bias=False))
		return output


	def fcresnet(self, x_input):
		num_blocks = self.num_blocks
		response = x_input
		for idx_block in range(self.num_blocks):
			with tf.variable_scope("unit_%d" % idx_block, reuse=tf.AUTO_REUSE):
				response = self.residual_block(response)

		with tf.variable_scope("output"):
			response = tf.layers.dense(response, 1)
		return response

	def train(self):
		"""train step"""

		z = tf.placeholder(tf.float32, shape=[None, 2]) # two dimensional (the number of spatial variables) PDE

		if self.net_type == 'FC':
			u = ((z[:,0])**2 - 1) * ((z[:,1])**2 - 1) * self.fcnet(z) # assign boundary condition to neural networks
		elif self.net_type == 'Res':
			u = ((z[:,0])**2 - 1) * ((z[:,1])**2 - 1) * self.fcresnet(z)
		else:
			raise ValueError("net_type is not supported")

		gradients = tf.gradients(u, [z])[0]
		slopes = tf.reduce_sum(tf.square(gradients), reduction_indices=[1])
		# define variational loss for different pde
		if self.pde_type == 'Diffusion':
			variational_loss = tf.reduce_mean(0.5*slopes - u)
		elif self.pde_type == 'Helmholtz':
			variational_loss = tf.reduce_mean(0.5 * slopes - 0.5 * ref_coeff**2 * u**2 - u)
		else:
			raise ValueError("pde type is not supported")

		train_op = tf.train.AdamOptimizer(self.lr_rate).minimize(variational_loss)
		if os.path.exists(self.output_path):
			shutil.rmtree(self.output_path)
		os.mkdir(self.output_path)

		saver = tf.train.Saver()
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			for idx_iter in range(self.num_iters):
				batch_data = self.spatial_range[0] + (self.spatial_range[1]-self.spatial_range[0])*np.random.rand(self.batch_size, 2)
				loss_cur, _ = sess.run([variational_loss, train_op],
												feed_dict={z: batch_data})
				self.variational_loss_history.append(loss_cur)
				print('iteration: {}, variational_loss: {:.4}'. format(idx_iter+1, self.variational_loss_history[-1]))
				saver.save(sess, os.path.join(self.output_path, "model"), global_step=idx_iter)
				decay_flag = (idx_iter + 1) % 10000 == 0
				if decay_flag:
					self.lr_rate *= self.lr_decay

	def test(self, test_input):
		chkpt_fname_final = tf.train.latest_checkpoint(self.output_path)
		saver = tf.train.Saver()
		assert test_input.shape[1] == 2
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			saver.restore(sess, chkpt_fname_final)
			if self.net_type == 'FC':
				u_test = ((test_input[:,0])**2 - 1) * ((test_input[:,1])**2 - 1) * sess.run(self.fcnet(test_input))
			elif self.net_type == 'Res':
				u_test = ((test_input[:,0])**2 - 1) * ((test_input[:,1])**2 - 1) * sess.run(self.fcresnet(test_input))
			else:
				raise ValueError("net_type is not supported")

		return u_test, self.pde_type

