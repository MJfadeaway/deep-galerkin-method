from __future__ import division
import numpy as np
import tensorflow as tf
import os
import shutil

# Author: Kejun Tang
# Last Revised: Dec 07, 2018

class VarNets():
	"""VarNets for two dimensional PDEs"""

	"""
	variational principle for solving PDEs, including Diffusion equation, Helmholtz equation, and Stokes equation

	Parameters:
	-----------
	spatial_range: spatial domain (square domain), [-1,1] or [0,1], list
	pde_type: which PDE, Diffusion, Helmholtz or Stokes, string
	net_type: which networks, fully connected (FC) or residual nets (Res), string
	num_hidden: hidden units in each layer, int
	num_blocks: the number of residual blocks for residual nets, int
	batch_size: batch size, int
	num_iters: the number of iterations, int
	lr_rate: step size (learning rate) for stochastic optimization, float
	lr_decay: decay of step size (learning rate) after each 10000 iterations, float
	output_path: save path for tensorflow model, string
	output_shape: if Stokes, output_shape is 2, and output_shape 1 otherwise, int
	"""

	def __init__(self, spatial_range, pde_type, net_type, num_hidden=256, num_blocks=10, pnum_blocks=5, lam=0.001, batch_size=200, num_iters=100000, lr_rate=1e-3, lr_decay=1.0, output_path='VarNets'):
		self.spatial_range = spatial_range # for example, spatial range = [-1,1]
		self.pde_type = pde_type # pde_type: Diffusion, Stokes, Helmholtz
		self.net_type = net_type # FC or Res
		self.num_hidden = num_hidden
		self.num_blocks = num_blocks
		self.pnum_blocks = pnum_blocks
		self.lam = lam
		self.batch_size = batch_size
		self.num_iters = num_iters
		self.lr_rate = lr_rate
		self.lr_decay = lr_decay
		self.output_path = output_path
		if pde_type == 'Stokes':
			self.output_shape = 2
			self.velocity_varloss_history = []
			self.pressure_varloss_history = []
			#self.boundary_loss_history = []
		else:
			self.output_shape = 1
			self.variational_loss_history = []

	def fcnet(self, x_input):
		"""fcnet for all pde_type"""
		num_hidden = self.num_hidden
		with tf.variable_scope("fcnet", reuse=tf.AUTO_REUSE):
				h1 = tf.nn.relu(tf.layers.dense(x_input, num_hidden))
				h2 = tf.nn.relu(tf.layers.dense(h1, num_hidden))
				h3 = tf.nn.relu(tf.layers.dense(h2, num_hidden))
				h4 = tf.nn.relu(tf.layers.dense(h3, num_hidden))
				h5 = tf.nn.relu(tf.layers.dense(h4, num_hidden))
				h6 = tf.nn.relu(tf.layers.dense(h5, num_hidden))
				output = tf.layers.dense(h6, self.output_shape)
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

		with tf.variable_scope("output", reuse=tf.AUTO_REUSE):
			response = tf.layers.dense(response, self.output_shape)
		return response

	def fcnet_pressure(self, x_input):
		num_hidden = self.num_hidden
		with tf.variable_scope("pressure", reuse=tf.AUTO_REUSE):
				h1 = tf.nn.relu(tf.layers.dense(x_input, num_hidden))
				h2 = tf.nn.relu(tf.layers.dense(h1, num_hidden))
				h3 = tf.nn.relu(tf.layers.dense(h2, num_hidden))
				h4 = tf.nn.relu(tf.layers.dense(h3, num_hidden))
				h5 = tf.nn.relu(tf.layers.dense(h4, num_hidden))
				h6 = tf.nn.relu(tf.layers.dense(h5, num_hidden))
				output = tf.layers.dense(h6, 1)
		return output

	def fcresnet_pressure(self, x_input):
		"""Resnet for pressure in Stokes equation"""
		num_blocks = self.pnum_blocks
		response = x_input
		for idx_block in range(self.pnum_blocks):
			with tf.variable_scope("pres_unit_%d" % idx_block, reuse=tf.AUTO_REUSE):
				response = self.residual_block(response)

		with tf.variable_scope("pres_output", reuse=tf.AUTO_REUSE):
			response = tf.layers.dense(response, 1)
		return response

	def velocity_net(self, x_input):
		"""for Stokes equation, using neural networks to approximate velocity"""
		with tf.variable_scope("velocity", reuse=tf.AUTO_REUSE):
			response = self.fcresnet(x_input)
		return response

	def pressure_net(self, x_input):
		"""for Stokes equation, using neural networks to approximate pressure"""
		with tf.variable_scope("pressure", reuse=tf.AUTO_REUSE):
			response = self.fcresnet_pressure(x_input)
		return responseraise
	"""
	def divergence(self, x_input):
	
		x_input:
		--------
		batch spatial variables, tf.tensor type data

		y_output:
		---------
		batch velocity data, tf.tensor type data

		returns:
		--------
		numerical estimation of divergence at x_input position
		
		if self.net_type == 'Res':
			y_output = self.velocity_net(x_input)
		elif self.net_type == 'FC':
			y_output = self.fcnet(x_input)
		else:
			raise ValueError("net_type is not supported")
		
		num_points, num_variables = x_input.get_shape().as_list()
		y_output_x = y_output[:, 0]
		y_output_y = y_output[:, 1]
		div_vel = np.zeros((num_points, 1))
		with tf.Session() as sess:
			for idx_numpts in range(num_points):
				v1_grad_idxnumpts = tf.gradients(y_output_x[idx_numpts], [x_input])[0]
				v2_grad_idxnumpts = tf.gradients(y_output_y[idx_numpts], [x_input])[0]
				div_idxnumpts = v1_grad_idxnumpts[idx_numpts,0] + v2_grad_idxnumpts[idx_numpts,1]
				div_vel[idx_numpts] = sess.run(div_idxnumpts)
			

		return div_vel
	"""

	def train(self):
		"""train step"""

		z = tf.placeholder(tf.float32, shape=[None, 2]) # two dimensional (the number of spatial variables) PDE

		if self.pde_type != 'Stokes':
			if self.net_type == 'FC':
				# assign boundary condition to neural networks
				u = ((tf.reshape(z[:,0], [-1, 1]))**2 - tf.convert_to_tensor(1., tf.float32)) * ((tf.reshape(z[:,1], [-1, 1]))**2 - tf.convert_to_tensor(1., tf.float32)) * self.fcnet(z) 
			elif self.net_type == 'Res':
				u = ((tf.reshape(z[:,0], [-1, 1]))**2 - tf.convert_to_tensor(1., tf.float32)) * ((tf.reshape(z[:,1], [-1, 1]))**2 - tf.convert_to_tensor(1., tf.float32)) * self.fcresnet(z)
			else:
				raise ValueError("net_type is not supported")

			gradients = tf.gradients(u, [z])[0]
			slopes = tf.reduce_sum(tf.square(gradients), reduction_indices=[1])
		else: # Stokes equation
			if self.net_type == 'FC':
				velocity = (tf.reshape(z[:,0], [-1, 1])) * ((tf.reshape(z[:,0], [-1, 1])) - tf.convert_to_tensor(1., tf.float32)) * (tf.reshape(z[:,1], [-1, 1])) * ((tf.reshape(z[:,1], [-1, 1])) - tf.convert_to_tensor(1., tf.float32)) * self.fcnet(z)

			elif self.net_type == 'Res':
				velocity = (tf.reshape(z[:,0], [-1, 1])) * ((tf.reshape(z[:,0], [-1, 1])) - tf.convert_to_tensor(1., tf.float32)) * (tf.reshape(z[:,1], [-1, 1])) * ((tf.reshape(z[:,1], [-1, 1])) - tf.convert_to_tensor(1., tf.float32)) * self.velocity_net(z)

			else:
				raise ValueError("net_type is not supported")


		# define variational loss for different pde
		if self.pde_type == 'Diffusion':
			variational_loss = tf.reduce_mean(0.5*slopes - u)
		elif self.pde_type == 'Stokes':
			v1_grad = tf.gradients(velocity[:, 0], [z])[0]
			v2_grad = tf.gradients(velocity[:, 1], [z])[0]
			v1_slopes = tf.reduce_sum(tf.square(v1_grad), reduction_indices=[1])
			v2_slopes = tf.reduce_sum(tf.square(v2_grad), reduction_indices=[1])
			div_velocity = tf.reshape(v1_grad[:,0], [-1,1]) + tf.reshape(v2_grad[:,1], [-1,1])

			# source function f = [0, g]
			# construct variational loss for saddle point, Stokes equation
			fv_dotprod = tf.reshape(velocity[:,1], [-1,1]) * tf.convert_to_tensor(9.8, tf.float32)
			velocity_varloss = tf.reduce_mean(0.5*(v1_slopes+v2_slopes)) - tf.reduce_mean((self.lam)**(-2)*(div_velocity)**2) - tf.reduce_mean(fv_dotprod) # minimize velocity
			pressure = (self.lam)**(-2) * div_velocity
			pressure_varloss = tf.reduce_mean((pressure)**2)
			velocity_varloss += pressure_varloss
			# extract training variables
			if self.net_type == 'FC':
				velocity_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'fcnet')
			elif self.net_type == 'Res':
				velocity_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'velocity')
			else:
				raise ValueError("net_type is not supported")

		elif self.pde_type == 'Helmholtz':
			variational_loss = tf.reduce_mean(0.5 * slopes - 0.5 * ref_coeff**2 * u**2 - u)
		else:
			raise ValueError("pde type is not supported")

		if self.pde_type != "Stokes":
			train_op = tf.train.AdamOptimizer(self.lr_rate).minimize(variational_loss)
		else: # Stokes equation
			velocity_train_op = tf.train.AdamOptimizer(self.lr_rate).minimize(velocity_varloss, var_list=velocity_vars)

		if os.path.exists(self.output_path):
			shutil.rmtree(self.output_path)
		os.mkdir(self.output_path)

		saver = tf.train.Saver()
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			if self.pde_type != 'Stokes':
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
			else: # Stokes equation
				for idx_iter in range(self.num_iters):
					batch_data = self.spatial_range[0] + (self.spatial_range[1]-self.spatial_range[0])*np.random.rand(self.batch_size, 2)
					velocity_loss_cur, _ = sess.run([velocity_varloss, velocity_train_op], feed_dict={z: batch_data})
					pressure_loss_cur = sess.run(pressure_varloss, feed_dict={z: batch_data})
					self.velocity_varloss_history.append(velocity_loss_cur)
					self.pressure_varloss_history.append(pressure_loss_cur)
					print('iteration: {}, v_varloss: {:.4}, p_varloss: {:.4}'. format(idx_iter+1, self.velocity_varloss_history[-1], self.pressure_varloss_history[-1]))
					saver.save(sess, os.path.join(self.output_path, "model"), global_step=idx_iter)
					decay_flag = (idx_iter + 1) % 10000 == 0
					lam_decay_flag = (idx_iter + 1) % 1000 == 0
					if lam_decay_flag:
						self.lam *= 0.95
					if decay_flag:
						self.lr_rate *= self.lr_decay

	def test(self, test_input):
		chkpt_fname_final = tf.train.latest_checkpoint(self.output_path)
		saver = tf.train.Saver()
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			saver.restore(sess, chkpt_fname_final)
			if self.pde_type != 'Stokes':
				assert test_input.shape[1] == 2
				if self.net_type == 'FC':
					u_test = sess.run(self.fcnet(test_input))
				elif self.net_type == 'Res':
					u_test = sess.run(self.fcresnet(test_input))
				else:
					raise ValueError("net_type is not supported")
				u_test = sess.run(((tf.reshape(test_input[:,0], [-1, 1]))**2 - tf.convert_to_tensor(1., tf.float32)) * ((tf.reshape(test_input[:,1], [-1,1]))**2 - tf.convert_to_tensor(1., tf.float32))) * u_test

			else: # Stokes equation
				test_inputv, test_inputp = test_input[0], test_input[1]
				assert test_inputv.shape[1] == 2 and test_inputp.shape[1] == 2
				if self.net_type == 'FC':
					u_test = sess.run(self.fcnet(test_inputv))
					u_for_p = self.fcnet(test_inputp)
					#v1_grad = tf.gradients(u_for_p[:, 0], [test_inputp])[0]
					#v2_grad = tf.gradients(u_for_p[:, 1], [test_inputp])[0]
					num_points, num_variables = test_inputp.get_shape().as_list()
					u_forp_x, u_forp_y  = u_for_p[:, 0], u_for_p[:, 1]
					div_velocity = np.zeros((num_points, 1))
					for idx_numpts in range(num_points):
						v1_grad_idxnumpts = tf.gradients(u_forp_x[idx_numpts], [test_inputp])[0]
						v2_grad_idxnumpts = tf.gradients(u_forp_y[idx_numpts], [test_inputp])[0]
						div_velocity[idx_numpts] = sess.run(v1_grad_idxnumpts[idx_numpts,0] + v2_grad_idxnumpts[idx_numpts,1])
	
					p_test = (self.lam)**(-2) * div_velocity
				elif self.net_type == 'Res':
					u_test = sess.run(self.velocity_net(test_inputv))
					u_for_p = self.velocity_net(test_inputp)
					#print('-----u_for_p shape-----', u_for_p.shape)
					#print('-----test input p-----', test_inputp.shape)
					#v1_grad = tf.gradients(u_for_p[:, 0], [test_inputp])
					#v2_grad = tf.gradients(u_for_p[:, 1], [test_inputp])
					#print('-----v1_grad-----', v1_grad)
					#print('-----v2_grad-----', v2_grad)
					num_points, num_variables = test_inputp.get_shape().as_list()
					u_forp_x, u_forp_y  = u_for_p[:, 0], u_for_p[:, 1]
					div_velocity = np.zeros((num_points, 1))
					for idx_numpts in range(num_points):
						v1_grad_idxnumpts = tf.gradients(u_forp_x[idx_numpts], [test_inputp])[0]
						v2_grad_idxnumpts = tf.gradients(u_forp_y[idx_numpts], [test_inputp])[0]
						print('-----idx numpts-----', idx_numpts)
						div_velocity[idx_numpts] = sess.run(v1_grad_idxnumpts[idx_numpts,0] + v2_grad_idxnumpts[idx_numpts,1])
						print('-----idx numpts done-----', idx_numpts)

					p_test = (self.lam)**(-2) * div_velocity
				else:
					raise ValueError("net_type is not supported")
				#u_test = np.reshape(u_test, (-1,1), order='F')
				u_test = sess.run((tf.reshape(test_inputv[:,0], [-1, 1])) * ((tf.reshape(test_inputv[:,0], [-1, 1])) - tf.convert_to_tensor(1., tf.float32)) * (tf.reshape(test_inputv[:,1], [-1, 1])) * ((tf.reshape(test_inputv[:,1], [-1, 1])) - tf.convert_to_tensor(1., tf.float32))) * u_test
				u_test = np.reshape(u_test, (-1,1), order='F')
#				print('-----u shape-----', u_test.shape)
#				print('-----p shape-----', p_test.shape)
				u_test = np.concatenate((u_test, p_test), axis=0)
#				print('----- solution vector shape-----', u_test.shape)

		return u_test, self.pde_type

