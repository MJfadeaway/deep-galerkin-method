from __future__ import division
import numpy as np
import tensorflow as tf
import tensorflow.contrib.opt
import os
import shutil

# Author: Ke Li, Kejun Tang
# Last Revised: April 12, 2019

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

    def __init__(self, spatial_range, pde_type, net_type, num_hidden=256, num_blocks=10, batch_size=200, num_iters=100000, lr_rate=1e-3, lr_decay=1.0, output_path='VarNets'):
        self.spatial_range = spatial_range # for example, spatial range = [-1,1]
        self.pde_type = pde_type # pde_type: Diffusion, Stokes, Helmholtz
        self.net_type = net_type # FC or Res
        self.num_hidden = num_hidden
        self.num_blocks = num_blocks
        self.batch_size = batch_size
        self.num_iters = num_iters
        self.lr_rate = lr_rate
        self.lr_decay = lr_decay
        self.output_path = output_path
        if pde_type == 'Stokes':
            self.output_shape = 2
            self.velocity_varloss_history = []
            self.pressure_varloss_history = []
        else:
            self.output_shape = 3
            self.variational_loss_history = []

    def fcnet(self, x_input):
        """fcnet for all pde_type"""
        num_hidden = self.num_hidden
        with tf.variable_scope("fcnet", reuse=tf.AUTO_REUSE):
                h1 = tf.nn.tanh(tf.layers.dense(x_input, num_hidden))
                h2 = tf.nn.tanh(tf.layers.dense(h1, num_hidden))
                h3 = tf.nn.tanh(tf.layers.dense(h2, num_hidden))
                h4 = tf.nn.tanh(tf.layers.dense(h3, num_hidden))
                h5 = tf.nn.tanh(tf.layers.dense(h4, num_hidden))
                h6 = tf.nn.tanh(tf.layers.dense(h5, num_hidden))
                h7 = tf.nn.tanh(tf.layers.dense(h6, num_hidden))
                h8 = tf.nn.tanh(tf.layers.dense(h7, num_hidden))
                # h1 = tf.nn.relu(tf.layers.dense(x_input, num_hidden))
                # h2 = tf.nn.relu(tf.layers.dense(h1, num_hidden))
                # h3 = tf.nn.relu(tf.layers.dense(h2, num_hidden))
                # h4 = tf.nn.relu(tf.layers.dense(h3, num_hidden))
                # h5 = tf.nn.relu(tf.layers.dense(h4, num_hidden))
                # h6 = tf.nn.relu(tf.layers.dense(h5, num_hidden))
                # h7 = tf.nn.relu(tf.layers.dense(h6, num_hidden))
                # h8 = tf.nn.relu(tf.layers.dense(h7, num_hidden))
                output = tf.layers.dense(h8, self.output_shape)
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


    def train(self):
        """train step"""

        z = tf.placeholder(tf.float32, shape=[None, 2]) # two dimensional (the number of spatial variables) PDE
        zbc_top = tf.placeholder(tf.float32, shape=[None, 2]) # for top boundary
        zbc_bottom = tf.placeholder(tf.float32, shape=[None, 2]) # for bottom boundary
        zbc_left = tf.placeholder(tf.float32, shape=[None, 2]) # for left boundary
        zbc_right = tf.placeholder(tf.float32, shape=[None, 2]) # for right boundary
        beta = tf.placeholder(tf.float32, shape=[None, 1])
        b = 10*np.ones((1,1))

        if self.pde_type != 'Stokes':
            if self.net_type == 'FC':
                # assign boundary condition to neural networks
                U = self.fcnet(z)
                u = U[:,0]
                ut = self.fcnet(zbc_top)[:,0]
                ub = self.fcnet(zbc_bottom)[:,0]
                ul = self.fcnet(zbc_left)[:,0]
                ur = self.fcnet(zbc_right)[:,0]
            elif self.net_type == 'Res':
                U = self.fcresnet(z)
                u = U[:,0]
                ut = self.fcresnet(zbc_top)[:,0]
                ub = self.fcresnet(zbc_bottom)[:,0]
                ul = self.fcresnet(zbc_left)[:,0]
                ur = self.fcresnet(zbc_right)[:,0]
            else:
                raise ValueError("net_type is not supported")

            gradients = tf.gradients(u, [z])[0]
            if self.net_type == 'FC':
                tau = U[:,1:3]
            elif self.net_type == 'Res':
                tau = U[:,1:3]
            grad_tau_ver = tf.gradients(tau[:, 0], [z])[0][:,0]
            grad_tau_hor = tf.gradients(tau[:, 1], [z])[0][:,1]
            slopes = tf.reduce_sum(tf.square(gradients), reduction_indices=[1])
        


        # define variational loss for different pde
        if self.pde_type == 'Diffusion':
            #variational_loss = tf.reduce_mean(0.5*slopes - u)
            variational_loss = tf.reduce_mean(tf.square(gradients + tau)) + tf.reduce_mean(tf.square(grad_tau_ver + grad_tau_hor - tf.convert_to_tensor([[1.]], tf.float32)))
            top_loss = tf.reduce_mean(tf.square(ut-tf.convert_to_tensor([[0.]], tf.float32)))
            bottom_loss = tf.reduce_mean(tf.square(ub-tf.convert_to_tensor([[0.]], tf.float32)))
            left_loss = tf.reduce_mean(tf.square(ul-tf.convert_to_tensor([[0.]], tf.float32)))
            right_loss = tf.reduce_mean(tf.square(ur-tf.convert_to_tensor([[0.]], tf.float32)))
            boundary_loss = top_loss+bottom_loss+left_loss+right_loss
            total_loss = variational_loss + beta*boundary_loss
            #total_loss1 = variational_loss + 10*boundary_loss
        elif self.pde_type == 'Stokes':
            v1_grad = tf.gradients(velocity[:, 0], [z])[0]
            v2_grad = tf.gradients(velocity[:, 1], [z])[0]
            v1_slopes = tf.reduce_sum(tf.square(v1_grad), reduction_indices=[1])
            v2_slopes = tf.reduce_sum(tf.square(v2_grad), reduction_indices=[1])
            div_velocity = tf.reshape(v1_grad[:,0], [-1,1]) + tf.reshape(v2_grad[:,1], [-1,1])

            # source function here is zero
            # construct variational loss for saddle point, Stokes equation
            velocity_varloss = tf.reduce_mean(0.5*(v1_slopes+v2_slopes)) - tf.reduce_mean(pressure*div_velocity) # minimize velocity
            pressure_varloss = tf.reduce_mean(pressure*div_velocity) # maximum pressure

            # boundary penalty loss term, here is driven leaky cavity flow, u=[1,0]^T on the top line x2 = 1, and u = [0,0]^T on 
            # all other boundaries, where x = [x1, x2]^T and the spatial domain D = (0,1) * (0,1)
            top_loss = tf.reduce_mean(tf.square(velocity_top-tf.convert_to_tensor([[1., 0.]], tf.float32)))
            bottom_loss = tf.reduce_mean(tf.square(velocity_bottom-tf.convert_to_tensor([[0., 0.]], tf.float32)))
            left_loss = tf.reduce_mean(tf.square(velocity_left-tf.convert_to_tensor([[0., 0.]], tf.float32)))
            right_loss = tf.reduce_mean(tf.square(velocity_right-tf.convert_to_tensor([[0., 0.]], tf.float32)))
            boundary_loss = top_loss + bottom_loss + left_loss + right_loss

            velocity_varloss += boundary_loss



        if self.pde_type != "Stokes":
            #train_op1 = tf.train.AdamOptimizer(self.lr_rate).minimize(total_loss1)
            #train_op = tf.train.AdamOptimizer(self.lr_rate).minimize(total_loss)
            self.Opt = tf.contrib.opt.ScipyOptimizerInterface(total_loss, method="L-BFGS-B", options={'maxiter': 50})
            

        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
        os.mkdir(self.output_path)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if self.pde_type != 'Stokes':
                for idx_iter in range(self.num_iters):
                    batch_data = self.spatial_range[0]+0.0625 + (self.spatial_range[1]-self.spatial_range[0]-0.125)*np.random.rand(32*self.batch_size, 2)
                    top_data = np.hstack((self.spatial_range[0] + (self.spatial_range[1]-self.spatial_range[0])*np.random.rand(self.batch_size, 1),np.ones((self.batch_size, 1))))
                    bottom_data = np.hstack((self.spatial_range[0] + (self.spatial_range[1]-self.spatial_range[0])*np.random.rand(self.batch_size, 1),-1*np.ones((self.batch_size, 1))))
                    left_data = np.hstack((self.spatial_range[0] * np.ones((self.batch_size,1)),-1+2*np.random.rand(self.batch_size, 1)))
                    right_data = np.hstack((self.spatial_range[1] * np.ones((self.batch_size,1)),-1+2*np.random.rand(self.batch_size, 1)))
                    self.Opt.minimize(sess, feed_dict={z: batch_data,zbc_top:top_data,zbc_bottom:bottom_data,zbc_left:left_data,zbc_right:right_data,beta:b})
                    #loss_cur, _ = sess.run(self.Opt,
                                                    #feed_dict={z: batch_data,zbc_top:top_data,zbc_bottom:bottom_data,zbc_left:left_data,zbc_right:right_data})
                    #loss_cur, _ = sess.run([total_loss, train_op],
                                                    #feed_dict={z: batch_data,zbc_top:top_data,zbc_bottom:bottom_data,zbc_left:left_data,zbc_right:right_data,beta:b})
                    #self.variational_loss_history.append(loss_cur)
                    #if idx_iter > 0.8*self.num_iters:
                    #    b = 500*np.ones((1,1))
                    #print('iteration: {}, variational_loss: {:.4}'. format(idx_iter+1, self.variational_loss_history[-1]))
                    #print('iteration:', idx_iter+1)
                    print(idx_iter+1)
                    saver.save(sess, os.path.join(self.output_path, "model"), global_step=idx_iter)
                    decay_flag = (idx_iter + 1) % 10000 == 0
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
                    u_test = sess.run(self.fcnet(test_input))[:,0]
                elif self.net_type == 'Res':
                    u_test = sess.run(self.fcresnet(test_input))[:,0]
                else:
                    raise ValueError("net_type is not supported")
                #u_test = sess.run(((tf.reshape(test_input[:,0], [-1, 1]))**2 - tf.convert_to_tensor(1., tf.float32)) * ((tf.reshape(test_input[:,1], [-1,1]))**2 - tf.convert_to_tensor(1., tf.float32))) * u_test
                u_test = u_test.reshape(-1,1)
                print(u_test.shape)
        return u_test, self.pde_type