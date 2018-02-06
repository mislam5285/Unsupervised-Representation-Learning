import sys, os, glob
import tensorflow as tf
from libs.activations import lrelu
import math
import numpy as np
import random as rnd

tf.reset_default_graph()

# %%
# define parameters for input sizes and filters for autoencoder layers
batch_size = 1
input_shape = [batch_size, 40, 320, 1]
n_filters = [1, 70, 70, 70]
filter_sizes = [[8, 10], [8, 14], [2, 8]]
strides_i = [[4, 5], [1, 7], [1, 1]]
    
# %%
# input to the network
x = tf.placeholder(tf.float32, input_shape, name='x')

# %%
# ensure input is 4-d.
if len(x.get_shape()) == 4:
    x_tensor = x
else:
    raise ValueError('Unsupported input dimensions')
current_input = x_tensor

# %%
# Build the encoder
encoder = []
bias_list = []
shapes = []
# Zs = []
# beta = 50
# betas = []
for layer_i, n_output in enumerate(n_filters[1:]):
    n_input = current_input.get_shape().as_list()[3]
    shapes.append(current_input.get_shape().as_list())
    W = tf.Variable(
        tf.random_uniform([filter_sizes[layer_i][0], filter_sizes[layer_i][1], n_input, n_output],
            -1.0 / math.sqrt(n_input), 1.0 / math.sqrt(n_input)))
    encoder.append(W)
    b = tf.Variable(tf.zeros([n_output]))
    bias_list.append(b)
    output = lrelu(
        tf.add(tf.nn.conv2d(
            current_input, W, strides=[1, strides_i[layer_i][0], strides_i[layer_i][1], 1], padding='VALID'), b))
#     if layer_i > 2:
#         Zs.append(output)
#         beta *= 5.0
#         betas.append(beta)
    print('layer ', layer_i, 'has output shape =', output.get_shape())
    current_input = output

# %%
# the latent representation
z = current_input
z_shape = z.get_shape()
print('z.shape =', z_shape) #, 'beta =', beta)

# %%
# create head and place cells
new_shape = np.asarray([-1, z_shape[1]*z_shape[2]*z_shape[3]]).astype(int)
z_flat = tf.reshape(z, new_shape)
Head = tf.layers.dense(inputs=z_flat, units=40, activation=lrelu, name='head')
Place = tf.layers.dense(inputs=z_flat, units=30, activation=lrelu, name='place')
# HeadAndPlace = tf.concat([Head, Place], 1)
print('Head shape =', Head.get_shape(), 'Place shape =', Place.get_shape()) #, 'HeadAndPlace shape =', HeadAndPlace.get_shape())
z_head_out = tf.layers.dense(inputs=Head, units=z_shape[1]*z_shape[2]*z_shape[3], activation=lrelu, name='z_head')
z_place_out = tf.layers.dense(inputs=Place, units=z_shape[1]*z_shape[2]*z_shape[3], activation=lrelu, name='z_place')
z_out = z_head_out + z_place_out
new_shape = np.asarray([-1, z_shape[1], z_shape[2], z_shape[3]]).astype(int)
current_input = tf.reshape(z_out, new_shape)
print('z_out shape =', current_input.get_shape())

Head_variables = []
Place_variables = []
for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
#     print(v.name)
    if v.name == Head.name.split('/')[0] + '/kernel:0':
        Head_weights = v
        Head_variables.append(Head_weights)
    if v.name == Head.name.split('/')[0] + '/bias:0':
        Head_bias = v
        Head_variables.append(Head_bias)
    if v.name == Place.name.split('/')[0] + '/kernel:0':
        Place_weights = v
        Place_variables.append(Place_weights)
    if v.name == Place.name.split('/')[0] + '/bias:0':
        Place_bias = v
        Place_variables.append(Place_bias)
    if v.name == z_head_out.name.split('/')[0] + '/kernel:0':
        z_head_out_weights = v
        Head_variables.append(z_head_out_weights)
    if v.name == z_head_out.name.split('/')[0] + '/bias:0':
        z_head_out_bias = v
        Head_variables.append(z_head_out_bias)
    if v.name == z_place_out.name.split('/')[0] + '/kernel:0':
        z_place_out_weights = v
        Place_variables.append(z_place_out_weights)
    if v.name == z_place_out.name.split('/')[0] + '/bias:0':
        z_place_out_bias = v
        Place_variables.append(z_place_out_bias)

encoder.reverse()
shapes.reverse()
strides_i.reverse()

# %%
# Build the decoder using the same weights
for layer_i, shape in enumerate(shapes):
    W = encoder[layer_i]
    b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
    bias_list.append(b)
    output = lrelu(tf.add(
        tf.nn.conv2d_transpose(
            current_input, W,
            tf.stack([x.get_shape()[0], shape[1], shape[2], shape[3]]),
            strides=[1, strides_i[layer_i][0], strides_i[layer_i][1], 1], padding='VALID'), b))
#     if layer_i == 0:
#         Zs.append(output)
#         beta = beta/10.0
#         betas.append(beta)
    print('layer ', layer_i, 'has output shape =', output.get_shape())
    current_input = output

# %%
# the reconstruction through the network
y = current_input
# cost function measures pixel-wise difference
cost1 = tf.reduce_sum(tf.square(y - x_tensor))
cost = cost1

# %%
# define action relevant costs on saved layers
# steps = tf.placeholder(tf.float32, [batch_size-1], name='steps')
# angles = tf.placeholder(tf.float32, [batch_size-1], name='angles')
# cost2s = []
# print('betas =', betas)

# for idx, tmp in enumerate(Zs):
#     print(idx, 'beta =', betas[idx], 'tmp.shape =', tmp.shape)
#     Head, Place = tf.split(tmp, [40, 30], 3)
#     Head1 = Head[0:batch_size-1, :, :, :]
#     Head2 = Head[1:batch_size, :, :, :]
#     Place1 = Place[0:batch_size-1, :, :, :]
#     Place2 = Place[1:batch_size, :, :, :]
#     cost2 = tf.reduce_sum((steps != 0)*tf.reduce_sum(tf.pow((Head1-Head2), 2), [1, 2, 3]))\
#         + tf.reduce_sum((angles != 0)*tf.reduce_sum(tf.pow((Place1-Place2), 2), [1, 2, 3]))
#     cost2s.append(cost2)
#     cost += betas[idx]*cost2

# %%
# dict for inputs, outputs, and costs
ae = {'x': x,\
      'z': z, 'Head': Head, 'Place': Place, 'y': y,\
      'cost': cost, 'cost1': cost1}

# %%
# define the optimizer
# Head_weights, Head_bias, z_head_out_weights, z_head_out_bias
# Place_weights, Place_bias, z_place_out_weights, z_place_out_bias
learning_rate = 0.001
trainable_variable_list1 = encoder + bias_list + Head_variables
optimizer1 = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'], var_list = trainable_variable_list1)

trainable_variable_list2 = encoder + bias_list + Place_variables
optimizer2 = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'], var_list = trainable_variable_list2)

optimizer3 = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])

#########################################################################################
# %%
# load training image
train_images = np.load('train_images.npy')
print(type(train_images), train_images.shape)
mean_img = np.mean(train_images, axis=0)
print('mean_img.shape =', mean_img.shape)
# plt.figure()
# plt.imshow(mean_img[:, :, 0], cmap='gray')

# load training cmd
train_cmds = np.load("train_cmds.npy")
print(type(train_cmds), train_cmds.shape)
train_steps = train_cmds[:, 0]
train_angles = train_cmds[:, 1]
print(train_steps.shape, train_angles.shape)

# load training image2
train_images2 = np.load('train_images2.npy')
print(type(train_images2), train_images2.shape)
mean_img2 = np.mean(train_images2, axis=0)
print('mean_img2.shape =', mean_img2.shape)
# plt.figure()
# plt.imshow(mean_img2[:, :, 0], cmap='gray')

# load training cmd2
train_cmds2 = np.load("train_cmds2.npy")
print(type(train_cmds2), train_cmds2.shape)
train_steps2 = train_cmds2[:, 0]
train_angles2 = train_cmds2[:, 1]
print(train_steps2.shape, train_angles2.shape)

#################################################################################################
# %%
# initialize the weights or load the trained weights
sess = tf.Session()
saved_model = 'modelxxx'

sess.run(tf.global_variables_initializer())

# restorer = tf.train.Saver()
# restorer.restore(sess, 'model/' + saved_model)

##################################################################################################
# %%
# Fit training data
n_epochs = 60
for epoch_i in range(n_epochs):
    for batch_i in range(train_images.shape[0]):
        batch_xs = train_images[batch_i:(batch_i+batch_size), :, :, :]
        train = np.array([img - mean_img for img in batch_xs])
        if epoch_i != 0:
            steps_train = train_steps[batch_i-1]
            angles_train = train_angles[batch_i-1]
        else:
            steps_train = 0
            angles_train = 0
        if steps_train != 0:
            sess.run(optimizer2, feed_dict={ae['x']: train})
        elif angles_train != 0:
            sess.run(optimizer1, feed_dict={ae['x']: train})
        else:
            sess.run(optimizer3, feed_dict={ae['x']: train})
        if batch_i % 4000 == 0:
            cost, head_val, place_val = sess.run([ae['cost'], ae['Head'], ae['Place']], feed_dict={ae['x']: train})
            print(epoch_i, batch_i, 'cost =', cost, '\n\tHead:', head_val, '\n\tPlace:', place_val)
        #     print('z.shape =', z.shape, '\n', np.reshape(z, (batch_size, 70)))
    for batch_i in range(train_images.shape[0]):
        batch_xs = train_images2[batch_i:(batch_i+batch_size), :, :, :]
        train = np.array([img - mean_img2 for img in batch_xs])
        if epoch_i != 0:
            steps_train = train_steps2[batch_i-1]
            angles_train = train_angles2[batch_i-1]
        else:
            steps_train = 0
            angles_train = 0
        if steps_train != 0:
            sess.run(optimizer2, feed_dict={ae['x']: train})
        elif angles_train != 0:
            sess.run(optimizer1, feed_dict={ae['x']: train})
        else:
            sess.run(optimizer3, feed_dict={ae['x']: train})
        if batch_i % 4000 == 0:
            cost, head_val, place_val = sess.run([ae['cost'], ae['Head'], ae['Place']], feed_dict={ae['x']: train})
            print(epoch_i, batch_i, 'cost =', cost, '\n\tHead:', head_val, '\n\tPlace:', place_val)
        #     print('z.shape =', z.shape, '\n', np.reshape(z, (batch_size, 70)))

saver = tf.train.Saver()
save_path = saver.save(sess, 'model/' + saved_model)
print("Model saved in file: %s" % save_path)