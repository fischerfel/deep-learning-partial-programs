import tensorflow as tf
#import matplotlib.pyplot as plt
import numpy as np
import datetime
from sklearn.metrics import roc_auc_score


def graph_embed(X, msg_mask, N_x, N_embed, N_o, iter_level, Wnode, Wembed, W_output, b_output):
	#X  --  affine(W1)  --  ReLU  --  (Message  --  affine(W2)  --  add (with aff W1) --  ReLU  --  )MessageAll  --  output
	node_val = tf.reshape(tf.matmul( tf.reshape(X, [-1, N_x]) , Wnode), [tf.shape(X)[0], -1, N_embed])
	
	cur_msg = tf.nn.relu(node_val)   #[batch, node_num, embed_dim]
	for t in range(iter_level):
		#Message convey
		Li_t = tf.matmul(msg_mask, cur_msg)  #[batch, node_num, embed_dim]
		#Complex Function
		cur_info = tf.reshape(Li_t, [-1, N_embed])
		for Wi in Wembed:
			if (Wi == Wembed[-1]):
				cur_info = tf.matmul(cur_info, Wi)
			else:
				cur_info = tf.nn.relu(tf.matmul(cur_info, Wi))
		neigh_val_t = tf.reshape(cur_info, tf.shape(Li_t))
		#Adding
		tot_val_t = node_val + neigh_val_t
		#Nonlinearity
		tot_msg_t = tf.nn.tanh(tot_val_t)
		cur_msg = tot_msg_t   #[batch, node_num, embed_dim]


	g_embed = tf.reduce_sum(cur_msg, 1)   #[batch, embed_dim]
	output = tf.matmul(g_embed, W_output) + b_output
	
	return output

def vert_embed(X, msg_mask, N_x, N_embed, N_o, iter_level, Wnode, Wembed, W_output, b_output):
	#X  --  affine(W1)  --  ReLU  --  (Message  --  affine(W2)  --  add (with aff W1) --  ReLU  --  )MessageAll  --  output
	node_val = tf.reshape(tf.matmul( tf.reshape(X, [-1, N_x]) , Wnode), [tf.shape(X)[0], -1, N_embed])
	
	cur_msg = tf.nn.relu(node_val)   #[batch, node_num, embed_dim]
	for t in range(iter_level):
		#Message convey
		Li_t = tf.matmul(msg_mask, cur_msg)  #[batch, node_num, embed_dim]
		#Complex Function
		cur_info = tf.reshape(Li_t, [-1, N_embed])
		for Wi in Wembed:
			if (Wi == Wembed[-1]):
				cur_info = tf.matmul(cur_info, Wi)
			else:
				cur_info = tf.nn.relu(tf.matmul(cur_info, Wi))
		neigh_val_t = tf.reshape(cur_info, tf.shape(Li_t))
		#Adding
		tot_val_t = node_val + neigh_val_t
		#Nonlinearity
		tot_msg_t = tf.nn.tanh(tot_val_t)
		cur_msg = tot_msg_t   #[batch, node_num, embed_dim]


	v_embed = tf.matmul(tf.reshape(cur_msg, [-1, N_embed]), W_output) #[node_num * batch, embed_dim]
	output = tf.reshape(v_embed, tf.shape(cur_msg)) #[batch, node_num, embed_dim]
	
	return output

class graphnn(object):
	def __init__(self,
					N_x,
					Dtype, 
					N_embed,
					depth_embed,
					N_o,
					ITERATION_LEVEL,
					lr,
					device = '/cpu:0',
					assigned_vec = None
				):
		self.NODE_LABEL_DIM = N_x

		tf.reset_default_graph()
		tf.logging.set_verbosity(tf.logging.DEBUG)
		#with tf.device(device):
		Wnode = tf.Variable(tf.truncated_normal(shape = [N_x, N_embed], stddev = 0.1, dtype = Dtype))
		Wembed = []
		for i in range(depth_embed):
			Wembed.append(tf.Variable(tf.truncated_normal(shape = [N_embed, N_embed], stddev = 0.1, dtype = Dtype)))

		W_output = tf.Variable(tf.truncated_normal(shape = [N_embed, N_o], stddev = 0.1, dtype = Dtype))
		b_output = tf.Variable(tf.constant(0, shape = [N_o], dtype = Dtype))
		
		X1 = tf.placeholder(Dtype, [None, None, N_x])   #[batch, node_num, feature_dim]
		msg1_mask = tf.placeholder(Dtype, [None, None, None])   #[batch, node_num, node_num]
		self.X1 = X1
		self.msg1_mask = msg1_mask
		embed1 = graph_embed(X1, msg1_mask, N_x, N_embed, N_o, ITERATION_LEVEL, Wnode, Wembed, W_output, b_output)  #[batch, output_dim]
		v_embed1 = vert_embed(X1, msg1_mask, N_x, N_embed, N_o, ITERATION_LEVEL, Wnode, Wembed, W_output, b_output)  #[batch, output_dim]
		if assigned_vec is None:
			X2 = tf.placeholder(Dtype, [None, None, N_x])   #[batch, node_num, feature_dim]
			msg2_mask = tf.placeholder(Dtype, [None, None, None])   #[batch, node_num, node_num]
			self.X2 = X2
			self.msg2_mask = msg2_mask
			embed2 = graph_embed(X2, msg2_mask, N_x, N_embed, N_o, ITERATION_LEVEL, Wnode, Wembed, W_output, b_output)  #[batch, output_dim]
			v_embed2 = vert_embed(X2, msg2_mask, N_x, N_embed, N_o, ITERATION_LEVEL, Wnode, Wembed, W_output, b_output)  #[batch, output_dim]
		else:
			embed2 = tf.constant(assigned_vec)
			v_embed2 = tf.constant(assigned_vec)

		label = tf.placeholder(Dtype, [None, ])   #[batch]  1 if same and -1 if different
		self.label = label
		self.embed1 = embed1
		self.v_embed1 = v_embed1
		
		if assigned_vec is None:
			cos = tf.reduce_sum( embed1*embed2, 1 ) / tf.sqrt( tf.reduce_sum( embed1**2, 1) * tf.reduce_sum( embed2**2, 1) + 1e-10)	
			dot = tf.matmul( v_embed1, v_embed2, transpose_b=True)
			magn1 = tf.reduce_sum( tf.square( v_embed1 ), 2, keep_dims=True)
			magn2 = tf.reduce_sum( tf.square( v_embed2 ), 2, keep_dims=True)
			cos_v = dot / tf.sqrt( tf.matmul( magn1, tf.transpose( magn2, perm=[0, 2, 1])) + 1e-10) # [batch, node_num1, node_num2]
		else:
			cos = tf.reduce_sum( embed1*embed2, 1 ) / tf.sqrt( tf.reduce_sum( embed1**2, 1) + 1e-10)
			cos_v = cos # FIXME 

		max1 = tf.reduce_max( cos_v, axis=2 ) # [batch, node_num1] max per row
		max2 = tf.reduce_max( tf.transpose( cos_v, perm=[0, 2, 1]), axis=2) # [batch, node_num2] max per column
		#tf.Print(max1, tf.shape(max1))
		#tf.Print(max2, tf.shape(max2))
		#tf.Print(max1, max1)
		#tf.Print(max2, max2)
		
		# FIXME deletes valid values, e.g. cos(x,y) = 0 for x, y != 0, and what if 0 is a valid embedding?
		non_zero1 = tf.cast(tf.count_nonzero(max1, axis=1), Dtype)  # [batch] count non-zero
		non_zero2 = tf.cast(tf.count_nonzero(max2, axis=1), Dtype)  # [batch] count non-zero
		non_zero1 = tf.add( non_zero1, tf.fill( tf.shape(non_zero1), 1e-10) )
		non_zero2 = tf.add( non_zero2, tf.fill( tf.shape(non_zero2), 1e-10) )
		#tf.Print(non_zero1, tf.shape(non_zero1))
		#tf.Print(non_zero2, tf.shape(non_zero2))
		#tf.Print(non_zero1, non_zero1)
		#tf.Print(non_zero2, non_zero2)
		
		sum1 = tf.reduce_sum(max1, axis=1, keep_dims=False) # [batch]
		sum2 = tf.reduce_sum(max2, axis=1, keep_dims=False) # [batch]

		max_mean = tf.maximum( tf.divide( sum1, non_zero1  ), tf.divide( sum2, non_zero2 )) # [batch] 
		
		diff = -cos
		self.diff = diff
		loss = tf.reduce_mean( (diff + label) ** 2 )
		self.loss = loss

		diff_mean = -max_mean
		self.diff_mean = diff_mean
		loss_mean = tf.reduce_mean( (diff_mean + label) ** 2 )
		self.loss_mean = loss_mean
		
		#loss = loss_mean
		#self.loss = loss

		optimizer = tf.train.AdamOptimizer(learning_rate = lr).minimize(loss)
		self.optimizer = optimizer
	
	def say(self, string):
		print string
		if self.log_file != None:
			self.log_file.write(string+'\n')
	
	def init(self, LOAD_PATH, LOG_PATH):
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		sess = tf.Session(config=config)
		saver = tf.train.Saver()
		self.sess = sess
		self.saver = saver
		self.log_file = None
		if (LOAD_PATH is not None):
			if LOAD_PATH == '#LATEST#':
				checkpoint_path = tf.train.latest_checkpoint('./')
			else:
				checkpoint_path = LOAD_PATH
			saver.restore(sess, checkpoint_path)
			if LOG_PATH != None:
				self.log_file = open(LOG_PATH, 'a+')
			self.say('{}, model loaded from file: {}'.format(datetime.datetime.now(), checkpoint_path))
		else:
			sess.run(tf.global_variables_initializer())
			if LOG_PATH != None:
				self.log_file = open(LOG_PATH, 'w')
			self.say('Training start @ {}'.format(datetime.datetime.now()))
	
	def get_embed(self, X1, mask1):
		vec, = self.sess.run(fetches = [self.embed1], feed_dict = {self.X1:X1, self.msg1_mask:mask1})
		return vec
	
	def get_v_embed(self, X1, mask1):
		vec, = self.sess.run(fetches = [self.v_embed1], feed_dict = {self.X1:X1, self.msg1_mask:mask1})
		return vec

	def get_score(self, X1, mask1):  # For searching
		vec, = self.sess.run(fetches = [self.diff], feed_dict = {self.X1:X1, self.msg1_mask:mask1})
		return -vec

	def calc_loss(self, X1, X2, mask1, mask2, y):
		cur_loss, = self.sess.run(fetches = [self.loss], feed_dict = {
				self.X1:X1, self.X2:X2, self.msg1_mask:mask1, self.msg2_mask:mask2, self.label:y})

		return cur_loss
		
	def calc_diff(self, X1, X2, mask1, mask2):
		diff, = self.sess.run(fetches = [self.diff], feed_dict = {self.X1:X1, self.X2:X2, self.msg1_mask:mask1, self.msg2_mask:mask2})
		return diff

	def calc_v_loss(self, X1, X2, mask1, mask2, y):
		cur_loss, = self.sess.run(fetches = [self.loss_mean], feed_dict = {
				self.X1:X1, self.X2:X2, self.msg1_mask:mask1, self.msg2_mask:mask2, self.label:y})

		return cur_loss
	
	def calc_v_diff(self, X1, X2, mask1, mask2):
		diff, = self.sess.run(fetches = [self.diff_mean], feed_dict = {self.X1:X1, self.X2:X2, self.msg1_mask:mask1, self.msg2_mask:mask2})
		return diff
	
	def train(self, X1, X2, mask1, mask2, y):
		self.sess.run(self.optimizer, feed_dict = {self.X1:X1, self.X2:X2, self.msg1_mask:mask1, self.msg2_mask:mask2, self.label:y})
	
	def save(self, path, epoch):
		checkpoint_path = self.saver.save(self.sess, path, global_step=epoch)
		return checkpoint_path

        def close(self):
                print("Closing session")
                self.sess.close()
	
