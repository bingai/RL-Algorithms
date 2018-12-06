import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import argparse
import pprint as pp
import gym
import utils
import os
import time

# ===========================
#   Actor TD3		pi(s)
# ===========================
# ​    """
# ​    Input to the network is the state, output is the action
# ​    under a deterministic policy.
# ​    The output layer activation is a tanh to keep the action
# ​    between -action_bound and action_bound
# ​    """
class Actor(object):
	def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size):
		self.sess = sess
		self.s_dim = state_dim
		self.a_dim = action_dim
		self.action_bound = action_bound
		self.learning_rate = learning_rate
		self.tau = tau
		self.batch_size = batch_size

		self.input = tf.placeholder(shape = [None, self.s_dim], dtype = tf.float32)
		self.out, self.out_scaled = self.create_actor_network('main_actor')
		self.network_params = tf.trainable_variables()
        # # another way to get trainable variables
		# self.network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'main_actor')

		self.target_out, self.target_out_scaled = self.create_actor_network('target_actor')
		self.target_network_params = tf.trainable_variables()[ len(self.network_params): ]
        # # another way to get trainable variables
		# self.target_network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'target_actor')

		self.update_target_network_params = \
			[self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
												  tf.multiply(self.target_network_params[i], 1. - self.tau))
				for i in range(len(self.target_network_params))]         

	def create_actor_network(self, scope, reuse = False):
		with tf.variable_scope(scope, reuse = reuse):
			net = self.input
			net = slim.fully_connected(net, 400, activation_fn = tf.nn.relu)
			net = slim.fully_connected(net, 300, activation_fn = tf.nn.relu)
			net = slim.fully_connected(net, self.a_dim, activation_fn = tf.nn.tanh)
			out_scaled = tf.multiply(net, self.action_bound)
			return net, out_scaled

	def update_target_network(self):
		self.sess.run(self.update_target_network_params)

	def predict(self, inputs):
		return self.sess.run(self.out_scaled, feed_dict={
			self.input: inputs
		})

	def predict_target(self, inputs):
		return self.sess.run(self.target_out_scaled, feed_dict={
			self.input: inputs
		})


# ===========================
#   Critic TD3      Q(s,a)
# ===========================
#   """
#    Input to the network is the state and action, output is Q(s,a).
#    The action must be obtained from the output of the Actor network.
#    """
class Critic(object):
	def __init__(self, sess, state_dim, action_dim, learning_rate, tau, actor_inputs_scaled):
		self.sess = sess
		self.s_dim = state_dim
		self.a_dim = action_dim
		self.learning_rate = learning_rate
		self.tau = tau

		# Q(s,a) input 1: state
		self.state_input = tf.placeholder(shape = [None, self.s_dim], dtype = tf.float32)
		
		# Q(s,a) input 2: action 
		self.actor_input_scaled = actor_inputs_scaled
		self.actor_input = tf.placeholder(shape = [None, self.a_dim], dtype = tf.float32)

		# Q1(s,a) & Q2(s,a)
		self.total_out_scaled,_ = self.create_critic_network('main_critic', self.actor_input_scaled)
		self.out1, self.out2 = self.create_critic_network('main_critic', self.actor_input, reuse = True)

		# target_Q1(s,a) & target_Q2(s,a)
		self.target_out1, self.target_out2 = self.create_critic_network('target_critic', self.actor_input)

		self.network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'main_critic')
		self.target_network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'target_critic')
		self.update_target_network_params = \
				[self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
												      tf.multiply(self.target_network_params[i], 1. - self.tau))
				for i in range(len(self.target_network_params))]

		# update critics 
		self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])
		self.loss = tf.reduce_mean(tf.square(self.out1 - self.predicted_q_value)) + tf.reduce_mean(tf.square(self.out2 - self.predicted_q_value))
		self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list = self.network_params)


	def create_critic_network(self, scope, actions, reuse = False):
		with tf.variable_scope(scope, reuse = reuse):
			# Q1(s,a)
			net = tf.concat([self.state_input, actions], axis = 1)
			net = slim.fully_connected(net, 400, activation_fn = tf.nn.relu)
			net = slim.fully_connected(net, 300, activation_fn = tf.nn.relu)
			net = slim.fully_connected(net, 1, activation_fn = None)
			net1 = net

			# Q2(s,a)
			net = tf.concat([self.state_input, actions], axis = 1)
			net = slim.fully_connected(net, 400, activation_fn = tf.nn.relu)
			net = slim.fully_connected(net, 300, activation_fn = tf.nn.relu)
			net = slim.fully_connected(net, 1, activation_fn = None)
			net2 = net
		return net1, net2	
				
	def update_target_network(self):
		self.sess.run(self.update_target_network_params)
	
	def predict1(self, state_inputs, actor_inputs):
		return self.sess.run(self.out1, feed_dict={
            self.state_input: state_inputs,
            self.actor_input: actor_inputs
        })

	def predict2(self, state_inputs, actor_inputs):
		return self.sess.run(self.out2, feed_dict={
            self.state_input: state_inputs,
            self.actor_input: actor_inputs
        })
		
	def predict_target1(self, state_inputs, actor_inputs):
		return self.sess.run(self.target_out1, feed_dict={
            self.state_input: state_inputs,
            self.actor_input: actor_inputs
        })

	def predict_target2(self, state_inputs, actor_inputs):
		return self.sess.run(self.target_out2, feed_dict={
            self.state_input: state_inputs,
            self.actor_input: actor_inputs
        })


#===========================
#  Tensorflow Summary Ops
#===========================
def test_summaries():
    episode_r = tf.Variable(0.)
    tf.summary.scalar("Test Reward", episode_r)
    episode_timesteps = tf.Variable(0.)
    tf.summary.scalar("Steps before DONE", episode_timesteps)

    summary_vars = [episode_r, episode_timesteps]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars


def test(sess, env, actor):

	# Set up summary Ops
	summary_ops, summary_vars = test_summaries()
	sess.run(tf.global_variables_initializer())
	writer = tf.summary.FileWriter(args['test_dir'], sess.graph)

	s = env.reset()
	done = False
	episode_r = 0
	episode_timesteps = 0

	while not done:
		env.render()
		action = actor.predict(np.reshape(s, (1, actor.s_dim)))
		s2, r, done, _ = env.step(action)
		time.sleep(0.05)
		episode_r += r
		s = s2
		episode_timesteps += 1
		summary_str = sess.run(summary_ops, feed_dict={
											summary_vars[0]: np.asscalar(episode_r),
											summary_vars[1]: episode_timesteps})
		writer.add_summary(summary_str,episode_timesteps)
		writer.flush()
	print("During evaluation the mean episode reward is {}, and it took {} steps before Done".format(np.asscalar(episode_r), episode_timesteps))	


def main(args):
	if not os.path.exists(args['save_dir']) :
		os.makedirs(args['save_dir'])

	with tf.Session() as sess:
		env = gym.make(args['env'])
		np.random.seed(int(args['random_seed']))
		tf.set_random_seed(int(args['random_seed']))
		env.seed(int(args['random_seed']))

		state_dim = env.observation_space.shape[0]
		action_dim = env.action_space.shape[0]
		action_bound = int(env.action_space.high[0])

		actor = Actor(sess, state_dim, action_dim, action_bound,
							 float(args['actor_lr']), float(args['tau']),
							 int(args['minibatch_size']))

		critic = Critic(sess, state_dim, action_dim,
							   float(args['critic_lr']), float(args['tau']),
							   actor.out_scaled)
        
		saver = tf.train.Saver()
		saver.restore(sess, os.path.join(args['save_dir'], args['env']))
		test(sess, env, actor)
		time.sleep(5)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='provide arguments for TD3 agent')

	parser.add_argument('--actor_lr', help='actor network learning rate', default=0.001)
	parser.add_argument('--critic_lr', help='critic network learning rate', default=0.001)
	parser.add_argument("--start_timesteps", help='action sampling strategy time trigger', default=1e4, type=int)
	parser.add_argument('--tau', help='soft target update parameter', default=0.005)
	parser.add_argument('--minibatch_size', help='size of minibatch for minibatch-SGD', default=100)
	parser.add_argument("--policy_noise", help = 'std of noise added ', default=0.2, type=float)		
	parser.add_argument("--noise_clip", default=0.5, type=float)		
	parser.add_argument("--discount", default=0.99, type=float)	
	parser.add_argument("--policy_freq", default=2, type=int)
	parser.add_argument("--expl_noise", default=0.1, type=float)

	
	# parser.add_argument('--env', help='choose the gym env', default='Pendulum-v0')
	# parser.add_argument('--env', help='choose the gym env', default='InvertedPendulum-v2')
	# parser.add_argument('--env', help='choose the gym env', default='InvertedDoublePendulum-v2')
	# parser.add_argument('--env', help='choose the gym env', default='MountainCarContinuous-v0')
	parser.add_argument('--env', help='choose the gym env', default='Reacher-v2')
	# parser.add_argument('--env', help='choose the gym env', default='HalfCheetah-v2')
	# parser.add_argument('--env', help='choose the gym env', default='Hopper-v2')
	# parser.add_argument('--env', help='choose the gym env', default='Walker2d-v2')
	# parser.add_argument('--env', help='choose the gym env', default='Ant-v2')
	# parser.add_argument('--env', help='choose the gym env', default='Humanoid-v2')
	# parser.add_argument('--env', help='choose the gym env', default='HumanoidStandup-v2')
	# parser.add_argument('--env', help='choose the gym env', default='Swimmer-v2')

	parser.add_argument('--random_seed', help='random seed for repeatability', default=0)
	parser.add_argument("--max_timesteps", default=1e6, type=float)	
	parser.add_argument("--eval_episodes", default=100, type=float)
	parser.add_argument("--save_dir", default='./models/', help = 'save directory')	
	parser.add_argument("--save_timesteps", default=2e5, type=float)

	#tensorboard book-keeping: training
	parser.add_argument('--summary_dir', help='directory for storing tensorboard info', default='./models/tensorboard')
	#tensorboard book-keeping: tratesting
	parser.add_argument('--test_dir', help='directory for storing tensorboard info : test', default='./models/tensorboard_test')

	args = vars(parser.parse_args())
	
	pp.pprint(args)

	main(args)
