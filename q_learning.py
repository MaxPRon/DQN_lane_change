import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random





class qnetwork:

    def __init__(self,input_dim,output_dim,hidden_units,layers,learning_rate,clip_value):

        # Input
        self.input_state = tf.placeholder(tf.float32,[None,input_dim],name = "input_placeholder")

        # Network Architecture
        self.hidden_layer = tf.layers.dense(self.input_state,hidden_units,activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())

        for n in range(1,layers):
            self.hidden_layer = tf.layers.dense(self.hidden_layer,hidden_units,activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())

        # Network Architecture
        #self.h1 = tf.layers.dense(self.input_state,hidden_units,activation=tf.nn.relu)
        #self.h2 = tf.layers.dense(self.h1,hidden_units,activation=tf.nn.relu)
        #### Implementation Dueling DQN
        ## Q(s,a) = V(s) + A(s,a)

        ## Calculation V(s)
        self.value_fc = tf.layers.dense(inputs=self.hidden_layer,units=hidden_units,activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.value = tf.layers.dense(inputs=self.value_fc,units=1,activation=None,kernel_initializer=tf.contrib.layers.xavier_initializer())

        ## Calculation A(s,a)
        self.advantage_fc = tf.layers.dense(inputs = self.hidden_layer,units=hidden_units,activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.advantage = tf.layers.dense(inputs=self.advantage_fc,units=output_dim,activation=None,kernel_initializer=tf.contrib.layers.xavier_initializer())

        self.output_q_predict = self.value + tf.subtract(self.advantage,tf.reduce_mean(self.advantage,axis=1,keepdims=True))

        #self.output_q_predict = tf.layers.dense(self.hidden_layer,output_dim)
        # Clip values just in case
        self.output_q_predict = tf.clip_by_value(self.output_q_predict,-clip_value,clip_value)
        # Get action (highest q-value)
        self.action_pred = tf.argmax(self.output_q_predict,1) # second axis

        # Compute Cost/Loss
        self.actions = tf.placeholder(tf.int32,shape = [None])
        self.q_gt = tf.placeholder(tf.float32, [None]) # Q-value groundtruth
        # Encode into onehot to select q-value
        self.actions_onehot = tf.one_hot(self.actions,output_dim)

        # select single Q-value given the action
        self.q_action = tf.reduce_sum(tf.multiply(self.output_q_predict,self.actions_onehot),axis = 1)

        self.cost = tf.losses.mean_squared_error(self.q_gt,self.q_action)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.update = self.optimizer.minimize(self.cost)


#### Design Replay buffer

class replay_buffer():
    def __init__(self,buffer_size = 50000):
        self.buffer = []
        self.buffer_size = buffer_size


    def add(self,exp):
        #### Check if buffer full
        if(len(self.buffer)+ len(exp) >= self.buffer_size):
            # Remove oldest exp which is too much
            self.buffer[0:(len(exp)+ len(self.buffer))-self.buffer_size] = []
        self.buffer.extend(exp)

    def sample(self,size):
        return np.reshape(np.array(random.sample(self.buffer,size)),[size,5]) # state,action, reward,state_1, done


#### Helper function for target network update

def updateNetwork(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []

    for idx, var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx + total_vars//2].assign((var.value()*tau)+ ((1-tau)*tfVars[idx+total_vars//2].value())))
    return  op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)


def bayes_objective(reward,window):

    objective = np.mean(reward[-window:],axis=0)
    objective = np.mean(objective)

    return objective

