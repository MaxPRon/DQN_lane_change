import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import road_env
import q_learning


def vectorize_state(state):
    v1 = state['lane']
    v2 = state['y_pos']
    v3 = state['y_velo']
    state_v = np.concatenate((v1,v2))
    state_v = np.concatenate((state_v, v3))
    return state_v


#### Environment parameters ####
num_of_cars = 10
num_of_lanes = 2
track_length = 1000
speed_limit = 50
mode = "constant"

## Ego Init ##
ego_lane_init = 1
ego_pos_init = 0
ego_speed_init = 0.25*speed_limit




#### Network paramters
input_dim = (num_of_cars+1)*3
output_dim = 23
hidden_units = 99
layers = 4
clip_value = 300
learning_rate = 0.001
buffer_size = 50000
batch_size = 32
update_freq = 1000

## RL parameters
gamma = 0.99
eStart = 1
eEnd = 0.1
estep = 1000

r_seed = 1
random.seed(r_seed)

max_train_episodes = 5000
pre_train_steps = 10000 #Fill up buffer

tau = 1 # Factor of copying parameters

#### Start training process ####

## Set up networks ##

tf.reset_default_graph()

mainQN = q_learning.qnetwork(input_dim,output_dim,hidden_units,layers,learning_rate,clip_value)
targetQN = q_learning.qnetwork(input_dim,output_dim,hidden_units,layers,learning_rate,clip_value)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

trainables = tf. trainable_variables()

targetOps = q_learning.updateNetwork(trainables,tau)

random_sweep= 5

## Init environment ##

states = []
actions = []
reward_time = []
reward_average = []
reward_episode = 0
total_steps = 0

done = False

final_save_path = "./bayes/model_random_139075/random_1_Final.ckpt"
#final_save_path = "./random2/model_random_8/random_0_Final.ckpt"

num_tries = 100
finished = 0
#for r in range(1,random_sweep)
for t in range(0,num_tries):
    if t % 100 == 0:
        print("Number of tries:" + str(t))
    with tf.Session() as sess:
        done = False
        sess.run(init)
        saver.restore(sess,final_save_path)
        env = road_env.highway(num_of_lanes, num_of_cars, track_length, speed_limit, ego_lane_init, ego_pos_init,ego_speed_init, mode,r_seed)
        state,_,_ = env.get_state()
        state_v = vectorize_state(state)
        rewards = []
        test = 0
        while done == False:
            action = sess.run(mainQN.action_pred,feed_dict={mainQN.input_state:[state_v]})
            #action = random.randint(0,22)
            state1,reward,done, success = env.step(action)
            rewards.append(reward)
            test += reward
            if test < -10:
                env.render()
            state1_v = vectorize_state(state1)
            state_v = state1_v

        reward_time.append(sum(rewards))
        if success == True:
            finished += 1


average_reward = sum(reward_time)/num_tries

print("Average reward for " + str(num_tries) + " is:" + str(average_reward))
print("Finished succesfully: %s/%s" % (finished, num_tries))