import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import road_env
import q_learning
import pandas as pd
import seaborn as sns


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
random_seed = 1
random.seed(random_seed)

## Ego Init ##
ego_lane_init = 1
ego_pos_init = 0
ego_speed_init = 0.25*speed_limit




#### Network paramters
input_dim = (num_of_cars+1)*3
output_dim = 23
hidden_units = 64 # 64 or 128
layers = 5
clip_value = 300
learning_rate = 0.001
buffer_size = 50000
batch_size = 16
update_freq = 2000 #2000 according to sweep

## RL parameters
gamma = 0.99
eStart = 1
eEnd = 0.1
estep = 50000

max_train_episodes = 25000
pre_train_steps = 10000 #Fill up buffer

tau = 1 # Factor of copying parameters

#path = "./update_frequency/model_h" + str(hidden_units)+"_L" + str(layers) + "__e" + str(max_train_episodes) + "_uf_"+ str(update_freq) + "_" + str(num_of_lanes) + str(num_of_cars)+ "_"+ mode+"/"

param_sweep = 6
random_sweep = 5

#reward_average = np.empty(random_sweep)
average_window = 100

for param in range(1,param_sweep+1):
    ### Change param
    hidden_units = 16*2**param


    reward_average = np.zeros((random_sweep,int(max_train_episodes/average_window)))


    for r_seed in range(0,random_sweep):

        random.seed(r_seed)
        #### Start training process ####
        states = []
        actions = []
        reward_time = []

        folder_path = "./hidden_units_sweep/"

        path = folder_path+"model_h" + str(hidden_units) + "_L" + str(layers) + "__e" + str(max_train_episodes) + "_uf_" + str(update_freq) + "_" + str(num_of_lanes) + str(num_of_cars) + "_" + mode + "/"

        ## Set up networks ##

        tf.reset_default_graph()

        mainQN = q_learning.qnetwork(input_dim,output_dim,hidden_units,layers,learning_rate,clip_value)
        targetQN = q_learning.qnetwork(input_dim,output_dim,hidden_units,layers,learning_rate,clip_value)

        init = tf.global_variables_initializer()

        saver = tf.train.Saver()

        trainables = tf. trainable_variables()

        targetOps = q_learning.updateNetwork(trainables,tau)

        load_model = False
        ## Create replay buffer ##
        exp_buffer = q_learning.replay_buffer(buffer_size)

        ## Randomness of actions ##
        epsilon = eStart
        stepDrop = (eStart-eEnd)/estep

        ## Init environment ##

        #states = []
        #actions = []
        #reward_time = []
        #reward_average = []
        #reward_episode = 0
        total_steps = 0

        done = False

        ## Start Session ##
        with tf.Session() as sess:
            sess.run(init)

            for episode in range(max_train_episodes):
                episode_buffer = q_learning.replay_buffer(buffer_size)
                env = road_env.highway(num_of_lanes, num_of_cars, track_length, speed_limit, ego_lane_init, ego_pos_init,ego_speed_init, mode,r_seed)
                state,_,_ = env.get_state()
                state_v = vectorize_state(state)

                done = False
                reward_episode = 0

                while done == False:
                    if(np.random.random() < epsilon or total_steps < pre_train_steps):
                        action = random.randint(0,22)

                    else:
                        action = sess.run(mainQN.action_pred,feed_dict={mainQN.input_state:[state_v]})

                    state1, reward, done,_ = env.step(action)
                    state1_v = vectorize_state(state1)

                    total_steps += 1

                    episode_buffer.add(np.reshape(np.array([state_v,action,reward,state1_v,done]),[1,5])) # [s,a,r,s1,d]

                    ## Decrease randomness ##
                    if total_steps > pre_train_steps:
                        if epsilon > eEnd:
                            epsilon -= estep

                        trainBatch = exp_buffer.sample(batch_size)
                        ## Calculate Q-value: Q = r(s,a) + gamma*Q(s1,a_max)
                        # Use main network to predict the action a_max
                        action_max = sess.run(mainQN.action_pred,feed_dict={mainQN.input_state:np.vstack(trainBatch[:,3])}) #a_max
                        Qt1_vec = sess.run(targetQN.output_q_predict,feed_dict={targetQN.input_state: np.vstack(trainBatch[:,3])}) # All Q-values for s1

                        end_multiplier = - (trainBatch[:,4]-1) # When last step Q = r(s,a)
                        Qt1 = Qt1_vec[range(batch_size),action_max] # select Q(s1,a_max)

                        # Q = r(s,a) + gamma*Q(s1,a_max)
                        Q_gt = trainBatch[:,2] + gamma*Qt1*end_multiplier

                        ## Optimize network parameters
                        _ = sess.run(mainQN.update,feed_dict={mainQN.input_state:np.vstack(trainBatch[:,0]),mainQN.q_gt:Q_gt,mainQN.actions:trainBatch[:,1]})

                        ## Update target network ##
                        if total_steps % update_freq == 0:
                            print("Update target network")
                            q_learning.updateTarget(targetOps,sess)

                    reward_episode += reward
                    states.append(state)
                    actions.append(action)
                    state_v = state1_v

                exp_buffer.add(episode_buffer.buffer)
                reward_time.append(reward_episode)

                if episode % 5000 == 0:
                    save_path = saver.save(sess,path+"modelRL_"+str(r_seed)+"_" + str(episode)+  ".ckpt")
                    print("Model saved in: %s",save_path)

                if episode % average_window == 0:
                    print("Total steps: ", total_steps, "Average reward over 100 Episodes: ",np.mean(reward_time[-average_window:]),"Episode:", episode)
                    #reward_average[r_seed,int(episode/average_window)].append(np.mean(reward_time[-average_window:]))
                    reward_average[r_seed, int(episode / average_window)]=(np.mean(reward_time[-average_window:]))
            final_save_path = saver.save(sess,path+"random_"+str(r_seed)+"_"  + "Final.ckpt")
            print("Model saved in: %s", final_save_path)

        plt.figure(2)
        axa = plt.subplot(1, 1, 1)
        axa.hist(actions, bins=23, range=(0, 22))
        axa.set_title("Action distribution during training")
        axa.set_xlabel("action")
        axa.set_ylabel("times taken")
        axa.grid()
        plt.tight_layout()
        manager = plt.get_current_fig_manager()
        manager.resize(*manager.window.maxsize())

        #plt.show(block=False)

        plt.savefig(path + 'action_hist_'+str(r_seed)+'.png')
        plt.close()


        with tf.Session() as sess:
            done = False
            sess.run(init)
            saver.restore(sess, final_save_path)
            env = road_env.highway(num_of_lanes, num_of_cars, track_length, speed_limit, ego_lane_init, ego_pos_init,
                                   ego_speed_init, mode)
            state, _, _ = env.get_state()
            state_v = vectorize_state(state)
            rewards = []
            reward_sum = 0
            reward_time = []
            actions = []
            while done == False:
                #print("Net")
                action = sess.run(mainQN.action_pred, feed_dict={mainQN.input_state: [state_v]})
                state1, reward, done = env.step(action)
                rewards.append(reward)
                reward_sum += reward
                reward_time.append(reward_sum)
                actions.append(action)
                state1_v = vectorize_state(state1)
                state_v = state1_v
                # env.render()

        plt.figure(3)
        ax1 = plt.subplot(3, 1, 1)  # rows cols index
        ax1.set_title("Reward for action")
        ax1.set_xlabel("timestep")
        ax1.set_ylabel("reward")
        ax1.grid()
        ax1.plot(rewards)

        ax2 = plt.subplot(3, 1, 2)
        ax2.plot(actions, "ro")
        ax2.set_title("Action taken")
        ax2.set_xlabel("timestep")
        ax2.set_ylabel("action")
        ax2.grid()

        ax3 = plt.subplot(3, 1, 3)  # rows cols index
        ax3.set_title("Reward over time")
        ax3.set_xlabel("timestep")
        ax3.set_ylabel("reward")
        ax3.grid()
        ax3.plot(reward_time)

        plt.tight_layout()
        manager = plt.get_current_fig_manager()
        manager.resize(*manager.window.maxsize())
        #plt.show(block=False)
        plt.savefig(path + 'reward_action_final_'+ str(r_seed)+ '.png')
        plt.close()




    plt.figure(4)
    #ax = plt.subplot(2,1,1)
    #ax.set_title("Reward")
    #ax.set_xlabel("timestep")
    #ax.set_ylabel("reward")
    #ax.plot(rewards)

    ax = plt.subplot(1,1,1)
    ax.set_title("Reward over time")
    ax.set_xlabel("epsiode/100")
    ax.set_ylabel("reward")
    ax.grid()
    #ax.plot(reward_average)
    sns.tsplot(reward_average)
    manager = plt.get_current_fig_manager()
    manager.resize(*manager.window.maxsize())
    #plt.show(block=False)
    plt.tight_layout()
    #plt.savefig(path+'reward'+str(param)+'.png')
    plt.savefig(path + 'reward' + str(param) + '.png')
    #plt.show()
    plt.close()







