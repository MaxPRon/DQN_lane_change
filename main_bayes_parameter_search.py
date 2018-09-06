import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import road_env
import q_learning
import pandas as pd
import seaborn as sns
import csv
from hyperopt import hp
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import fmin, space_eval
import pickle
import time

def vectorize_state(state):
    v1 = state['lane']
    v2 = state['y_pos']
    v3 = state['y_velo']
    state_v = np.concatenate((v1,v2))
    state_v = np.concatenate((state_v, v3))
    return state_v


def compute_objective(space):
    id = int(time.clock())
    num_of_cars = 50
    num_of_lanes = 5
    track_length = 1000
    speed_limit = 50
    mode = "dyn"
    random_seed = 1
    random.seed(random_seed)

    input_dim = (num_of_cars + 1) * 3
    output_dim = 23
    clip_value = 300
    ## Ego Init ##
    ego_lane_init = 1
    ego_pos_init = 0
    ego_speed_init = 0.25 * speed_limit


    #### Reward Variables
    random_sweep = 3
    max_train_episodes = 20000
    objective_window = 100
    average_window = 100

    reward_average = np.zeros((random_sweep, int(max_train_episodes / average_window)))
    reward_objective = np.zeros((random_sweep, max_train_episodes))
    reward_eval_obj = np.zeros((random_sweep, objective_window))

    #### Assign Hyperparameters
    layers = space['layers']
    hidden_units = space['hidden_units']
    learning_rate = space['learning_rate']
    buffer_size = space['buffer_size']
    pre_train_steps = 0.2 * buffer_size
    batch_size = space['batch_size']
    update_freq = space['update_frequency']
    tau = space['tau']
    # RL params
    gamma = space['gamma']
    eStart = space['eStart']
    eEnd = space['eEnd']
    estep = space['estep']

    random_sweep = 3



    for r_seed in range(0,random_sweep):

        random.seed(r_seed)
        #### Start training process ####
        states = []
        actions = []
        reward_time = []

        folder_path = "./bayes_dyn/"

        path = folder_path + "model_random_" + str(id) + "/"

        #### Store Results
        if r_seed == 1:  # Only write for first time

            file = open(path + 'params' + str(id) + '.txt', 'w')
            # file = open(complete_file, 'w')
            file.write('NETWORK PARAMETERS: \n\n')
            file.write('Layers: ' + str(layers) + '\n')
            file.write('Hidden units: ' + str(hidden_units) + '\n')
            file.write('Learning rate: ' + str(learning_rate) + '\n')
            file.write('Buffer size: ' + str(buffer_size) + '\n')
            file.write('Pre_train_steps: ' + str(pre_train_steps) + '\n')
            file.write('Batch_size: ' + str(batch_size) + '\n')
            file.write('Update frequency: ' + str(update_freq) + '\n')
            file.write('Tau: ' + str(tau) + '\n\n')

            file.write('RL PARAMETERS: \n\n')
            file.write('Gamma: ' + str(gamma) + '\n')
            file.write('Epsilon start: ' + str(eStart) + '\n')
            file.write('Epsilon end: ' + str(eEnd) + '\n')
            file.write('Epsilon steps: ' + str(estep) + '\n')

            file.write('SCENARIO PARAMETERS: \n\n')
            file.write('Cars: ' + str(num_of_cars) + '\n')
            file.write('Lanes: ' + str(num_of_lanes) + '\n')
            file.write('Ego speed init: ' + str(ego_speed_init) + '\n')
            file.write('Ego pos init: ' + str(ego_pos_init) + '\n')
            file.write('Ego lane init: ' + str(ego_lane_init) + '\n')

            file.close()

        # Set up networks ##

        tf.reset_default_graph()

        mainQN = q_learning.qnetwork(input_dim, output_dim, hidden_units, layers, learning_rate, clip_value)
        targetQN = q_learning.qnetwork(input_dim, output_dim, hidden_units, layers, learning_rate, clip_value)

        init = tf.global_variables_initializer()

        saver = tf.train.Saver()

        trainables = tf.trainable_variables()

        targetOps = q_learning.updateNetwork(trainables, tau)

        load_model = False
        ## Create replay buffer ##
        exp_buffer = q_learning.replay_buffer(buffer_size)

        ## Randomness of actions ##
        epsilon = eStart
        stepDrop = (eStart - eEnd) / estep

        ## Init environment ##
        total_steps = 0
        done = False

        ## Start Session ##


        with tf.Session() as sess:
            sess.run(init)

            for episode in range(max_train_episodes):
                episode_buffer = q_learning.replay_buffer(buffer_size)
                env = road_env.highway(num_of_lanes, num_of_cars, track_length, speed_limit, ego_lane_init,
                                       ego_pos_init, ego_speed_init, mode, r_seed)
                state, _, _ = env.get_state()
                state_v = vectorize_state(state)

                done = False
                reward_episode = 0

                while done == False:
                    if (np.random.random() < epsilon or total_steps < pre_train_steps):
                        action = random.randint(0, 22)

                    else:
                        action = sess.run(mainQN.action_pred, feed_dict={mainQN.input_state: [state_v]})

                    state1, reward, done,_ = env.step(action)
                    state1_v = vectorize_state(state1)

                    total_steps += 1

                    episode_buffer.add(
                        np.reshape(np.array([state_v, action, reward, state1_v, done]), [1, 5]))  # [s,a,r,s1,d]

                    ## Decrease randomness ##
                    if total_steps > pre_train_steps:
                        if epsilon > eEnd:
                            epsilon -= estep

                        trainBatch = exp_buffer.sample(batch_size)
                        ## Calculate Q-value: Q = r(s,a) + gamma*Q(s1,a_max)
                        # Use main network to predict the action a_max
                        action_max = sess.run(mainQN.action_pred,
                                              feed_dict={mainQN.input_state: np.vstack(trainBatch[:, 3])})  # a_max
                        Qt1_vec = sess.run(targetQN.output_q_predict, feed_dict={
                            targetQN.input_state: np.vstack(trainBatch[:, 3])})  # All Q-values for s1

                        end_multiplier = - (trainBatch[:, 4] - 1)  # When last step Q = r(s,a)
                        Qt1 = Qt1_vec[range(batch_size), action_max]  # select Q(s1,a_max)

                        # Q = r(s,a) + gamma*Q(s1,a_max)
                        Q_gt = trainBatch[:, 2] + gamma * Qt1 * end_multiplier

                        ## Optimize network parameters
                        _ = sess.run(mainQN.update,
                                     feed_dict={mainQN.input_state: np.vstack(trainBatch[:, 0]), mainQN.q_gt: Q_gt,
                                                mainQN.actions: trainBatch[:, 1]})

                        ## Update target network ##
                        if total_steps % update_freq == 0:
                            print("Update target network")
                            q_learning.updateTarget(targetOps, sess)

                    reward_episode += reward
                    states.append(state)
                    actions.append(action)
                    state_v = state1_v

                exp_buffer.add(episode_buffer.buffer)
                reward_time.append(reward_episode)
                reward_objective[r_seed, episode] = reward_episode

                if episode % 5000 == 0:
                    save_path = saver.save(sess, path + "modelRL_" + str(r_seed) + "_" + str(episode) + ".ckpt")
                    print("Model saved in: %s", save_path)

                if episode % average_window == 0:
                    print("Total steps: ", total_steps, "Average reward over 100 Episodes: ",
                          np.mean(reward_time[-average_window:]), "Episode:", episode)
                    # reward_average[r_seed,int(episode/average_window)].append(np.mean(reward_time[-average_window:]))
                    reward_average[r_seed, int(episode / average_window)] = (np.mean(reward_time[-average_window:]))
            final_save_path = saver.save(sess, path + "random_" + str(r_seed) + "_" + "Final.ckpt")
            print("Model saved in: %s", final_save_path)

            with open(path + 'reward_time' + str(id) + str(r_seed) + '.csv', 'w') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                wr.writerow(reward_time)
            myfile.close()

            reward_eval = []

            for t in range(objective_window):
                env = road_env.highway(num_of_lanes, num_of_cars, track_length, speed_limit, ego_lane_init,
                                       ego_pos_init, ego_speed_init, mode, r_seed)
                state, _, _ = env.get_state()
                state_v = vectorize_state(state)

                done = False
                reward_episode = 0

                while done == False:
                    action = sess.run(mainQN.action_pred, feed_dict={mainQN.input_state: [state_v]})
                    state1, reward, done, _ = env.step(action)
                    reward_episode += reward

                reward_eval.append(reward_episode)

        reward_eval_obj[r_seed] = reward_eval

    plt.figure(4)

    ax = plt.subplot(1, 1, 1)
    ax.set_title("Reward over time")
    ax.set_xlabel("epsiode/100")
    ax.set_ylabel("reward")
    ax.grid()

    sns.tsplot(reward_average)
    manager = plt.get_current_fig_manager()
    manager.resize(*manager.window.maxsize())
    # plt.show(block=False)
    plt.tight_layout()
    plt.savefig(folder_path + 'reward' + str(id) + '.png')
    # plt.show()
    plt.close()

    with open(path + 'reward_average' + str(id) + '.csv',
              'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerows(reward_average)
    myfile.close()

    with open(path + 'reward_time' + str(id) + '.csv',
              'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(reward_time)
    myfile.close()

    objective = q_learning.bayes_objective(reward_eval_obj, objective_window)

    return objective

num_of_cars = 10
num_of_lanes = 2
track_length = 1000
speed_limit = 50



input_dim = (num_of_cars + 1) * 3
output_dim = 23

    # Define Parameter space
space = {
    'layers': hp.choice('layers',range(2,7,1)),'hidden_units': hp.choice('hidden_units',range(input_dim,10*input_dim,input_dim)),
    'learning_rate': hp.choice('learning_rate',np.arange(0.00001,0.001,2*0.00001)),'buffer_size': hp.choice('buffer_size',range(10000,11*10000,10000)),
    'batch_size': hp.choice('batch_size',range(16,512,16)),'update_frequency': hp.choice('update_frequency',range(1000,11000,1000)),
    'tau':hp.uniform('tau',0.1,1),'gamma': hp.uniform('gamma',0.75,0.999),'eStart': hp.uniform('eStart',0.75,1),'eEnd': hp.uniform('eEnd',0.01,0.2),
    'estep': hp.choice('estep',range(10000,11*10000,10000))}
tpe_algo = tpe.suggest
tpe_trials = Trials()

best = fmin(compute_objective,space=space,algo=tpe_algo,trials=tpe_trials,max_evals=100)

print(space_eval(space, best))
print(best)













