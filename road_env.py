import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random





class highway:

    def __init__(self,lanes,cars,length = 500,speed_limit=50,ego_lane=1,ego_pos=100,ego_speed=10,mode = "constant",seed = 0):

        # Road environment
        random.seed(seed)
        self.n_lanes = lanes
        self.n_cars = cars
        self.speed_limit = speed_limit
        self.length = length
        self.s0 = 40 # min_dist
        self.T = 2 # safe time
        self.done = False
        self.timestep = 0
        self.mode = mode
        self.reward = 0
        self.dist_t = self.length-ego_pos
        self.dist_t1 = self.length - ego_pos
        self.success = False

        # Vehicle Parameters
        self.vehicle_length = 3
        self.a = 0.3*self.speed_limit #max_acc
        self.b = 0.4*self.speed_limit # comfortable decceleration
        self.car_velo_con = speed_limit/2

        # Calculation parameter
        self.delta = 4 # acceleration exponent

        # Road outline
        dtype = [('lane',int),('y_pos',float),('y_velo',float),('id',int)]
        self.car_pos = np.zeros(self.n_cars+1,dtype=dtype) # [lane,y-pos ,y-velo,id]


        # Ego vehicle
        self.ego_lane_init = ego_lane
        self.ego_pos_init = ego_pos
        self.ego_velo_init = ego_speed
        self.ego_id = 0


        self.ego = (ego_lane,ego_pos,ego_speed,self.ego_id)
        self.car_pos[0] = self.ego



        # Other traffic members
        for i in range(1,self.n_cars+1):
            self.car_pos[i] = (random.randint(1,self.n_lanes),np.random.choice(np.arange(0.2,1,0.05))*self.length, random.random()*self.speed_limit,i)


        self.car_pos = np.sort(self.car_pos,order='y_pos')


    def render(self):
        self.car_pos = np.sort(self.car_pos, order='id')
        plt.figure(1)
        ax1 = plt.subplot(1, 1, 1)  # rows, cols, index
        ax1.set_xlabel("lane")
        ax1.set_ylabel("y-position")
        ax1.set_xlim(0, self.n_lanes + 1)
        ax1.set_ylim(0, self.length + 100)
        ax1.axhline(self.length, color="g", linestyle="-", linewidth=1)
        # Plot position of other cars
        ax1.plot(self.car_pos['lane'][1:], self.car_pos['y_pos'][1:], 'ro')
        # Plot ego vehicle position
        ax1.plot(self.car_pos['lane'][0], self.car_pos['y_pos'][0], 'bx')
        ax1.grid()
        plt.show(block=False)
        plt.pause(1)
        plt.clf()

        ### Add field of view ###
        #plt.close()

    def car_behaviour_dyn(self):
        self.car_pos = np.sort(self.car_pos,order='y_pos')
        # Safe current state

        for idx,car in enumerate(self.car_pos):

            if car['id'] != 0:

                temp = self.car_pos[:][self.car_pos['lane'] == car['lane']]
                temp = np.sort(temp, order='y_pos')
                idx_next = np.where(temp['id'] == car['id'])
                if temp.shape[0] > 1 and idx_next[0] < temp.shape[0] - 1:

                    idx_next = idx_next[0] + 1
                    next_car = temp[idx_next]

                    v_a1 = next_car['y_velo']
                    x_a1 = next_car['y_pos']
                else:
                    v_a1 = self.speed_limit
                    x_a1 = self.length + 100



                ##### Adapt to every lane #####
                #if idx == self.n_cars:
                #    v_a1 = self.speed_limit
                #    x_a1 = self.length + 100
                #else:
                #    v_a1 = self.car_pos['y_velo'][idx + 1]
                #    x_a1 = self.car_pos['y_pos'][idx + 1]

                v_a = car['y_velo']
                delta_v = v_a-v_a1
                x_a = car['y_pos']
                s_a = x_a1 - x_a - self.length + 0.1


                s_star = self.s0 + v_a*self.T + np.divide(v_a*delta_v,2*np.sqrt(self.a*self.b))
                acc = self.a*(1-np.divide(v_a,self.speed_limit)**self.delta - np.divide(s_star,s_a))**2
                self.car_pos['y_velo'][idx] = v_a + acc

                if self.car_pos['y_velo'][idx] > self.speed_limit:
                    self.car_pos['y_velo'][idx] = self.speed_limit
                self.car_pos['y_pos'][idx] = x_a + self.car_pos['y_velo'][idx]

                if(self.car_pos['y_pos'][idx]>= self.length):
                    self.car_pos['y_pos'][idx] = 0


    def car_behaviour_const(self):

        for idx,car in enumerate(self.car_pos):
            if car['id'] != 0:
                self.car_pos['y_velo'][idx] = self.car_velo_con
                self.car_pos['y_pos'][idx] = car['y_pos'] + self.car_velo_con
                if (self.car_pos['y_pos'][idx] >= self.length):
                    self.car_pos['y_pos'][idx] = 0


    def ego_action(self,action):
        y_pos = self.car_pos['y_pos'][self.car_pos['id'] == 0]
        y_velo = self.car_pos['y_velo'][self.car_pos['id'] == 0]
        lane = self.car_pos['lane'][self.car_pos['id'] == 0]
        # No acceleration
        if action == 0:
            self.car_pos['y_velo'][self.car_pos['id'] == 0] = y_velo
            self.car_pos['y_pos'][self.car_pos['id'] == 0] = y_pos + y_velo
        # positive acceleration 0.1
        if action == 1:
            self.car_pos['y_velo'][self.car_pos['id'] == 0] = y_velo + 0.1*self.a
            self.car_pos['y_pos'][self.car_pos['id'] == 0] = y_pos + y_velo
            # positive acceleration 0.2
        if action == 2:
            self.car_pos['y_velo'][self.car_pos['id'] == 0] = y_velo + 0.2 * self.a
            self.car_pos['y_pos'][self.car_pos['id'] == 0] = y_pos + y_velo
        # positive acceleration 0.3
        if action == 3:
            self.car_pos['y_velo'][self.car_pos['id'] == 0] = y_velo + 0.3 * self.a
            self.car_pos['y_pos'][self.car_pos['id'] == 0] = y_pos + y_velo
        # positive acceleration 0.4
        if action == 4:
            self.car_pos['y_velo'][self.car_pos['id'] == 0] = y_velo + 0.4 * self.a
            self.car_pos['y_pos'][self.car_pos['id'] == 0] = y_pos + y_velo
        # positive acceleration 0.5
        if action == 5:
            self.car_pos['y_velo'][self.car_pos['id'] == 0] = y_velo + 0.5 * self.a
            self.car_pos['y_pos'][self.car_pos['id'] == 0] = y_pos + y_velo
        # positive acceleration 0.6
        if action == 6:
            self.car_pos['y_velo'][self.car_pos['id'] == 0] = y_velo + 0.6* self.a
            self.car_pos['y_pos'][self.car_pos['id'] == 0] = y_pos + y_velo
        # positive acceleration 0.7
        if action == 7:
            self.car_pos['y_velo'][self.car_pos['id'] == 0] = y_velo + 0.7 * self.a
            self.car_pos['y_pos'][self.car_pos['id'] == 0] = y_pos + y_velo
        # positive acceleration 0.8
        if action == 8:
            self.car_pos['y_velo'][self.car_pos['id'] == 0] = y_velo + 0.8 * self.a
            self.car_pos['y_pos'][self.car_pos['id'] == 0] = y_pos + y_velo
        # positive acceleration 0.9
        if action == 9:
            self.car_pos['y_velo'][self.car_pos['id'] == 0] = y_velo + 0.9 * self.a
            self.car_pos['y_pos'][self.car_pos['id'] == 0] = y_pos + y_velo
        # positive acceleration 1
        if action == 10:
            self.car_pos['y_velo'][self.car_pos['id'] == 0] = y_velo + self.a
            self.car_pos['y_pos'][self.car_pos['id'] == 0] = y_pos + y_velo

        # negative acceleration -0.1
        if action == 11:
            self.car_pos['y_velo'][self.car_pos['id'] == 0] = y_velo - 0.1 * self.b
            self.car_pos['y_pos'][self.car_pos['id'] == 0] = y_pos + y_velo
        # negative acceleration -0.2
        if action == 12:
            self.car_pos['y_velo'][self.car_pos['id'] == 0] = y_velo - 0.2 * self.b
            self.car_pos['y_pos'][self.car_pos['id'] == 0] = y_pos + y_velo
        # negative acceleration -0.3
        if action == 13:
            self.car_pos['y_velo'][self.car_pos['id'] == 0] = y_velo - 0.3 * self.b
            self.car_pos['y_pos'][self.car_pos['id'] == 0] = y_pos + y_velo
        # negative acceleration -0.4
        if action == 14:
            self.car_pos['y_velo'][self.car_pos['id'] == 0] = y_velo - 0.4 * self.b
            self.car_pos['y_pos'][self.car_pos['id'] == 0] = y_pos + y_velo
        # negative acceleration -0.5
        if action == 15:
            self.car_pos['y_velo'][self.car_pos['id'] == 0] = y_velo - 0.5 * self.b
            self.car_pos['y_pos'][self.car_pos['id'] == 0] = y_pos + y_velo
        # negative acceleration -0.6
        if action == 16:
            self.car_pos['y_velo'][self.car_pos['id'] == 0] = y_velo - 0.6 * self.b
            self.car_pos['y_pos'][self.car_pos['id'] == 0] = y_pos + y_velo
        # negative acceleration -0.7
        if action == 17:
            self.car_pos['y_velo'][self.car_pos['id'] == 0] = y_velo - 0.7 * self.b
            self.car_pos['y_pos'][self.car_pos['id'] == 0] = y_pos + y_velo
        # negative acceleration -0.8
        if action == 18:
            self.car_pos['y_velo'][self.car_pos['id'] == 0] = y_velo - 0.8 * self.b
            self.car_pos['y_pos'][self.car_pos['id'] == 0] = y_pos + y_velo
        # negative acceleration -0.9
        if action == 19:
            self.car_pos['y_velo'][self.car_pos['id'] == 0] = y_velo - 0.9 * self.b
            self.car_pos['y_pos'][self.car_pos['id'] == 0] = y_pos + y_velo
        # negative acceleration -1
        if action == 20:
            self.car_pos['y_velo'][self.car_pos['id'] == 0] = y_velo - self.b
            self.car_pos['y_pos'][self.car_pos['id'] == 0] = y_pos + y_velo
        # Lane change Left
        if action == 21:
            self.car_pos['y_velo'][self.car_pos['id'] == 0] = y_velo
            self.car_pos['y_pos'][self.car_pos['id'] == 0] = y_pos + y_velo
            self.car_pos['lane'][self.car_pos['id'] == 0] = lane - 1
        # Lane change right
        if action == 22:
            self.car_pos['y_velo'][self.car_pos['id'] == 0] = y_velo
            self.car_pos['y_pos'][self.car_pos['id'] == 0] = y_pos + y_velo
            self.car_pos['lane'][self.car_pos['id'] == 0] = lane + 1


    def reward_function(self):

        self.car_pos = np.sort(self.car_pos, order='y_pos')
        # Get ego vehicle
        ego = self.car_pos[:][self.car_pos['id'] == 0]  # Lane,pos, velo, id

        ##### Get car in front of ego vehicle #####
        temp = self.car_pos[:][self.car_pos['lane'] == ego['lane']]
        temp = np.sort(temp, order='y_pos')
        idx_ego = np.where(temp['id'] == ego['id'])

        if temp.shape[0]>1 and idx_ego[0] < temp.shape[0]-1:
            #idx_next = np.where(temp['id']==ego['id'])
            idx_next = idx_ego[0] + 1
            next_car = temp[idx_next]
            dist_front = abs(next_car['y_pos'] - ego['y_pos'])
        else:
            dist_front = self.length
            #dtype = [('lane', int), ('y_pos', float), ('y_velo', float), ('id', int)]
            #next_car = np.zeros(1, dtype=dtype)  # [lane,y-pos ,y-velo,id]


        #dist_front = abs(next_car['y_pos']-ego['y_pos'])

        ##### Get car in front of ego vehicle #####
        if temp.shape[0]>1 and idx_ego[0] > 0:
            idx_behind = idx_ego[0] - 1
            car_behind = temp[idx_behind]
            dist_back = abs(ego['y_pos']- car_behind['y_pos'])
        else:
            dist_back = self.length

        self.dist_t = self.length - ego['y_pos']

        # Calculate reward
        self.reward = 0
        self.reward += -1
        # Out of bounds
        if ego['lane'] > self.n_lanes or ego['lane'] < 1:
            self.reward += -100
            #print("Fail")
            self.done = True
            self.success = False

        # Get closer to goal
        if self.dist_t < self.dist_t1:
            #print(self.dist_t)
            #print(self.dist_t1)
            self.reward +=1

        # Collision
        if dist_front < self.s0*0.5: # maybe add plus term for being in certain distance
            self.reward += -100
            #print("Fail")
            self.done = True
            self.success = False

        if dist_back < self.s0*0.5:
            self.reward += -100
            self.done = True
            self.success = False

        if dist_front < self.s0: # maybe add plus term for being in certain distance
            self.reward += -10

        if dist_back < self.s0:
            self.reward += -10

        # Driving in the wrong direction
        if ego['y_pos'] < 0:
            self.reward += -100
            self.done = True
            self.success = False

        # Reaching goal
        if ego['y_pos'] >=  self.length:
            self.reward += 100
            #print("Success")
            self.done = True
            self.success = True
        #print(self.reward)

        self.dist_t1 = self.dist_t


    def get_state(self):

        return self.car_pos, self.reward, self.done




    def step(self,action):
        if self.mode == "constant":
            self.car_behaviour_const()
        elif self.mode == "dyn":
            self.car_behaviour_dyn()
            #print("Dyn")

        self.ego_action(action)

        self.reward_function()

        self.timestep += 1
        if self.timestep > 150:
            self.done = True

        return self.car_pos, self.reward, self.done, self.success


#env = highway(4,5,1000,100,1,0,30)

#done = False
#while done == False:
# #    action = random.randint(0,20)
# #    state = env.get_state()
# #    #env.render()
# #    state, reward, done = env.step(action)





