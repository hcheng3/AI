import numpy as np
import random

# 0:EAST 1:SOUTH 2:WEST 3:NORTH

EAST  =0
SOUTH =1
WEST  =2
NORTH =3
# this is the hmm model I used
class hmm:

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.T_matrice = self.init_T_matrice()
        # self.O_matrice = self.init_O_matrice()
        self.f_matrice = self.init_f_matrice()

    #  initialze the begining f matrice, which is the same probability for all statues include all the position and orientation
    def init_f_matrice(self):
        len = self.width*self.height*4
        pior = [1.0/len]*len
        return np.array(pior)

    # for one current states assign one probability to one previous status
    def transfer(self,prev_state_prob,prev_state_x,prev_state_y,facing,p):
        prev_state_prob[prev_state_x*4+prev_state_y*4*self.width+facing] = p

    # caculate the probabilty for all the possible previous states for a giving current states
    def caculate_previous_state(self,current_state):
        x = current_state[0]
        y = current_state[1]
        facing = current_state[2]

        # 0:facing EAST 1:facing SOUTH 2:facing WEST 3:facing NORTH
        # the previous positions robot could be
        possible_prev =[(x-1,y),(x,y-1),(x+1,y),(x,y+1)]

        # the previous position which is possible with the giving current facing direction
        prev_state_x = possible_prev[facing][0]
        prev_state_y = possible_prev[facing][1]

        # if previous position is beyond the grid boundary, directly return an empty array,cause this situation won't happen
        # so the current position can't be like this
        prev_state_prob = np.array(np.zeros(self.width*self.height*4))

        if prev_state_x <0 or prev_state_x > self.width -1 or prev_state_y <0 or prev_state_y > self.height-1:
            return prev_state_prob
        # for current states, set the probability of facing previous states to 0.8
        self.transfer(prev_state_prob,prev_state_x,prev_state_y,facing,0.8)

        possible_dir_prev = [EAST,SOUTH,WEST,NORTH]
        possible_dir_prev.remove(facing)
        # now possible_dir_prev include all the direction except the facing one
        face_wall_state = []
        #different direction for previous state except the facing one, check all other possible direction

        if EAST in possible_dir_prev:
            if prev_state_x == self.width-1:
                face_wall_state.append((prev_state_x,prev_state_y,EAST))
            else:
                self.transfer(prev_state_prob,prev_state_x,prev_state_y,EAST,0.2/3)
        if SOUTH in possible_dir_prev:
            if prev_state_y ==self.height-1:
                face_wall_state.append((prev_state_x,prev_state_y,SOUTH))
            else:
                self.transfer(prev_state_prob,prev_state_x,prev_state_y,SOUTH,0.2/3)
        if WEST in possible_dir_prev:
            if prev_state_x ==0:
                face_wall_state.append((prev_state_x,prev_state_y,WEST))
            else:
                self.transfer(prev_state_prob,prev_state_x,prev_state_y,WEST,0.2/3)
        if NORTH in possible_dir_prev:
            if prev_state_y ==0:
                face_wall_state.append((prev_state_x,prev_state_y,NORTH))
            else:
                self.transfer(prev_state_prob,prev_state_x,prev_state_y,NORTH,0.2/3)

        # directly choose the direction which is not a wall rather than randomly choose the direction which may has a wall
        for i in face_wall_state:
            self.transfer(prev_state_prob,i[0],i[1],i[2],1.0/(4-len(face_wall_state)))
        # print prev_state_prob
        return prev_state_prob

    # initialize the transition matrice using the probablity caculated before
    def init_T_matrice(self):
        T = np.array(np.zeros(shape=(self.width*self.height*4,self.width*self.height*4)))
        #  i represent the current states
        for i in range(self.width*self.height*4):
            x = (i/4)% self.width
            y = i/(self.width*4)
            facing = i % 4
            # print x ,y,facing
            current_state  =[x,y,facing]
            #  this previous_state_p store with the current states all the probabilities of the previous states
            previous_state_p = self.caculate_previous_state(current_state)
            T[i] = previous_state_p
        return T


    def find_adjacent(self,s_loc):
        x = s_loc[0]
        y = s_loc[1]

        adjacent_1 =[(x-1, y-1), (x-1, y), (x-1, y+1), (x, y-1), (x, y+1), (x+1, y-1), (x+1, y), (x+1, y+1)]

        for i in reversed(adjacent_1):
            if i[0] < 0 or i[0] > self.width-1 or i[1]<0 or i[1] > self.height-1:
                adjacent_1.remove(i)

        adjacent_2 =[(x-2, y-2), (x-2, y-1), (x-2, y),   (x-2, y+1), (x-2, y+2), (x-1, y-2),
                     (x-1, y+2), (x, y-2),   (x, y+2),   (x+1, y-2), (x+1, y+2), (x+2, y-2),
                     (x+2, y-1), (x+2, y),   (x+2, y+1), (x+2, y+2)]
        for j in reversed(adjacent_2):
            if j[0] < 0 or j[0] > self.width-1 or j[1]<0 or j[1] > self.height-1:
                adjacent_2.remove(j)

        adjacent=[adjacent_1,adjacent_2]

        return adjacent

    #  no read condition could happened in two circumstance, one is normal condition, another is
    # sensed location outside the wall when robot is near the boundary
    def O_noread_matrice(self):
        O_n = np.array(np.zeros(shape=(self.width*self.height*4,self.width*self.height*4)))
        for i in range(self.width*self.height*4):
            x = (i/4)% self.width
            y = i/(self.width*4)
            adajacent=self.find_adjacent((x,y))
            O_n[i,i] = 0.1 + 0.25*(8-len(adajacent[0]))+0.05*(16-len(adajacent[1]))
        return O_n


    # s_loc=(x,y)
    #  initialize the O matrice
    def O_read_matrice(self,s_loc):
        O = np.array(np.zeros(shape=(self.width*self.height*4,self.width*self.height*4)))
        # idx = s_loc[0]*4+s_loc[1]*self.width*4
        adjacent = self.find_adjacent(s_loc)
        self.p_e(s_loc[0],s_loc[1],0.1,O)
        for i in adjacent[0]:
            self.p_e(i[0],i[1],0.05,O)
        for j in adjacent[1]:
            self.p_e(j[0],j[1],0.025,O)
        return O
    # assign the probability to the element in the O matrice
    def p_e(self,x,y,probability,O):
        idx = x*4+y*self.width*4
        for j in range(4):
            O[idx+j,idx+j] = probability

    def init_O_matrice(self,s_loc):
        if s_loc is None:
            return self.O_noread_matrice()
        else:
            return self.O_read_matrice(s_loc)

#  class robot could move on the grid and sense the postion
class robot:

    def __init__(self,height,width,x,y,facing):
        self.grid_height = height
        self.grid_width  = width
        self.position_x =x
        self.position_y =y
        self.facing = facing

    # 0:EAST 1:SOUTH 2:WEST 3:NORTH
    # when the robot is in the conner, it will directly choose the direction from reachable rather than try all other 3 direction options
    # match with the hmm
    def move(self):

        x = self.position_x
        y = self.position_y
        # this duplicate reference is used for getting the direction after change the direction
        possible_next_reference= [(x+1,y),(x,y+1),(x-1,y),(x,y-1)]
        possible_next = [(x+1,y),(x,y+1),(x-1,y),(x,y-1)]
        p_move = random.random()
        if 0 <= p_move <= 0.8:
            if possible_next[self.facing][0] < 0 or possible_next[self.facing][0]>= self.grid_width or possible_next[self.facing][1]< 0 or possible_next[self.facing][1]>=self.grid_height:
                for i in reversed(possible_next):
                    if i[0] < 0 or i[0] >= self.grid_width or i[1]<0 or i[1] >= self.grid_height:
                        possible_next.remove(i)
                (self.position_x,self.position_y) = possible_next[random.randrange(0,len(possible_next))]
                # changing the facing when it changed the direction
                self.facing = possible_next_reference.index((self.position_x,self.position_y))
            else:
                self.position_x = possible_next[self.facing][0]
                self.position_y = possible_next[self.facing][1]

        if 0.8 <=p_move <= 1.0:
            possible_next.remove(possible_next[self.facing])
            for j in reversed(possible_next):
                    if j[0] < 0 or j[0] >= self.grid_width or j[1]<0 or j[1] >= self.grid_height:
                        possible_next.remove(j)
            (self.position_x,self.position_y) = possible_next[random.randrange(0,len(possible_next))]
            # changing the faing when it changed the direction
            self.facing = possible_next_reference.index((self.position_x,self.position_y))


    # sense the position using the actual position, if the sensed position is outside the grid boundary return None
    def sense_position(self):
        x = self.position_x
        y = self.position_y
        adjacent_1 =[(x-1, y-1), (x-1, y), (x-1, y+1), (x, y-1), (x, y+1), (x+1, y-1), (x+1, y),
                        (x+1, y+1)]

        adjacent_2 =[(x-2, y-2), (x-2, y-1), (x-2, y),   (x-2, y+1), (x-2, y+2), (x-1, y-2),
                     (x-1, y+2), (x, y-2),   (x, y+2),   (x+1, y-2), (x+1, y+2), (x+2, y-2),
                     (x+2, y-1), (x+2, y),   (x+2, y+1), (x+2, y+2)]

        p_sense = random.random()

        if 0 <= p_sense <= 0.1:
            return (self.position_x,self.position_y)
        if 0.1 < p_sense <= 0.2:
            return None
        if 0.2 < p_sense <= 0.6:
            sensed_pos = adjacent_1[random.randrange(0,7)]
            if sensed_pos[0] < 0 or sensed_pos[0]>= self.grid_width or sensed_pos[1]< 0 or sensed_pos[1]>=self.grid_height:
                return None
            else:
                return sensed_pos
        if 0.6 < p_sense <= 1.0:
            sensed_pos = adjacent_2[random.randrange(0,15)]
            if sensed_pos[0] < 0 or sensed_pos[0]>= self.grid_width or sensed_pos[1]< 0 or sensed_pos[1]>=self.grid_height:
                return None
            else:
                return sensed_pos


# here is the the main for the trial of this program, you can enter the parameters you want to know the average error for
# estimate the postion


# enter width and height of the grid robot will move on
# enter the start postion and facing of the robot
# enter the moves you want in each testing_times
# enter the times you want do the testing

width  =25
height =15
start_postion_x = 15
start_postion_y =7
start_facing = 0
moves = 20
testing_times =20



#  initialze the robot and hmm with the data you enter
robot = robot(height,width,start_postion_x,start_postion_y,start_facing)
hmm = hmm(width,height)
T = hmm.init_T_matrice()


times = 0
m_distance_e = 0

for times in range(testing_times):
    print "times:" ,times
    robot.position_x =start_postion_x
    robot.position_y =start_postion_y
    robot.facing =start_facing
    f = hmm.init_f_matrice()
    # here we do the forward step to calculate the position of the current state
    for i in range(moves):
        O = hmm.init_O_matrice(robot.sense_position())
        mid = np.dot(O,T)
        f = np.dot(mid,f)
        max_prob_idx = np.argmax(f)
        x = (max_prob_idx / 4) % width
        y = max_prob_idx / (width * 4)
        robot.move()
        print "move",i
    print "actual position:" ,robot.position_x,robot.position_y
    print "caculated position:",x,y
    m_distance_e += abs(robot.position_x-x)+abs(robot.position_y-y)
    times +=1


average_error = float(m_distance_e)/times
print "average manhattan distance error:",  average_error

