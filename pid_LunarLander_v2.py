import gym
import tempfile
#import scipy
import numpy as np

from gym import wrappers

tdir = tempfile.mkdtemp()
env = gym.make('LunarLander-v2')
env = wrappers.Monitor(env, tdir, force = True)
#print (dir(env.env.env))
print (env.action_space)

def pid_func(env, observation):

    # observation[0] -> x coordinate
    # observation[2] -> vx i.e. dx/dt i.e. velocity in x direction
    target_angle = observation[0]*0.7 + observation[2]*1.0

    if target_angle >  0.5: target_angle =  0.5
    if target_angle < -0.5: target_angle = -0.5
    target_y = np.abs(observation[0])

    # PD control for angle:
    # observation[4]: angle, observation[5]; angularSpeed
    angle_PID = (target_angle - observation[4]) - (observation[5])
    #print("target_angle=%f, angle_todo=%f" % (target_angle, angle_PID))

    # PD control for descent:
    # observation[1]: y coordinate, observation[3]: vy i.e. dy/dt i.e. velocity in y direction
    y_PID = (target_y - observation[1])*0.5 - (observation[3])*0.5
    #print("target_y=%f, y_pid=%f" % (target_y, y_PID))

    # If either of the legs have contact
    if observation[6] or observation[7]:
        angle_PID = 0
        y_PID = -(observation[3])*0.5

    # Selecting action from action_space
    action = 0
    if y_PID > np.abs(angle_PID) and y_PID > 0.05: action = 2
    elif angle_PID < -0.05: action = 3
    elif angle_PID > +0.05: action = 1
    return action

# 100 trials for landing
for t in range(100):
    observation = env.reset()
    while 1:
        env.render()
        #print(observation)
        # select action using pid method
        action = pid_func(env,observation)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t))
            break

env.close()

