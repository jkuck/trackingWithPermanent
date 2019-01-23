import numpy as np
import matplotlib.pyplot as plt
from collections import deque

class GenerativeParameters:
    def __init__(self, num_time_steps, num_targets, initial_position_means, initial_velocity_means, initial_position_variance, initial_vel_variance, measurement_variance, spring_constants, dt):
        '''
        Parameters used to generate data
        Inputs:
        - num_time_steps: (int) the number of time steps of data to generate
        - num_targets: (int) the number of targets
        - initial_position_means: (list of floats) the initial position of target i is sampled from a gaussian
                                  with mean initial_position_means[i] and variance initial_position_variance
        - initial_velocity_means: (list of floats) the initial velocity of target i is sampled from a gaussian
                                  with mean initial_velocity_means[i] and variance initial_vel_variance
        - initial_position_variance: (float) the initial position of target i is sampled from a gaussian
                                  with mean initial_position_means[i] and variance initial_position_variance       
        - initial_vel_variance: (float) the initial velocity of target i is sampled from a gaussian
                                  with mean initial_velocity_means[i] and variance initial_vel_variance 
        - measurement_variance: (float) measurements are sampled from a gaussian with mean of the true target
                                location and variance measurement_variance
        - spring_constants: (list of floats) spring_constants[i] is the  spring constant for target i, acceleration = -spring_constants[i]*x (mass = 1)
        - dt: (float) time step
        '''
        self.num_time_steps = num_time_steps
        self.num_targets = num_targets
        self.spring_constants = spring_constants 
        self.dt = dt

        self.initial_position_means = initial_position_means
        self.initial_velocity_means = initial_velocity_means
        self.initial_position_variance = initial_position_variance
        self.initial_vel_variance = initial_vel_variance
        self.measurement_variance = measurement_variance

        #covariance of the observation noise
        # self.r_matrix = np.array([[measurement_variance]])    
        # self.r_matrix = np.array([[measurement_variance, 0],
        #                           [0, measurement_variance]])    
        self.r_matrix = np.array([[.1, 0],
                                  [0, 9999999.0]])    
        
        #covariance of the process noise
        self.q_matrix = np.array([[0.03, 0.0, 0.0],
                                  [0.0, 0.03, 0.0],
                                  [0.0, 0.0, 0.03]])
        
        # self.q_matrix = np.array([[1.0, 0.0, 0.0],
        #                           [0.0, 1.0, 0.0],
        #                           [0.0, 0.0, 1.0]])
        

        #measurement function matrix
        # self.h_matrix = np.array([[1.0, 0.0, 0.0]]) #measurement is just position  

        #measurement is position and velocity
        self.h_matrix = np.array([[1.0,  0.0, 0.0],
                                  [0.0,  1.0, 0.0]])

def get_parameters_and_data(num_time_steps, num_targets, initial_position_means, initial_velocity_means, initial_position_variance,\
                            initial_vel_variance, measurement_variance, spring_constants, dt):
    '''
    High level function that generates data and the parameters used to generate the data
    Inputs:
    - num_time_steps: (int) the number of time steps in the time series

    Outputs:
    - all_states: (list of lists) all_states[i][j] is the ith target's state at the jth time instance
    - all_measurements: (list of lists) all_measurements[i][j] is the ith target's measurement at the jth time instance
    - gen_params: (GenerativeParameters) the parameters used to generate the data

    '''

    #generate the actual data
    gen_params = GenerativeParameters(num_time_steps=num_time_steps, num_targets=num_targets, initial_position_means=initial_position_means,\
                                      initial_velocity_means=initial_velocity_means,\
                                      initial_position_variance=initial_position_variance, initial_vel_variance=initial_vel_variance,\
                                      measurement_variance=measurement_variance, spring_constants=spring_constants, dt=dt)

    (all_states, all_measurements) = generate_all_target_data(gen_params)

    return (all_states, all_measurements, gen_params)




def plot_generated_data(gen_params):
    all_xs = []
    all_zs = []
    ######################## GENERATE DATA ########################
    for target_idx in range(gen_params.num_targets):
        xs, zs = generate_single_target_data(gen_params, target_idx)
        all_xs.append(xs)
        all_zs.append(zs)

    ######################## PLOT DATA ########################
    for target_idx in range(gen_params.num_targets):
        xs = all_xs[target_idx]
        zs = all_zs[target_idx]
        print "states:", [x[0] for x in xs]
        plt.plot([x[0] for x in xs], label='states', marker='+', linestyle="None")
        # print "measurements:", zs
        # plt.plot(zs, label='measurements', marker='x', linestyle="None")
#     plt.ylabel('some numbers')
    plt.legend()
    plt.show()
    plt.close()



def generate_single_target_data(gen_params, target_idx):
    '''
    Inputs:
    - gen_params: (GenerativeParameters) the parameters used to generate the data
    
    Outputs:
    - xs: (list of length gen_params.num_time_steps) the true states of the system, xs[i] is an np.array
          representing the state at time i
    - zs: (list of length gen_params.num_time_steps) the measurements, zs[i] is an np.array
          representing the measurement at time i
    '''
    xs = []
    zs = []

    print 'targets states:'
    H = gen_params.h_matrix
    #sample the initial state and measurement

    initial_position = np.random.normal(loc=gen_params.initial_position_means[target_idx], scale=np.sqrt(gen_params.initial_position_variance))
    initial_velocity = np.random.normal(loc=gen_params.initial_velocity_means[target_idx], scale=np.sqrt(gen_params.initial_vel_variance))
    k = gen_params.spring_constants[target_idx]    
    # initial_velocity = 0
    # x1 = np.array([[                     initial_position],
    #                [                     initial_velocity],
    #                [-k*initial_position]])
    x1 = np.array([initial_position, initial_velocity, -k*initial_position])
    xs.append(x1)

    # z1 = np.random.normal(loc=initial_position, scale=np.sqrt(gen_params.measurement_variance))
    z1 = np.random.multivariate_normal(mean=np.squeeze(np.dot(H, x1)), cov=gen_params.r_matrix)

    zs.append(z1)
    
    dt = gen_params.dt
    #sample the remaining states and measurements
    for idx in range(1, gen_params.num_time_steps):
        # print xs[-1]
        F = np.array([[1.0,    dt,    .5*(dt**2)],
                      [0.0,   1.0,            dt],
                      [ -k, -k*dt, -k*.5*(dt**2)]])
        # F = np.array([[1.0,    dt,    .5*(dt**2)],
        #               [0.0,   1.0,            dt],
        #               [ -k*(1 - k*(dt**2)/2), -k*(2*dt - k*(dt**3)/2), k*(1.5*(dt**2) - .25*k*(dt**4))]])        
        x_t = np.dot(F, xs[-1]) + np.random.multivariate_normal(mean=np.zeros(3), cov=gen_params.q_matrix)
        # print "np.dot(F, xs[-1]):", np.dot(F, xs[-1])
        # print "np.random.multivariate_normal(mean=np.zeros(3), cov=gen_params.q_matrix):", np.random.multivariate_normal(mean=np.zeros(3), cov=gen_params.q_matrix)
        # print "x_t:", x_t
        # print

        xs.append(x_t)
        # z_t = np.random.normal(loc=x_t[0,0], scale=np.sqrt(gen_params.measurement_variance))
        # print "np.squeeze(np.dot(H, x_t)):", np.squeeze(np.dot(H, x_t))
        # print "np.dot(H, x_t):", np.dot(H, x_t)
        # print "x_t:", x_t
        # print "H:", H

        z_t = np.random.multivariate_normal(mean=np.squeeze(np.dot(H, x_t)), cov=gen_params.r_matrix)
        zs.append(z_t)

    return xs, zs


def generate_all_target_data(gen_params):
    '''
    Generate state and measurement data for all targets

    Inputs:
    - gen_params: (GenerativeParameters) the parameters used to generate the data

    Outputs:
    - all_states: (list of lists) all_states[i][j] is the ith target's state at the jth time instance
    - all_measurements: (list of lists) all_measurements[i][j] is the ith target's measurement at the jth time instance

    '''
    #all_states[i][j] is the ith target's state at the jth time instance
    all_states = []
    #all_measurements[i][j] is the ith target's measurement at the jth time instance
    all_measurements = []
    for target_idx in range(gen_params.num_targets):
        cur_target_states, cur_target_measurements = generate_single_target_data(gen_params, target_idx)    
        all_states.append(cur_target_states)
        all_measurements.append(cur_target_measurements)
    return all_states, all_measurements

if __name__ == "__main__":
    np.random.seed(0)

    num_targets = 3

    gen_params = GenerativeParameters(num_time_steps=100, num_targets=num_targets,\
                                      initial_position_means=[40*np.random.rand() for i in range(num_targets)],\
                                      initial_velocity_means=[10*np.random.rand() for i in range(num_targets)],\
                                      # initial_position_means=[20*np.random.rand() for i in range(num_targets)],\
                                      initial_position_variance=20, initial_vel_variance=30, measurement_variance=1,\
                                      spring_constants=[100*np.random.rand() for i in range(num_targets)], dt=.01)


    plot_generated_data(gen_params)

    # get_parameters_and_data(num_time_steps=100, num_targets=3, initial_position_variance=5,\
    #                                   initial_vel_variance=2, measurement_variance=1, spring_constants=5, dt=.1)

