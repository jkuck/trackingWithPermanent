import numpy as np
from collections import deque

class GenerativeParameters:
    def __init__(self, num_time_steps, state_space, previous_dependent_states_shape, all_initial_state_probabilities, \
        transition_probabilities, emission_probabilities, markov_order, num_targets):
        '''
        Parameters used to generate data
        Inputs:
        - num_time_steps: (int) the number of time steps of data to generate
        - state_space: (1d np.array) specifies the size of our state space, e.g. the array
            [10, 5, 3] denotes a 3d state space with 10, 5, and 3 discrete values in each dimension
        - previous_dependent_states_shape: (tuple) state_space repeated markov_order times.  e.g. if markov_order = 2
            then (10, 5, 3, 10, 5, 3)
        - all_initial_state_probabilities: (list of np.array of length #targets) ith entry has shape state_space 
            and specifies the probability of the ith target beginning in each state (summing over all state must equal 1)        
        - transition_probabilities: (np.array) has shape concatenate(state_space, state_space) e.g.
            (10, 5, 3, 10, 5, 3) for the above example.  The element (a, b, c, d, e, f) specifies the
            probability of transitioning from state (a, b, c) to (d, e, f).  Summing over
            transition_probabilities(a,b,c, :, :, :) must be 1 to be a properly normalized distribution
        - emission_probabilities: (np.array) has shape concatenate(state_space, state_space) e.g.
            (10, 5, 3, 10, 5, 3) for the above example.  The element (a, b, c, d, e, f) specifies the
            probability of emitting the measurement (d, e, f) when the state is (a, b, c).  Summing over
            emission_probabilities(a,b,c, :, :, :) must be 1 to be a properly normalized distribution
        - markov_order: (int) number of previous states that transition probabilities are dependent on
        - num_targets: (int) the number of targets
        '''        
        self.num_time_steps = num_time_steps
        self.state_space = state_space
        self.previous_dependent_states_shape = previous_dependent_states_shape
        #1-d np.array's have length 0 which causes some problems
        #this is a tuple specifying the number of dimensions in the state space and the size of each dimension
        if state_space.size > 1: #does state_space have more than one dimension? 
            self.state_space_tuple = tuple(state_space)
        else:
            self.state_space_tuple = (state_space,)        
        self.all_initial_state_probabilities = all_initial_state_probabilities
        self.transition_probabilities = transition_probabilities
        self.emission_probabilities = emission_probabilities
        self.markov_order = markov_order
        self.num_targets = num_targets


def get_parameters_and_data(num_time_steps, state_space, measurement_space,\
    markov_order, num_targets):
    '''
    High level function that generates data and the parameters used to generate the data
    Inputs:
    - num_time_steps: (int) the number of time steps in the time series
    - state_space: (np.array) [20,20,20] -> 3d state space of size 20x20
    - measurement_space: (np.array) [20,20] -> 2d measurement space of size 20x20
    - markov_order: (int) the number of previous states that determines the transition distribution

    Outputs:
    - all_states: (list of lists) all_states[i][j] is the ith target's state at the jth time instance
    - all_measurements: (list of lists) all_measurements[i][j] is the ith target's measurement at the jth time instance
    - gen_params: (GenerativeParameters) the parameters used to generate the data

    '''
    HIDDEN_STATE_NOT_IN_PRIOR = True

    ######################## DEFINE DATA GENERATION PARAMETERS ########################
    #1-d np.array's have length 0 which causes some problems below without STATE_SPACE_TUPLE
    #tuple specifying the number of dimensions in the state space and the size of each dimension
    if state_space.size > 1: #does state_space have more than one dimension? 
        STATE_SPACE_TUPLE = tuple(state_space)
    else:
        STATE_SPACE_TUPLE = (state_space,)
    #1-d np.array's have length 0 which causes some problems below without MEASUREMENT_SPACE_TUPLE
    #tuple specifying the number of dimensions in the state space and the size of each dimension
    if measurement_space.size > 1: #does measurement_space have more than one dimension? 
        MEASUREMENT_SPACE_TUPLE = tuple(measurement_space)
    else:
        MEASUREMENT_SPACE_TUPLE = (measurement_space,)


    #TRANSITION_PROBABILITIES has shape (markov_order+1)*state_space,
    #that is the transition function's domain is over the previous markov_order states
    #and the range is over the state space
    PREVIOUS_DEPENDENT_STATES_SHAPE = []
    transition_probs_shape = []
    for i in range(markov_order):
        PREVIOUS_DEPENDENT_STATES_SHAPE.extend(list(STATE_SPACE_TUPLE))
        transition_probs_shape.extend(list(STATE_SPACE_TUPLE))

    PREVIOUS_DEPENDENT_STATES_SHAPE = tuple(PREVIOUS_DEPENDENT_STATES_SHAPE)
    transition_probs_shape.extend(list(STATE_SPACE_TUPLE))
    transition_probs_shape = tuple(transition_probs_shape)

    # INITIAL_STATE_PROBABILITIES = np.random.rand(*STATE_SPACE_TUPLE)
    # INITIAL_STATE_PROBABILITIES = np.ones(STATE_SPACE_TUPLE)
    # INITIAL_STATE_PROBABILITIES = np.power(INITIAL_STATE_PROBABILITIES, 1)
    # INITIAL_STATE_PROBABILITIES /= np.sum(INITIAL_STATE_PROBABILITIES)

    ALL_INITIAL_STATE_PROBABILITIES = []

    use_deterministic_initial_states = False
    if use_deterministic_initial_states:
        deterministic_initial_target_states_1d = np.random.choice(np.prod(STATE_SPACE_TUPLE), num_targets, replace=False)
        print("deterministic_initial_target_states_1d:", deterministic_initial_target_states_1d)
        for t_idx in range(num_targets):
            INITIAL_STATE_PROBABILITIES = np.zeros(STATE_SPACE_TUPLE)
            cur_target_idx_1d = deterministic_initial_target_states_1d[t_idx]
            cur_target_idx_state_space_dimension = np.unravel_index(cur_target_idx_1d, STATE_SPACE_TUPLE)
            print("cur_target_idx_state_space_dimension:", cur_target_idx_state_space_dimension)
            INITIAL_STATE_PROBABILITIES[cur_target_idx_state_space_dimension] = 1.0
            ALL_INITIAL_STATE_PROBABILITIES.append(INITIAL_STATE_PROBABILITIES)

    else:
        for t_idx in range(num_targets):
            if HIDDEN_STATE_NOT_IN_PRIOR:
                INITIAL_STATE_PROBABILITIES = np.zeros(STATE_SPACE_TUPLE)
                for cur_state_index in np.ndindex(*MEASUREMENT_SPACE_TUPLE):
                    mask = np.random.binomial(n=1, p=1, size=MEASUREMENT_SPACE_TUPLE + (1,))    
                    cur_initial_state_probs = np.random.rand(*MEASUREMENT_SPACE_TUPLE+(1,)) * mask
                    cur_initial_state_probs = np.power(cur_initial_state_probs, 10.0)            
                    # cur_initial_state_probs /= np.sum(cur_initial_state_probs)
                    # print "INITIAL_STATE_PROBABILITIES.shape:", INITIAL_STATE_PROBABILITIES.shape
                    # print "INITIAL_STATE_PROBABILITIES[:, cur_state_index].shape:", INITIAL_STATE_PROBABILITIES[:, cur_state_index].shape
                    # print "cur_initial_state_probs.shape:", cur_initial_state_probs.shape
                    INITIAL_STATE_PROBABILITIES[:, cur_state_index] = cur_initial_state_probs
                INITIAL_STATE_PROBABILITIES /= np.sum(INITIAL_STATE_PROBABILITIES)
                ALL_INITIAL_STATE_PROBABILITIES.append(INITIAL_STATE_PROBABILITIES)


            else:
                INITIAL_STATE_PROBABILITIES = np.random.rand(*STATE_SPACE_TUPLE)
            #     INITIAL_STATE_PROBABILITIES = np.ones(STATE_SPACE_TUPLE)
                INITIAL_STATE_PROBABILITIES = np.power(INITIAL_STATE_PROBABILITIES, 10)
                INITIAL_STATE_PROBABILITIES /= np.sum(INITIAL_STATE_PROBABILITIES)
                ALL_INITIAL_STATE_PROBABILITIES.append(INITIAL_STATE_PROBABILITIES)




    TRANSITION_PROBABILITIES = np.zeros(transition_probs_shape)

    # SMOOTHING_POWER = 120
    # SMOOTHING_POWER = 75


    #MHT gets correct result for 3 particles, not 2 and gt is still ~ most probable
    #another run, MHT gets correct result for 50 particles, not 40 and gt is still ~ most probable
    #another run, too much noise
    # SMOOTHING_POWER = 65 #good noise for use_group_particles=False


    # SMOOTHING_POWER = 50 
    # SMOOTHING_POWER = 46
    # SMOOTHING_POWER = 42
    # SMOOTHING_POWER = 35
    # SMOOTHING_POWER = 25 #gt not most probable, 500 particles slightly finds slight better prob than 100, 100 better than 35, 35 better than 25

    SMOOTHING_POWER = 50 
    # SMOOTHING_POWER = 9  #GOOD EXAMPLE for use_group_particles=True, state space of 100

    # SMOOTHING_POWER = 7 #GOOD EXAMPLE for use_group_particles=True

    # SMOOTHING_POWER = 5 #ground truth doesn't have largest log-likelihood
    # SMOOTHING_POWER = 4
    # SMOOTHING_POWER = 3 #ground truth doesn't have largest log-likelihood



    use_constant_size_non_zero_transition_probs = True
    non_zero_transition_state_count = 3
    # non_zero_transition_probs = np.array([.98, .015, .005])
    non_zero_transition_probs = np.array([.999, .0009, .0001])


    #iterate over each state and create a transition distribution that sums to 1 over all next_states
    for cur_state_index in np.ndindex(*PREVIOUS_DEPENDENT_STATES_SHAPE):
        if use_constant_size_non_zero_transition_probs:
            non_zero_transition_states1d = np.random.choice(np.prod(STATE_SPACE_TUPLE), non_zero_transition_state_count, replace=False)
            for t_idx, cur_non_zero_transition_states1d in enumerate(non_zero_transition_states1d):
                cur_non_zero_transition_state_full_dimension = np.unravel_index(cur_non_zero_transition_states1d, STATE_SPACE_TUPLE)        
                #note, + used to concatenate tuples
                TRANSITION_PROBABILITIES[cur_state_index + cur_non_zero_transition_state_full_dimension] = non_zero_transition_probs[t_idx]

        else:
            mask = np.random.binomial(n=1, p=1.0, size=STATE_SPACE_TUPLE)
            TRANSITION_PROBABILITIES[cur_state_index] = np.random.rand(*STATE_SPACE_TUPLE) * mask
            TRANSITION_PROBABILITIES[cur_state_index] = np.power(TRANSITION_PROBABILITIES[cur_state_index], SMOOTHING_POWER)
            TRANSITION_PROBABILITIES[cur_state_index] /= np.sum(TRANSITION_PROBABILITIES[cur_state_index])

    TRANSITION_PROBABILITIES = [TRANSITION_PROBABILITIES for i in range(num_targets)]
        
    emission_probs_shape = list(STATE_SPACE_TUPLE)
    emission_probs_shape.extend(list(MEASUREMENT_SPACE_TUPLE))
    emission_probs_shape = tuple(emission_probs_shape)
        
    EMISSION_PROBABILITIES = np.zeros(emission_probs_shape)

    use_identical_emission_probs = False #use the same emission probabilities for every state for easy visualization, only implemented for 1-d state space
    add_constant_emission_probs = False
    #iterate over each state and create an emission distribution that sums to 1 over all measurements
    # for cur_state_index in np.ndindex(*STATE_SPACE_TUPLE):
    for cur_state_index in np.ndindex(*MEASUREMENT_SPACE_TUPLE):
        if use_identical_emission_probs:
    #         constant_emission_probs = np.array([.1, .2, .4, .2, .1])
    #         constant_emission_probs = np.array([.025, .075, .1, .15, .3, .15, .1, .075, .025])
    #         constant_emission_probs = np.array([1 for i in range(5)])
    #         constant_emission_probs = constant_emission_probs/np.sum(constant_emission_probs)
            constant_emission_probs = np.array([.005, .02, .95, .02, .005])
    #         constant_emission_probs = np.array([.01, .02, .07, .8, .07, .02, .01])

            for emission_idx in range(len(constant_emission_probs)):
                EMISSION_PROBABILITIES[cur_state_index, (cur_state_index[0] + emission_idx - len(constant_emission_probs)//2)%STATE_SPACE_TUPLE[0]] = constant_emission_probs[emission_idx]
        else:
            # mask = np.random.binomial(n=1, p=.3, size=MEASUREMENT_SPACE_TUPLE)    
            # EMISSION_PROBABILITIES[:, cur_state_index] = np.random.rand(*MEASUREMENT_SPACE_TUPLE) * mask
            # EMISSION_PROBABILITIES[:, cur_state_index] = np.power(EMISSION_PROBABILITIES[:, cur_state_index], SMOOTHING_POWER)
            # if add_constant_emission_probs:
            #     constant_emission_probs = 1*np.array([.03, .07, .8, .07, .03])
            #     for emission_idx in range(len(constant_emission_probs)):
            #         EMISSION_PROBABILITIES[cur_state_index, (cur_state_index[0] + emission_idx - len(constant_emission_probs)//2)%STATE_SPACE_TUPLE[0]] += constant_emission_probs[emission_idx]
            
            # EMISSION_PROBABILITIES[:, cur_state_index] /= np.sum(EMISSION_PROBABILITIES[:, cur_state_index])

            mask = np.random.binomial(n=1, p=.3, size=MEASUREMENT_SPACE_TUPLE)    
            cur_emission_probs = np.random.rand(*MEASUREMENT_SPACE_TUPLE) * mask
            cur_emission_probs = np.power(cur_emission_probs, SMOOTHING_POWER)          
            cur_emission_probs /= np.sum(cur_emission_probs)

            EMISSION_PROBABILITIES[:, cur_state_index] = cur_emission_probs
            # for cur_unobserved_state_index in np.ndindex(*MEASUREMENT_SPACE_TUPLE):
               #  cur_unobserved_state_emission_noise = np.random.rand(*MEASUREMENT_SPACE_TUPLE) * mask
               #  cur_unobserved_state_emission_noise /= np.sum(cur_unobserved_state_emission_noise)*10
               #  cur_emission_probs_plus_noise = cur_emission_probs + cur_unobserved_state_emission_noise
               #  cur_emission_probs_plus_noise /= np.sum(cur_emission_probs_plus_noise)
               #  EMISSION_PROBABILITIES[cur_unobserved_state_index, cur_state_index] = cur_emission_probs_plus_noise

    print "EMISSION_PROBABILITIES[0]:", EMISSION_PROBABILITIES[0]
    EMISSION_PROBABILITIES = [EMISSION_PROBABILITIES for i in range(num_targets)]

    #generate the actual data
    gen_params = GenerativeParameters(num_time_steps=num_time_steps, state_space=state_space, \
        previous_dependent_states_shape=PREVIOUS_DEPENDENT_STATES_SHAPE, all_initial_state_probabilities=ALL_INITIAL_STATE_PROBABILITIES, \
        transition_probabilities=TRANSITION_PROBABILITIES, emission_probabilities=EMISSION_PROBABILITIES, markov_order=markov_order, \
        num_targets=num_targets)
    (all_states, all_measurements) = generate_all_target_data(gen_params)

    return (all_states, all_measurements, gen_params)


def get_parameters_and_data_targets_identical_plus_noise(num_time_steps, state_space, measurement_space,\
    markov_order, num_targets):
    '''
    All targets have the same base initial state, transition, and measurement functions with some target specific noise added

    High level function that generates data and the parameters used to generate the data
    Inputs:
    - num_time_steps: (int) the number of time steps in the time series
    - state_space: (np.array) [20,20,20] -> 3d state space of size 20x20
    - measurement_space: (np.array) [20,20] -> 2d measurement space of size 20x20
    - markov_order: (int) the number of previous states that determines the transition distribution

    Outputs:
    - all_states: (list of lists) all_states[i][j] is the ith target's state at the jth time instance
    - all_measurements: (list of lists) all_measurements[i][j] is the ith target's measurement at the jth time instance
    - gen_params: (GenerativeParameters) the parameters used to generate the data

    '''
    HIDDEN_STATE_NOT_IN_PRIOR = True

    ######################## DEFINE DATA GENERATION PARAMETERS ########################
    #1-d np.array's have length 0 which causes some problems below without STATE_SPACE_TUPLE
    #tuple specifying the number of dimensions in the state space and the size of each dimension
    if state_space.size > 1: #does state_space have more than one dimension? 
        STATE_SPACE_TUPLE = tuple(state_space)
    else:
        STATE_SPACE_TUPLE = (state_space,)
    #1-d np.array's have length 0 which causes some problems below without MEASUREMENT_SPACE_TUPLE
    #tuple specifying the number of dimensions in the state space and the size of each dimension
    if measurement_space.size > 1: #does measurement_space have more than one dimension? 
        MEASUREMENT_SPACE_TUPLE = tuple(measurement_space)
    else:
        MEASUREMENT_SPACE_TUPLE = (measurement_space,)


    #TRANSITION_PROBABILITIES has shape (markov_order+1)*state_space,
    #that is the transition function's domain is over the previous markov_order states
    #and the range is over the state space
    PREVIOUS_DEPENDENT_STATES_SHAPE = []
    transition_probs_shape = []
    for i in range(markov_order):
        PREVIOUS_DEPENDENT_STATES_SHAPE.extend(list(STATE_SPACE_TUPLE))
        transition_probs_shape.extend(list(STATE_SPACE_TUPLE))

    PREVIOUS_DEPENDENT_STATES_SHAPE = tuple(PREVIOUS_DEPENDENT_STATES_SHAPE)
    transition_probs_shape.extend(list(STATE_SPACE_TUPLE))
    transition_probs_shape = tuple(transition_probs_shape)

    # INITIAL_STATE_PROBABILITIES = np.random.rand(*STATE_SPACE_TUPLE)
    # INITIAL_STATE_PROBABILITIES = np.ones(STATE_SPACE_TUPLE)
    # INITIAL_STATE_PROBABILITIES = np.power(INITIAL_STATE_PROBABILITIES, 1)
    # INITIAL_STATE_PROBABILITIES /= np.sum(INITIAL_STATE_PROBABILITIES)

    ALL_INITIAL_STATE_PROBABILITIES = []

    if HIDDEN_STATE_NOT_IN_PRIOR:
        BASE_INITIAL_STATE_PROBABILITIES = np.zeros(STATE_SPACE_TUPLE)
        for cur_state_index in np.ndindex(*MEASUREMENT_SPACE_TUPLE):
            mask = np.random.binomial(n=1, p=1, size=MEASUREMENT_SPACE_TUPLE + (1,))    
            cur_initial_state_probs = np.random.rand(*MEASUREMENT_SPACE_TUPLE+(1,)) * mask
            cur_initial_state_probs = np.power(cur_initial_state_probs, 10.0)            
            # cur_initial_state_probs /= np.sum(cur_initial_state_probs)
            # print "BASE_INITIAL_STATE_PROBABILITIES.shape:", BASE_INITIAL_STATE_PROBABILITIES.shape
            # print "BASE_INITIAL_STATE_PROBABILITIES[:, cur_state_index].shape:", BASE_INITIAL_STATE_PROBABILITIES[:, cur_state_index].shape
            # print "cur_initial_state_probs.shape:", cur_initial_state_probs.shape
            BASE_INITIAL_STATE_PROBABILITIES[:, cur_state_index] = cur_initial_state_probs
        BASE_INITIAL_STATE_PROBABILITIES /= np.sum(BASE_INITIAL_STATE_PROBABILITIES)
    else:
        BASE_INITIAL_STATE_PROBABILITIES = np.random.rand(*STATE_SPACE_TUPLE)
    #     BASE_INITIAL_STATE_PROBABILITIES = np.ones(STATE_SPACE_TUPLE)
        BASE_INITIAL_STATE_PROBABILITIES = np.power(BASE_INITIAL_STATE_PROBABILITIES, 10)
        BASE_INITIAL_STATE_PROBABILITIES /= np.sum(BASE_INITIAL_STATE_PROBABILITIES)

    for t_idx in range(num_targets):
        if HIDDEN_STATE_NOT_IN_PRIOR:
            CUR_NOISE_INITIAL_STATE_PROBABILITIES = np.zeros(STATE_SPACE_TUPLE)
            for cur_state_index in np.ndindex(*MEASUREMENT_SPACE_TUPLE):
                mask = np.random.binomial(n=1, p=1, size=MEASUREMENT_SPACE_TUPLE + (1,))    
                cur_initial_state_probs = np.random.rand(*MEASUREMENT_SPACE_TUPLE+(1,)) * mask
                cur_initial_state_probs = np.power(cur_initial_state_probs, 10.0)            
                # cur_initial_state_probs /= np.sum(cur_initial_state_probs)
                # print "CUR_NOISE_INITIAL_STATE_PROBABILITIES.shape:", CUR_NOISE_INITIAL_STATE_PROBABILITIES.shape
                # print "CUR_NOISE_INITIAL_STATE_PROBABILITIES[:, cur_state_index].shape:", CUR_NOISE_INITIAL_STATE_PROBABILITIES[:, cur_state_index].shape
                # print "cur_initial_state_probs.shape:", cur_initial_state_probs.shape
                CUR_NOISE_INITIAL_STATE_PROBABILITIES[:, cur_state_index] = cur_initial_state_probs
            CUR_NOISE_INITIAL_STATE_PROBABILITIES /= np.sum(CUR_NOISE_INITIAL_STATE_PROBABILITIES)
        else:
            CUR_NOISE_INITIAL_STATE_PROBABILITIES = np.random.rand(*STATE_SPACE_TUPLE)
        #     CUR_NOISE_INITIAL_STATE_PROBABILITIES = np.ones(STATE_SPACE_TUPLE)
            CUR_NOISE_INITIAL_STATE_PROBABILITIES = np.power(CUR_NOISE_INITIAL_STATE_PROBABILITIES, 10)
            CUR_NOISE_INITIAL_STATE_PROBABILITIES /= np.sum(CUR_NOISE_INITIAL_STATE_PROBABILITIES)
        CUR_INITIAL_STATE_PROBS = 1.0*BASE_INITIAL_STATE_PROBABILITIES + 0.0*CUR_NOISE_INITIAL_STATE_PROBABILITIES
        CUR_INITIAL_STATE_PROBS /= np.sum(CUR_INITIAL_STATE_PROBS)
        ALL_INITIAL_STATE_PROBABILITIES.append(CUR_INITIAL_STATE_PROBS)




    BASE_TRANSITION_PROBABILITIES = np.zeros(transition_probs_shape)

    # SMOOTHING_POWER = 120
    # SMOOTHING_POWER = 75


    #MHT gets correct result for 3 particles, not 2 and gt is still ~ most probable
    #another run, MHT gets correct result for 50 particles, not 40 and gt is still ~ most probable
    #another run, too much noise
    # SMOOTHING_POWER = 65 #good noise for use_group_particles=False


    # SMOOTHING_POWER = 50 
    # SMOOTHING_POWER = 46
    # SMOOTHING_POWER = 42
    # SMOOTHING_POWER = 35
    # SMOOTHING_POWER = 25 #gt not most probable, 500 particles slightly finds slight better prob than 100, 100 better than 35, 35 better than 25

    SMOOTHING_POWER = 30
    # SMOOTHING_POWER = 10 #good for 10 targets

    # SMOOTHING_POWER = 9  #GOOD EXAMPLE for use_group_particles=True, state space of 100

    # SMOOTHING_POWER = 7 #GOOD EXAMPLE for use_group_particles=True

    # SMOOTHING_POWER = 5 #ground truth doesn't have largest log-likelihood
    # SMOOTHING_POWER = 4
    # SMOOTHING_POWER = 3 #ground truth doesn't have largest log-likelihood



    use_constant_size_non_zero_transition_probs = True
    non_zero_transition_state_count = 3
    # non_zero_transition_probs = np.array([.98, .015, .005])
    non_zero_transition_probs = np.array([.999, .0009, .0001])

    MAKE_TRANSITION_PROBS_SIMILAR = True
    if MAKE_TRANSITION_PROBS_SIMILAR: 
        #iterate over each state and create a transition distribution that sums to 1 over all next_states
        for cur_state_index in np.ndindex(*PREVIOUS_DEPENDENT_STATES_SHAPE):
            if use_constant_size_non_zero_transition_probs:
                non_zero_transition_states1d = np.random.choice(np.prod(STATE_SPACE_TUPLE), non_zero_transition_state_count, replace=False)
                for t_idx, cur_non_zero_transition_states1d in enumerate(non_zero_transition_states1d):
                    cur_non_zero_transition_state_full_dimension = np.unravel_index(cur_non_zero_transition_states1d, STATE_SPACE_TUPLE)        
                    #note, + used to concatenate tuples
                    BASE_TRANSITION_PROBABILITIES[cur_state_index + cur_non_zero_transition_state_full_dimension] = non_zero_transition_probs[t_idx]

            else:
                mask = np.random.binomial(n=1, p=1.0, size=STATE_SPACE_TUPLE)
                BASE_TRANSITION_PROBABILITIES[cur_state_index] = np.random.rand(*STATE_SPACE_TUPLE) * mask
                BASE_TRANSITION_PROBABILITIES[cur_state_index] = np.power(BASE_TRANSITION_PROBABILITIES[cur_state_index], SMOOTHING_POWER)
                BASE_TRANSITION_PROBABILITIES[cur_state_index] /= np.sum(BASE_TRANSITION_PROBABILITIES[cur_state_index])

        TRANSITION_PROBABILITIES = []
        for i in range(num_targets):
            cur_noise_transition_probs = np.random.rand(*STATE_SPACE_TUPLE)
            cur_noise_transition_probs /= np.sum(cur_noise_transition_probs)
            TRANSITION_PROBABILITIES.append(1.0*BASE_TRANSITION_PROBABILITIES + 0.0*cur_noise_transition_probs)
    else:
        TRANSITION_PROBABILITIES = []
        for i in range(num_targets):
            CUR_TRANSITION_PROBABILITIES = np.zeros(transition_probs_shape)

            #iterate over each state and create a transition distribution that sums to 1 over all next_states
            for cur_state_index in np.ndindex(*PREVIOUS_DEPENDENT_STATES_SHAPE):
                if use_constant_size_non_zero_transition_probs:
                    non_zero_transition_states1d = np.random.choice(np.prod(STATE_SPACE_TUPLE), non_zero_transition_state_count, replace=False)
                    for t_idx, cur_non_zero_transition_states1d in enumerate(non_zero_transition_states1d):
                        cur_non_zero_transition_state_full_dimension = np.unravel_index(cur_non_zero_transition_states1d, STATE_SPACE_TUPLE)        
                        #note, + used to concatenate tuples
                        CUR_TRANSITION_PROBABILITIES[cur_state_index + cur_non_zero_transition_state_full_dimension] = non_zero_transition_probs[t_idx]
                else:
                    mask = np.random.binomial(n=1, p=1.0, size=STATE_SPACE_TUPLE)
                    CUR_TRANSITION_PROBABILITIES[cur_state_index] = np.random.rand(*STATE_SPACE_TUPLE) * mask
                    CUR_TRANSITION_PROBABILITIES[cur_state_index] = np.power(CUR_TRANSITION_PROBABILITIES[cur_state_index], SMOOTHING_POWER)
                    CUR_TRANSITION_PROBABILITIES[cur_state_index] /= np.sum(CUR_TRANSITION_PROBABILITIES[cur_state_index])

            TRANSITION_PROBABILITIES.append(CUR_TRANSITION_PROBABILITIES)

    emission_probs_shape = list(STATE_SPACE_TUPLE)
    emission_probs_shape.extend(list(MEASUREMENT_SPACE_TUPLE))
    emission_probs_shape = tuple(emission_probs_shape)
        
    BASE_EMISSION_PROBABILITIES = np.zeros(emission_probs_shape)

    use_identical_emission_probs = False #use the same emission probabilities for every state for easy visualization, only implemented for 1-d state space
    add_constant_emission_probs = False
    #iterate over each state and create an emission distribution that sums to 1 over all measurements
    # for cur_state_index in np.ndindex(*STATE_SPACE_TUPLE):
    for cur_state_index in np.ndindex(*MEASUREMENT_SPACE_TUPLE):
        if use_identical_emission_probs:
    #         constant_emission_probs = np.array([.1, .2, .4, .2, .1])
    #         constant_emission_probs = np.array([.025, .075, .1, .15, .3, .15, .1, .075, .025])
    #         constant_emission_probs = np.array([1 for i in range(5)])
    #         constant_emission_probs = constant_emission_probs/np.sum(constant_emission_probs)
            constant_emission_probs = np.array([.005, .02, .95, .02, .005])
    #         constant_emission_probs = np.array([.01, .02, .07, .8, .07, .02, .01])

            for emission_idx in range(len(constant_emission_probs)):
                BASE_EMISSION_PROBABILITIES[cur_state_index, (cur_state_index[0] + emission_idx - len(constant_emission_probs)//2)%STATE_SPACE_TUPLE[0]] = constant_emission_probs[emission_idx]
        else:
            # mask = np.random.binomial(n=1, p=.3, size=MEASUREMENT_SPACE_TUPLE)    
            # BASE_EMISSION_PROBABILITIES[:, cur_state_index] = np.random.rand(*MEASUREMENT_SPACE_TUPLE) * mask
            # BASE_EMISSION_PROBABILITIES[:, cur_state_index] = np.power(BASE_EMISSION_PROBABILITIES[:, cur_state_index], SMOOTHING_POWER)
            # if add_constant_emission_probs:
            #     constant_emission_probs = 1*np.array([.03, .07, .8, .07, .03])
            #     for emission_idx in range(len(constant_emission_probs)):
            #         BASE_EMISSION_PROBABILITIES[cur_state_index, (cur_state_index[0] + emission_idx - len(constant_emission_probs)//2)%STATE_SPACE_TUPLE[0]] += constant_emission_probs[emission_idx]
            
            # BASE_EMISSION_PROBABILITIES[:, cur_state_index] /= np.sum(BASE_EMISSION_PROBABILITIES[:, cur_state_index])

            mask = np.random.binomial(n=1, p=.3, size=MEASUREMENT_SPACE_TUPLE)    
            cur_emission_probs = np.random.rand(*MEASUREMENT_SPACE_TUPLE) * mask
            cur_emission_probs = np.power(cur_emission_probs, SMOOTHING_POWER)          
            cur_emission_probs /= np.sum(cur_emission_probs)

            if MEASUREMENT_SPACE_TUPLE == STATE_SPACE_TUPLE:
                BASE_EMISSION_PROBABILITIES[cur_state_index] = cur_emission_probs
            else:
                BASE_EMISSION_PROBABILITIES[:, cur_state_index] = cur_emission_probs
            # for cur_unobserved_state_index in np.ndindex(*MEASUREMENT_SPACE_TUPLE):
               #  cur_unobserved_state_emission_noise = np.random.rand(*MEASUREMENT_SPACE_TUPLE) * mask
               #  cur_unobserved_state_emission_noise /= np.sum(cur_unobserved_state_emission_noise)*10
               #  cur_emission_probs_plus_noise = cur_emission_probs + cur_unobserved_state_emission_noise
               #  cur_emission_probs_plus_noise /= np.sum(cur_emission_probs_plus_noise)
               #  BASE_EMISSION_PROBABILITIES[cur_unobserved_state_index, cur_state_index] = cur_emission_probs_plus_noise

    print "BASE_EMISSION_PROBABILITIES[0]:", BASE_EMISSION_PROBABILITIES[0]

    EMISSION_PROBABILITIES = []
    for i in range(num_targets):
        cur_emission_probs = np.random.rand(*MEASUREMENT_SPACE_TUPLE)
        cur_emission_probs = np.power(cur_emission_probs, SMOOTHING_POWER)          
        cur_emission_probs /= np.sum(cur_emission_probs)
        EMISSION_PROBABILITIES.append(1.0*BASE_EMISSION_PROBABILITIES + 0.0*cur_emission_probs)
        
    #generate the actual data
    gen_params = GenerativeParameters(num_time_steps=num_time_steps, state_space=state_space, \
        previous_dependent_states_shape=PREVIOUS_DEPENDENT_STATES_SHAPE, all_initial_state_probabilities=ALL_INITIAL_STATE_PROBABILITIES, \
        transition_probabilities=TRANSITION_PROBABILITIES, emission_probabilities=EMISSION_PROBABILITIES, markov_order=markov_order, \
        num_targets=num_targets)
    (all_states, all_measurements) = generate_all_target_data(gen_params)

    return (all_states, all_measurements, gen_params)




def get_parameters_and_data1(num_time_steps, state_space, measurement_space, num_targets):
    '''
    High level function that generates data and the parameters used to generate the data
    Inputs:
    - num_time_steps: (int) the number of time steps in the time series
    - state_space: (np.array) [20,20,20] -> 3d state space of size 20x20
    - measurement_space: (np.array) [20,20] -> 2d measurement space of size 20x20
    - markov_order: (int) the number of previous states that determines the transition distribution

    Outputs:
    - all_states: (list of lists) all_states[i][j] is the ith target's state at the jth time instance
    - all_measurements: (list of lists) all_measurements[i][j] is the ith target's measurement at the jth time instance
    - gen_params: (GenerativeParameters) the parameters used to generate the data

    '''
    markov_order = 1

    ######################## DEFINE DATA GENERATION PARAMETERS ########################
    #1-d np.array's have length 0 which causes some problems below without STATE_SPACE_TUPLE
    #tuple specifying the number of dimensions in the state space and the size of each dimension
    if state_space.size > 1: #does state_space have more than one dimension? 
        STATE_SPACE_TUPLE = tuple(state_space)
    else:
        STATE_SPACE_TUPLE = (state_space,)
    #1-d np.array's have length 0 which causes some problems below without MEASUREMENT_SPACE_TUPLE
    #tuple specifying the number of dimensions in the state space and the size of each dimension
    if measurement_space.size > 1: #does measurement_space have more than one dimension? 
        MEASUREMENT_SPACE_TUPLE = tuple(measurement_space)
    else:
        MEASUREMENT_SPACE_TUPLE = (measurement_space,)


    #TRANSITION_PROBABILITIES has shape (markov_order+1)*state_space,
    #that is the transition function's domain is over the previous markov_order states
    #and the range is over the state space
    PREVIOUS_DEPENDENT_STATES_SHAPE = []
    transition_probs_shape = []
    for i in range(markov_order):
        PREVIOUS_DEPENDENT_STATES_SHAPE.extend(list(STATE_SPACE_TUPLE))
        transition_probs_shape.extend(list(STATE_SPACE_TUPLE))

    PREVIOUS_DEPENDENT_STATES_SHAPE = tuple(PREVIOUS_DEPENDENT_STATES_SHAPE)
    transition_probs_shape.extend(list(STATE_SPACE_TUPLE))
    transition_probs_shape = tuple(transition_probs_shape)

    # INITIAL_STATE_PROBABILITIES = np.random.rand(*STATE_SPACE_TUPLE)
    # INITIAL_STATE_PROBABILITIES = np.ones(STATE_SPACE_TUPLE)
    # INITIAL_STATE_PROBABILITIES = np.power(INITIAL_STATE_PROBABILITIES, 1)
    # INITIAL_STATE_PROBABILITIES /= np.sum(INITIAL_STATE_PROBABILITIES)

    ALL_INITIAL_STATE_PROBABILITIES = []


    for t_idx in range(num_targets):
        INITIAL_STATE_PROBABILITIES = np.zeros(STATE_SPACE_TUPLE)
        for cur_state_index in np.ndindex(*MEASUREMENT_SPACE_TUPLE):
            mask = np.random.binomial(n=1, p=1, size=MEASUREMENT_SPACE_TUPLE + (1,))    
            cur_initial_state_probs = np.random.rand(*MEASUREMENT_SPACE_TUPLE+(1,)) * mask
            cur_initial_state_probs = np.power(cur_initial_state_probs, 10.0)            
            # cur_initial_state_probs /= np.sum(cur_initial_state_probs)
            # print "INITIAL_STATE_PROBABILITIES.shape:", INITIAL_STATE_PROBABILITIES.shape
            # print "INITIAL_STATE_PROBABILITIES[:, cur_state_index].shape:", INITIAL_STATE_PROBABILITIES[:, cur_state_index].shape
            # print "cur_initial_state_probs.shape:", cur_initial_state_probs.shape
            INITIAL_STATE_PROBABILITIES[:, cur_state_index] = cur_initial_state_probs
        INITIAL_STATE_PROBABILITIES /= np.sum(INITIAL_STATE_PROBABILITIES)
        ALL_INITIAL_STATE_PROBABILITIES.append(INITIAL_STATE_PROBABILITIES)




    TRANSITION_PROBABILITIES = np.zeros(transition_probs_shape)

    SMOOTHING_POWER = 5


    non_zero_transition_state_count = 3
    # non_zero_transition_probs = np.array([.98, .015, .005])
    non_zero_transition_probs = np.array([.999, .0009, .0001])

    print "TRANSITION_PROBABILITIES.shape", TRANSITION_PROBABILITIES.shape
    #iterate over each state and create a transition distribution that sums to 1 over all next_states
    for cur_state_index in np.ndindex(STATE_SPACE_TUPLE[-1]):
        print "cur_state_index:", cur_state_index
        non_zero_transition_states1d = np.random.choice(np.prod(STATE_SPACE_TUPLE), non_zero_transition_state_count, replace=False)
        print "non_zero_transition_states1d:", non_zero_transition_states1d
        for t_idx, cur_non_zero_transition_states1d in enumerate(non_zero_transition_states1d):
            cur_non_zero_transition_state_full_dimension = np.unravel_index(cur_non_zero_transition_states1d, STATE_SPACE_TUPLE)        
            #note, + used to concatenate tuples
            # print "cur_state_index + cur_non_zero_transition_state_full_dimension:", cur_state_index + cur_non_zero_transition_state_full_dimension
            TRANSITION_PROBABILITIES[(slice(None),) + cur_state_index + cur_non_zero_transition_state_full_dimension] = non_zero_transition_probs[t_idx]
            # print "TRANSITION_PROBABILITIES:", TRANSITION_PROBABILITIES
            # print "TRANSITION_PROBABILITIES[cur_state_index + cur_non_zero_transition_state_full_dimension]:", TRANSITION_PROBABILITIES[cur_state_index + cur_non_zero_transition_state_full_dimension]
            # print "TRANSITION_PROBABILITIES[(slice(None),) + cur_state_index + cur_non_zero_transition_state_full_dimension]:", TRANSITION_PROBABILITIES[(slice(None),) + cur_state_index + cur_non_zero_transition_state_full_dimension]

    # print "TRANSITION_PROBABILITIES:", TRANSITION_PROBABILITIES
    TRANSITION_PROBABILITIES = [TRANSITION_PROBABILITIES for i in range(num_targets)]

    emission_probs_shape = list(STATE_SPACE_TUPLE)
    emission_probs_shape.extend(list(MEASUREMENT_SPACE_TUPLE))
    emission_probs_shape = tuple(emission_probs_shape)
        
    EMISSION_PROBABILITIES = np.zeros(emission_probs_shape)

    use_identical_emission_probs = False #use the same emission probabilities for every state for easy visualization, only implemented for 1-d state space
    add_constant_emission_probs = False
    #iterate over each state and create an emission distribution that sums to 1 over all measurements
    # for cur_state_index in np.ndindex(*STATE_SPACE_TUPLE):
    for cur_state_index in np.ndindex(*MEASUREMENT_SPACE_TUPLE):
        # mask = np.random.binomial(n=1, p=.3, size=MEASUREMENT_SPACE_TUPLE)    
        # EMISSION_PROBABILITIES[:, cur_state_index] = np.random.rand(*MEASUREMENT_SPACE_TUPLE) * mask
        # EMISSION_PROBABILITIES[:, cur_state_index] = np.power(EMISSION_PROBABILITIES[:, cur_state_index], SMOOTHING_POWER)
        # if add_constant_emission_probs:
        #     constant_emission_probs = 1*np.array([.03, .07, .8, .07, .03])
        #     for emission_idx in range(len(constant_emission_probs)):
        #         EMISSION_PROBABILITIES[cur_state_index, (cur_state_index[0] + emission_idx - len(constant_emission_probs)//2)%STATE_SPACE_TUPLE[0]] += constant_emission_probs[emission_idx]
        
        # EMISSION_PROBABILITIES[:, cur_state_index] /= np.sum(EMISSION_PROBABILITIES[:, cur_state_index])

        mask = np.random.binomial(n=1, p=.3, size=MEASUREMENT_SPACE_TUPLE)    
        cur_emission_probs = np.random.rand(*MEASUREMENT_SPACE_TUPLE) * mask
        cur_emission_probs = np.power(cur_emission_probs, SMOOTHING_POWER)          
        cur_emission_probs /= np.sum(cur_emission_probs)

        EMISSION_PROBABILITIES[:, cur_state_index] = cur_emission_probs
        # for cur_unobserved_state_index in np.ndindex(*MEASUREMENT_SPACE_TUPLE):
           #  cur_unobserved_state_emission_noise = np.random.rand(*MEASUREMENT_SPACE_TUPLE) * mask
           #  cur_unobserved_state_emission_noise /= np.sum(cur_unobserved_state_emission_noise)*10
           #  cur_emission_probs_plus_noise = cur_emission_probs + cur_unobserved_state_emission_noise
           #  cur_emission_probs_plus_noise /= np.sum(cur_emission_probs_plus_noise)
           #  EMISSION_PROBABILITIES[cur_unobserved_state_index, cur_state_index] = cur_emission_probs_plus_noise

    print "EMISSION_PROBABILITIES[0]:", EMISSION_PROBABILITIES[0]
    EMISSION_PROBABILITIES = [EMISSION_PROBABILITIES for i in range(num_targets)]

    #generate the actual data
    gen_params = GenerativeParameters(num_time_steps=num_time_steps, state_space=state_space, \
        previous_dependent_states_shape=PREVIOUS_DEPENDENT_STATES_SHAPE, all_initial_state_probabilities=ALL_INITIAL_STATE_PROBABILITIES, \
        transition_probabilities=TRANSITION_PROBABILITIES, emission_probabilities=EMISSION_PROBABILITIES, markov_order=markov_order, \
        num_targets=num_targets)
    (all_states, all_measurements) = generate_all_target_data(gen_params)

    return (all_states, all_measurements, gen_params)




def plot_generated_data():
    ######################## GENERATE DATA ########################
    xs, zs = generate_data(NUM_TIME_STEPS, STATE_SPACE, INITIAL_STATE_PROBABILITIES,\
                         TRANSITION_PROBABILITIES, EMISSION_PROBABILITIES)


    ######################## PLOT DATA ########################
    print(xs)
    plt.plot(xs, label='states')
    plt.plot(zs, label='measurements')
#     plt.ylabel('some numbers')
    plt.show()

def sample_from_multi_dimensional_array(p):
    '''
    Inputs:
    - p: (np.array) the probability distribution, sum over all elements must be 1
    
    Outputs:
    - indices: (tuple) indices of the element in p we sampled
    '''
    original_array_shape = p.shape
    flattened_p = p.flatten()
    index_in_1d = np.random.choice(a=len(flattened_p), p=flattened_p)
    indices = np.unravel_index(index_in_1d, p.shape)
    return indices

def test_sample_from_multi_dimensional_array():
    #Test sample_from_multi_dimensional_array
    test_shape = (2,2,2)
    p = np.random.rand(*test_shape) #arguments are elements of the tuple using *
    p = p/np.sum(p)
    # print(p)
    sample_from_multi_dimensional_array(p)

    empirical_p = np.zeros(test_shape)
    T = 10000
    for i in range(T):
        empirical_p[sample_from_multi_dimensional_array(p)] += 1/T
    # print(empirical_p)
    assert(np.allclose(p, empirical_p,  rtol=1e-01))




def generate_single_target_data(gen_params, target_idx):
    '''
    Inputs:
    - gen_params: (GenerativeParameters) the parameters used to generate the data
    - target_idx: (int) the index of the target whose data we are generating (initial state probabilities
                  differ between targets)
    
    Outputs:
    - xs: (list of length gen_params.num_time_steps) the true states of the system, xs[i] is an np.array
          representing the state at time i
    - zs: (list of length gen_params.num_time_steps) the measurements, zs[i] is an np.array
          representing the measurement at time i
    '''
    xs = []
    zs = []
    
    #sample the initial state and measurement
    assert(np.allclose(np.sum(gen_params.all_initial_state_probabilities[target_idx]), 1.0)), (np.sum(gen_params.all_initial_state_probabilities[target_idx]), gen_params.all_initial_state_probabilities[target_idx])
    x1 = sample_from_multi_dimensional_array(gen_params.all_initial_state_probabilities[target_idx])
    xs.append(x1)

    assert(np.allclose(np.sum(gen_params.emission_probabilities[target_idx][x1]), 1.0)), gen_params.emission_probabilities[target_idx][x1]    
    z1 = sample_from_multi_dimensional_array(gen_params.emission_probabilities[target_idx][x1])
    zs.append(z1)

    state_queue = deque([x1 for i in range(gen_params.markov_order)])

    
    #sample the remaining states and measurements
    for t_index in range(1, gen_params.num_time_steps):
#         x_t = sample_from_multi_dimensional_array(gen_params.transition_probabilities[target_idx][xs[-1]])
#         print("tuple(state_queue):", tuple(state_queue))    
#         print("gen_params.transition_probabilities[target_idx][tuple(state_queue)]:", gen_params.transition_probabilities[target_idx][tuple(state_queue)])

        state_queue_tupleOfTuples = tuple(state_queue)
        state_queue_single_tuple = tuple([element for tupl in state_queue_tupleOfTuples for element in tupl])
        x_t = sample_from_multi_dimensional_array(gen_params.transition_probabilities[target_idx][state_queue_single_tuple])
        state_queue.popleft()
        state_queue.append(x_t)
        xs.append(x_t)
        
#         print("x_t:", x_t)    
#         print("gen_params.emission_probabilities[target_idx][x_t]:", gen_params.emission_probabilities[target_idx][x_t])

        
        z_t = sample_from_multi_dimensional_array(gen_params.emission_probabilities[target_idx][x_t])
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

