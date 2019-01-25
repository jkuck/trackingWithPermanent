from __future__ import division
import sys
import numpy as np
import math
from pymatgen.optimization import linear_assignment
from IPython.utils import io
from scipy.stats import multivariate_normal
from numpy.linalg import inv as matrix_inverse
import matplotlib.pyplot as plt
import copy
from filterpy.monte_carlo import stratified_resample

from permanent import permanent as rysers_permanent

import generate_data

RBPF_HOME_DIRECTORY = "/Users/jkuck/tracking_research/rbpf_fireworks/"
sys.path.insert(0, RBPF_HOME_DIRECTORY)
from rbpf_sampling_manyMeasSrcs import construct_log_probs_matrix3, convert_assignment_pairs_to_matrix3, convert_assignment_matrix3, construct_log_probs_matrix4

sys.path.insert(0, "%smht_helpers" % RBPF_HOME_DIRECTORY)
from k_best_assign_birth_clutter_death_matrix import k_best_assign_mult_cost_matrices

sys.path.insert(0, '/Users/jkuck/research/gumbel_sample_permanent')
# from tracking_specific_nestingUB_gumbel_sample_permanent import associationMatrix, multi_matrix_sample_associations_without_replacement
from constant_num_targets_sample_permenant import associationMatrix, multi_matrix_sample_associations_without_replacement, single_matrix_sample_associations_with_replacement

# N_PARTICLES = 5

class TargetStateNode:
    '''
    Represents the state of a target at one instance in time
    TargetStateNode's are linked to form a tree

    The target state is a 1d position, velocity and acceleration.  Acceleration
    is applied by a spring (Force = -K*x, mass = 1)

    state X = [x] position
              [v] velocity
              [a] acceleration
    '''
    def __init__(self, gen_params, parent_node=None, debug_info=None, target_idx=None):
        '''
        Inputs:
        - parent_node: (TargetStateNode) this node's parent
        - gen_params: (GenerativeParameters) object storing parameters used for data generation and inference
        - target_idx: (int) specifies initial state, emission, and transition probabilities for this target
        '''  

        #if the root node (first target state in the track) None, otherwise the previous target state (in time)
        self.parent_node = parent_node

        #TargetStateNode's that are children of this node will be added to this dictionary with:
        #   key: (int) association of the child target node 
        #       -1: clutter
        #       0 to (M-1): specifies the index of the measurement this child node is assocated with (M = #measurements)
        #   value: (TargetStateNode) the child TargetStateNode
        self.child_nodes = {}

        #posterior distribution of the target's state for the current time step after updating with its associated measurement
        #tuple of (X, P) with
        #   X: np.array of state means
        #   P: np.array of state covariance
        self.posterior =  None

        #prior distribution of the target's predicted state at the next time step
        #tuple of (X, P) with
        #   X: np.array of state means
        #   P: np.array of state covariance        
        self.prior = None

        self.target_idx = target_idx

        self.gen_params = gen_params
        self.debug_info = debug_info

        #used when different targets have different spring constants
        self.spring_constant = gen_params.spring_constants[target_idx]

    def print_measurement_associations(self, leaf_node = False):
        if leaf_node:
            # print 'posterior:'
            # print self.posterior
            # print 'likelihood:'
            # print self.debug_info['likelihood']
            # print 'prior:'
            # print self.debug_info['prior']
            print 'measurement indices associated with this target in reverse order:'
            print self.debug_info['measurement'],
        else:
            print self.debug_info['measurement'],
        if self.parent_node.parent_node is None:
            print
        else:
            self.parent_node.print_measurement_associations()

    def get_associated_measurements(self, leaf_node = False):
        '''
        Outputs:
        - associated_measurements: (list of measurements) associated_measurements[i] is the ith measurment associated with this
                                   target trajectory
        '''
        if self.parent_node.parent_node is None:
            return [self.debug_info['measurement']]
        else:
            parent_measurements = self.parent_node.get_associated_measurements()
            parent_measurements.append(self.debug_info['measurement'])
            return parent_measurements

    def get_associated_measurement_indices(self, leaf_node = False):
        '''
        Outputs:
        - associated_measurements: (list of measurements) associated_measurements[i] is the ith measurment associated with this
                                   target trajectory
        '''
        if self.parent_node.parent_node is None:
            return [self.debug_info['meas_idx']]
        else:
            parent_measurements = self.parent_node.get_associated_measurement_indices()
            parent_measurements.append(self.debug_info['meas_idx'])
            return parent_measurements


    def initialize_root_target_initialAssociationKnown(self, measurement, target_idx):
        '''

        '''
        assert(len(measurement) == 2)
        posterior_X = np.array([[                   measurement[0]],
                            [                       measurement[1]],
                            [-self.spring_constant*measurement[0]]])
        posterior_P = np.array([[self.gen_params.measurement_variance,                    0,                                  0],
                                [                   0, self.gen_params.initial_vel_variance,                                  0],
                                [                   0,                    0, (self.spring_constant**2)*self.gen_params.measurement_variance]]) 
           
        self.posterior = (posterior_X, posterior_P)
        self.prior = self.kf_predict()

    def initialize_root_target_initialAssociationNotKnown(self, target_idx):
        '''
        set the prior and posterior distributions of a root target (created on this time step) with only a prior 
        (when the initial association is unknown)
        '''
        position_mean = self.gen_params.initial_position_means[target_idx]
        velocity_mean = self.gen_params.initial_velocity_means[target_idx]
        initial_position_variance = self.gen_params.initial_position_variance
        initial_vel_variance = self.gen_params.initial_vel_variance
        prior_X = np.array([[                      position_mean],
                            [                      velocity_mean],
                            [-self.spring_constant*position_mean]])
        prior_P = np.array([[initial_position_variance,                    0,                                       0],
                            [                        0, initial_vel_variance,                                       0],
                            [                        0,                    0, (self.spring_constant**2)*initial_position_variance]]) 

        self.posterior = None
        self.prior = (prior_X, prior_P)


    def create_child_target(self, meas_idx, measurement):
        '''
        Update with a measurement from the next time step
        Inputs:
        - meas_idx: (int) index of the measurement, NOTE: should always be the same for this measurement
        - measurement: the measurement

        Outputs:
        - updated_target_state_node: (TargetStateNode) this target state updated with the measurement, a 
                                     child of this node
        '''
        if meas_idx in self.child_nodes:
            return self.child_nodes[meas_idx]
        else:
            new_child_node = TargetStateNode(gen_params=self.gen_params, parent_node=self, debug_info={'meas_idx':meas_idx, 'measurement':measurement},\
                                             target_idx=self.target_idx)
            new_child_node.posterior = self.kf_update(measurement)
            new_child_node.debug_info['prior'] = self.prior
            new_child_node.prior = new_child_node.kf_predict()
            self.child_nodes[meas_idx] = new_child_node
            return new_child_node

    def get_posteriors(self):
        '''
        Output:
        - posteriors: (list of np.array's) posteriors[j] is the posterior distribution of the 
                      target's state at the jth time step

        '''
        if INITIAL_ASSOCIATIONS_KNOWN and self.parent_node is None:
            posteriors = [self.posterior[0][0]]
        elif not INITIAL_ASSOCIATIONS_KNOWN and self.parent_node.parent_node is None:
            posteriors = [self.posterior[0][0]]
        else:
            posteriors = self.parent_node.get_posteriors()   
            posteriors.append(self.posterior[0][0])
                    
        return posteriors

    def get_priors(self):
        '''
        Output:
        - priors: (list of np.array's) priors[j] is the prior distribution of the 
                      target's state at the jth time step

        '''
        if self.parent_node is None:
            priors = [self.prior[0][0]]
        else:
            priors = self.parent_node.get_priors()
            priors.append(self.prior[0][0])         
        return priors


    def kf_predict(self):
        """
        Run kalman filter prediction on this target
        Inputs:
            -dt: time step to run prediction on
        Output:
            -x_predict: predicted state, numpy array with dimensions 
            -P_predict: predicted covariance, numpy array with dimensions 

        """
        dt = self.gen_params.dt
        k = self.spring_constant #spring constant, Force = -k*x
        F = np.array([[1.0,    dt,    .5*(dt**2)],
                      [0.0,   1.0,            dt],
                      [ -k, -k*dt, -k*.5*(dt**2)]])
        # F = np.array([[1.0,    dt,    .5*(dt**2)],
        #               [0.0,   1.0,            dt],
        #               [ -k*(1 - k*(dt**2)/2), -k*(2*dt - k*(dt**3)/2), k*(1.5*(dt**2) - .25*k*(dt**4))]])
        self_X = self.posterior[0]
        self_P = self.posterior[1]
        x_predict = np.dot(F, self_X)
        P_predict = np.dot(np.dot(F, self_P), F.T) + GEN_PARAMS.q_matrix
        assert(P_predict[0][0] > 0 and
               P_predict[1][1] > 0 and
               P_predict[2][2] > 0), (self_P, GEN_PARAMS.q_matrix, P_predict[0][0])

        return (x_predict, P_predict)

    def kf_update(self, measurement):
        """ Perform Kalman filter update step
        Input:
            - measurement: the measurement (numpy array)
            - cur_time: time when the measurement was taken (float)
        Output:
            -updated_x: updated state, numpy array with dimensions 
            -updated_P: updated covariance, numpy array with dimensions 

!!!!!!!!!PREDICTION HAS BEEN RUN AT THE BEGINNING OF TIME STEP FOR EVERY TARGET!!!!!!!!!
        """

        measurement = np.expand_dims(measurement, axis=1)

        self_X = self.prior[0]
        self_P = self.prior[1]
        H = self.gen_params.h_matrix


        S = np.dot(np.dot(H, self_P), H.T) + self.gen_params.r_matrix
        K = np.dot(np.dot(self_P, H.T), matrix_inverse(S))
        residual = measurement - np.dot(H, self_X)
        # print '@'*20
        # print "H:", H
        # print "self_P:", self_P
        # print "self.gen_params.r_matrix:", self.gen_params.r_matrix
        # print
        # print "np.dot(H, self_X)", np.dot(H, self_X)
        # print "K.shape:", K.shape
        # print "residual.shape:", residual.shape
        # print "np.dot(H, self_X).shape:", np.dot(H, self_X).shape
        # print "np.dot(K, residual).shape:", np.dot(K, residual).shape
        # print 'measurement:', measurement
        # print "self_X:", self_X
        updated_x = self_X + np.dot(K, residual)
        # print "updated_x:", updated_x
    #   updated_self_P = np.dot((np.eye(self_P.shape[0]) - np.dot(K, H)), self_P) #NUMERICALLY UNSTABLE!!!!!!!!
        updated_P = self_P - np.dot(np.dot(K, S), K.T) #not sure if this is numerically stable!!
        assert(updated_P[0][0] > 0 and
               updated_P[1][1] > 0 and
               updated_P[2][2] > 0), (self_P, K, updated_P)
        return (updated_x, updated_P)


def normalize(pdf):
    pdf /= np.sum(pdf)
    return pdf


class Particle:
    def __init__(self, parent_particle, generative_parameters, log_importance_weight=None, log_importance_weight_normalization=0.0,\
                 log_importance_weight_debug=0.0):
        '''
        Represents a sample of target states at one time step
        Inputs:
        - parent_particle: (Particle) this particle's parent
        - generative_parameters: (GenerativeParameters) object storing parameters used for data generation and inference
        '''
        self.parent_particle = parent_particle

        #list of Particles that are children of this particle
        #each of the child particles had the same state at the previous time step (represented by this particle)
        #but differ in state at their time step
        self.child_particles = []

        #list of TargetStateNode's
        self.targets = []

        if log_importance_weight is None:
            self.log_importance_weight = np.log(1.0/N_PARTICLES)
        else:
            self.log_importance_weight = log_importance_weight
        self.importance_weight = np.exp(log_importance_weight)

        #log of the product of each importance weight normalization 
        self.log_importance_weight_normalization = log_importance_weight_normalization

        #GenerativeParameters object storing parameters used for data generation and inference
        self.generative_parameters = generative_parameters

        self.log_importance_weight_debug = log_importance_weight_debug
        self.normalized_importance_weight_debug = 1/N_PARTICLES

    def create_child_particle(self, log_child_imprt_weight, measurements, measurement_associations):
        '''
        Inputs:
        - log_child_imprt_weight: (float) the logarithm of the importance weight of the child particle
        - measurements: (list of measurements) contains measurements to update target states with
                        NOTE: this list of measurements should be passed in the same order to every
                        particle because particles may share targets and the measurement indices 
                        must be consistent for TargetStateNode updates
        - measurement_associations: (list of ints) measurement_associations[i] = j specifies that
            specifies that the ith measurement in measurements should be used to update the jth
            target in self.targets
        '''
        child_particle = Particle(parent_particle=self, generative_parameters=self.generative_parameters, log_importance_weight=log_child_imprt_weight,\
                                  log_importance_weight_normalization=self.log_importance_weight_normalization, log_importance_weight_debug=self.log_importance_weight_debug)
        assert(len(measurements) == len(measurement_associations)), (measurements, measurement_associations)
        ITERATE_THROUGH_TARGETS_TO_PRESERVE_ORDER = True
        if ITERATE_THROUGH_TARGETS_TO_PRESERVE_ORDER: #for initial tests with all targets emitting, we guarantee plot colors match initially this way
            assert(len(self.targets) == len(measurements))
            for target_idx in range(len(self.targets)):
                meas_idx = measurement_associations.index(target_idx)
                cur_meas = measurements[meas_idx]
                updated_target_state_node = self.targets[target_idx].create_child_target(meas_idx=meas_idx, measurement=cur_meas)
                child_particle.targets.append(updated_target_state_node)

        else:    
            for meas_idx in range(len(measurements)):
                cur_meas = measurements[meas_idx]
                target_idx = measurement_associations[meas_idx]
                #no clutter/births implemented currently
                assert(target_idx >= 0 and target_idx < len(self.targets)), (target_idx, target_idx)
                if target_idx >= 0 and target_idx < len(self.targets):
                    updated_target_state_node = self.targets[target_idx].create_child_target(meas_idx=meas_idx, measurement=cur_meas)
                    child_particle.targets.append(updated_target_state_node)

        assert(len(child_particle.targets) == len(self.targets)), "currently assume all targets should be associated with a measurement, didn't happen!!"
        return child_particle

    def get_log_likelihoods_over_time(self):
        '''
        Output:
        - log_likelihoods: (list floats) log_likelihoods[i] is the log-likelihood of this particle at time-step i

        '''      
        log_likelihoods = []

        if INITIAL_ASSOCIATIONS_KNOWN and self.parent_particle is None:
            log_likelihoods = [self.log_importance_weight_normalization + np.log(self.importance_weight)]
            # log_likelihoods = [self.normalized_importance_weight_debug]
        elif not INITIAL_ASSOCIATIONS_KNOWN and self.parent_particle.parent_particle is None:
            log_likelihoods = [self.log_importance_weight_normalization + np.log(self.importance_weight)]
            # log_likelihoods = [self.normalized_importance_weight_debug]

        # if self.parent_particle is None:
        #     log_likelihoods = [self.log_importance_weight_normalization + np.log(self.importance_weight)]
        else:
            log_likelihoods = self.parent_particle.get_log_likelihoods_over_time()   
            log_likelihoods.append(self.log_importance_weight_normalization + np.log(self.importance_weight))
            # log_likelihoods.append(self.normalized_importance_weight_debug)
                    
        return log_likelihoods



    def get_all_target_posteriors(self):
        '''
        Output:
        - target_posteriors: (list of list of np.array's) target_posteriors[i][j] is the posterior distribution of the 
                             ith target's state at the jth time step

        '''      
        target_posteriors = []
        for target in self.targets:
            cur_target_posteriors = target.get_posteriors()
            target_posteriors.append(cur_target_posteriors)
        return target_posteriors

    def get_all_target_priors(self):
        '''
        Output:
        - target_priors: (list of list of np.array's) target_priors[i][j] is the prior distribution of the 
                             ith target's state at the jth time step

        '''      
        target_priors = []
        for target in self.targets:
            cur_target_priors = target.get_priors()
            target_priors.append(cur_target_priors)
        return target_priors        

def gt_assoc_step(particle_set, cur_time_step_measurements):
    '''
    perform exact sampling without replacement using upper bounds on the permanent
    '''
    M = len(cur_time_step_measurements) #number of measurements
    T = M #currently using a fixed number of targets that always emit measurements
    all_association_matrices = []

    for particle in particle_set:

        # probs = construct_exact_sampling_matrix(targets=particle.targets, measurements=cur_time_step_measurements)
        probs = construct_exact_sampling_matrix_constantNumTargets(targets=particle.targets, measurements=cur_time_step_measurements)        
        print 'original probs:'
        print probs
        for m_idx in range(M):
            for t_idx in range(T):
                if m_idx != t_idx:
                    probs[m_idx][t_idx] = 0
        print 'zeroed probs:'
        print probs

        particle_prior_prob = particle.importance_weight

        print "particle_prior_prob =", particle_prior_prob
        cur_a_matrix = associationMatrix(matrix=probs, M=M, T=T,\
            conditional_birth_probs=[0.0 for i in range(M)], conditional_death_probs=[0.0 for i in range(T)],\
            prior_prob=particle_prior_prob)
        all_association_matrices.append(cur_a_matrix)

    sampled_associations = multi_matrix_sample_associations_without_replacement(num_samples=N_PARTICLES, all_association_matrices=all_association_matrices)


    new_particle_set = []
    for sampled_association in sampled_associations:
        # print "sampled_association.meas_grp_associations:", sampled_association.meas_grp_associations
        for target_idx in range(T):
            assert(sampled_association.meas_grp_associations[target_idx] == target_idx), sampled_association.meas_grp_associations
        child_particle = particle_set[sampled_association.matrix_index].create_child_particle(\
            log_child_imprt_weight=np.log(sampled_association.complete_assoc_probability),\
            measurements=cur_time_step_measurements,\
            measurement_associations=sampled_association.meas_grp_associations)

        assert(T == len(child_particle.targets))
          
        new_particle_set.append(child_particle)


    return new_particle_set

def exact_sampling_step(particle_set, cur_time_step_measurements):
    '''
    perform exact sampling without replacement using upper bounds on the permanent
    '''
    M = len(cur_time_step_measurements) #number of measurements
    T = M #currently using a fixed number of targets that always emit measurements
    all_association_matrices = []

    for particle in particle_set:

        # probs = construct_exact_sampling_matrix(targets=particle.targets, measurements=cur_time_step_measurements)
        probs = construct_exact_sampling_matrix_constantNumTargets(targets=particle.targets, measurements=cur_time_step_measurements)
        particle_prior_prob = particle.importance_weight

        # try to help numerical issues, might not help
        for row in range(probs.shape[0]):
            max_row_val = np.max(probs[row])
            if max_row_val > 0.0:
                probs[row] /= max_row_val
                particle_prior_prob *= max_row_val
        for col in range(probs.shape[1]):
            max_col_val = np.max(probs[:,col])
            if max_col_val > 0.0:
                probs[:,col] /= max_col_val
                particle_prior_prob *= max_col_val            

        # print "probs:"
        # print probs
        # print "particle_prior_prob =", particle_prior_prob
        cur_a_matrix = associationMatrix(matrix=probs, M=M, T=T,\
            conditional_birth_probs=[0.0 for i in range(M)], conditional_death_probs=[0.0 for i in range(T)],\
            prior_prob=particle_prior_prob)
        all_association_matrices.append(cur_a_matrix)

    sampled_associations = multi_matrix_sample_associations_without_replacement(num_samples=N_PARTICLES, all_association_matrices=all_association_matrices)


    new_particle_set = []
    for sampled_association in sampled_associations:
        print "sampled_association.meas_grp_associations", sampled_association.meas_grp_associations, "log prob:", np.log(sampled_association.complete_assoc_probability)
        child_particle = particle_set[sampled_association.matrix_index].create_child_particle(\
            log_child_imprt_weight=np.log(sampled_association.complete_assoc_probability),\
            measurements=cur_time_step_measurements,\
            measurement_associations=sampled_association.meas_grp_associations)

        assert(T == len(child_particle.targets))
          
        new_particle_set.append(child_particle)


    return new_particle_set


def sequential_proposal_distribution_sampling_step(particle_set, cur_time_step_measurements):
    '''
    perform exact sampling without replacement using upper bounds on the permanent
    '''
    M = len(cur_time_step_measurements) #number of measurements
    T = M #currently using a fixed number of targets that always emit measurements

    
    new_particle_set = []
    for particle in particle_set:

        particle_prior_prob = particle.importance_weight

        (list_of_measurement_associations, proposal_probability) = associate_measurements_sequentially(targets=particle.targets, measurements=cur_time_step_measurements)
        
        complete_association_likelihood = 1.0
        for (meas_idx, target_idx) in enumerate(list_of_measurement_associations):
            cur_association_likelihood = get_assoc_likelihood(target=particle.targets[target_idx], measurement=cur_time_step_measurements[meas_idx])
            complete_association_likelihood *= cur_association_likelihood

        child_particle = particle.create_child_particle(\
            log_child_imprt_weight=np.log(complete_association_likelihood) + np.log(particle_prior_prob),\
            # log_child_imprt_weight=np.log(particle_prior_prob) + np.log(complete_association_likelihood/proposal_probability),\
            # log_child_imprt_weight=np.log(sampled_association.complete_assoc_probability),\
            measurements=cur_time_step_measurements,\
            measurement_associations=list_of_measurement_associations)

        child_particle.log_importance_weight_debug += np.log(complete_association_likelihood/proposal_probability)
        assert(T == len(child_particle.targets))
          
        new_particle_set.append(child_particle)

    assert(len(new_particle_set) == len(particle_set))
    return new_particle_set
    

def calc_permanent_rysers(matrix):
    '''
    Exactly calculate the permanent of the given matrix user Ryser's method (faster than calc_permanent)
    '''
    N = matrix.shape[0]
    assert(N == matrix.shape[1])
    #this looks complicated because the method takes and returns a complex matrix,
    #we are only dealing with real matrices so set complex component to 0
    return np.real(rysers_permanent(1j*np.zeros((N,N)) + matrix))


def exact_sampling_step_debug(particle_set, cur_time_step_measurements):
    '''
    perform exact sampling without replacement using upper bounds on the permanent
    '''
    M = len(cur_time_step_measurements) #number of measurements
    T = M #currently using a fixed number of targets that always emit measurements

    
    new_particle_set = []
    for particle in particle_set:

        # probs = construct_exact_sampling_matrix(targets=particle.targets, measurements=cur_time_step_measurements)
        probs = construct_exact_sampling_matrix_constantNumTargets(targets=particle.targets, measurements=cur_time_step_measurements)
        # cur_permanent = calc_permanent_rysers(probs)
        particle_prior_prob = particle.importance_weight

        matrix_rescaling = 1.0
        # try to help numerical issues, might not help
        for row in range(probs.shape[0]):
            max_row_val = np.max(probs[row])
            if max_row_val > 0.0:
                probs[row] /= max_row_val
                # particle_prior_prob *= max_row_val
                matrix_rescaling *= max_row_val
        for col in range(probs.shape[1]):
            max_col_val = np.max(probs[:,col])
            if max_col_val > 0.0:
                probs[:,col] /= max_col_val
                # particle_prior_prob *= max_col_val            
                matrix_rescaling *= max_col_val            

        particle_prior_prob *= matrix_rescaling
        # print "probs:"
        # print probs
        # print "particle_prior_prob =", particle_prior_prob
        cur_a_matrix = associationMatrix(matrix=probs, M=M, T=T,\
            conditional_birth_probs=[0.0 for i in range(M)], conditional_death_probs=[0.0 for i in range(T)],\
            prior_prob=1.0)
            # prior_prob=particle_prior_prob)
        cur_prior_prob = particle_prior_prob
        cur_all_association_matrices = [cur_a_matrix]

        # sampled_associations = multi_matrix_sample_associations_without_replacement(num_samples=1, all_association_matrices=cur_all_association_matrices)
        (sampled_association, permanent_estimate) = single_matrix_sample_associations_with_replacement(num_samples=1, single_association_matrices=cur_all_association_matrices)
        permanent_estimate *= matrix_rescaling
        print "permanent_estimate = ", permanent_estimate,
        print "sampled_association.meas_grp_associations", sampled_association.meas_grp_associations, "log prob:", np.log(sampled_association.complete_assoc_probability)
        child_particle = particle.create_child_particle(\
            log_child_imprt_weight=np.log(sampled_association.complete_assoc_probability) + np.log(cur_prior_prob),\
            # log_child_imprt_weight=np.log(sampled_association.complete_assoc_probability),\
            measurements=cur_time_step_measurements,\
            measurement_associations=sampled_association.meas_grp_associations)

        child_particle.log_importance_weight_debug +=  np.log(permanent_estimate)
        assert(T == len(child_particle.targets))
          
        new_particle_set.append(child_particle)

    assert(len(new_particle_set) == len(particle_set))
    return new_particle_set

def exact_sampling_step_modifiedSIS(particle_set, cur_time_step_measurements):
    '''
    perform exact sampling without replacement using upper bounds on the permanent
    '''
    M = len(cur_time_step_measurements) #number of measurements
    T = M #currently using a fixed number of targets that always emit measurements
    all_association_matrices = []

    prior_probs = []

    for particle in particle_set:

        # probs = construct_exact_sampling_matrix(targets=particle.targets, measurements=cur_time_step_measurements)
        probs = construct_exact_sampling_matrix_constantNumTargets(targets=particle.targets, measurements=cur_time_step_measurements)
        particle_prior_prob = particle.importance_weight

        # try to help numerical issues, might not help
        for row in range(probs.shape[0]):
            max_row_val = np.max(probs[row])
            probs[row] /= max_row_val
            particle_prior_prob *= max_row_val
        for col in range(probs.shape[1]):
            max_col_val = np.max(probs[:,col])
            probs[:,col] /= max_col_val
            particle_prior_prob *= max_col_val            

        # print "probs:"
        # print probs
        # print "particle_prior_prob =", particle_prior_prob
        cur_a_matrix = associationMatrix(matrix=probs, M=M, T=T,\
            conditional_birth_probs=[0.0 for i in range(M)], conditional_death_probs=[0.0 for i in range(T)],\
            prior_prob=1.0)
        all_association_matrices.append(cur_a_matrix)

        prior_probs.append(particle_prior_prob)

    sampled_associations = multi_matrix_sample_associations_without_replacement(num_samples=N_PARTICLES, all_association_matrices=all_association_matrices)


    new_particle_set = []
    for sampled_association in sampled_associations:
        print "sampled_association.meas_grp_associations", sampled_association.meas_grp_associations, "log prob:", np.log(sampled_association.complete_assoc_probability)
        child_particle = particle_set[sampled_association.matrix_index].create_child_particle(\
            log_child_imprt_weight=np.log(sampled_association.complete_assoc_probability) + np.log(prior_probs[sampled_association.matrix_index]),\
            measurements=cur_time_step_measurements,\
            measurement_associations=sampled_association.meas_grp_associations)

        assert(T == len(child_particle.targets))
          
        new_particle_set.append(child_particle)


    return new_particle_set


def MHT_step(particle_set, cur_time_step_measurements, extra_assignments_to_find_factor = 10):
    '''
    perform multiple hypothesis tracking
    '''
    M = len(cur_time_step_measurements) #number of measurements
    T = M #currently using a fixed number of targets that always emit measurements

    new_particle_set = []

    perturbed_cost_matrices = [] #list of negative perturbed log prob matrices for each particle group
    log_prob_matrices = [] #list of log prob matrices for each particle group
    particle_neg_log_probs = [] # negative log probabilities 
    particle_costs = [] # negative log probabilities + 2*(M+T) of the min_cost for each particle group to pass along when solving minimum cost assignments
    #we need all entries of all cost matrices to be positive, so we keep track of the smallest value
    min_cost = 0.0

    # print
    # print
    for particle in particle_set:
        #1. construct log probs matrix for particle GROUP
        cur_log_probs = construct_MHT_matrix(targets=particle.targets, measurements=cur_time_step_measurements)
        log_prob_matrices.append(cur_log_probs) #store to calculate probabilities later
        # assert((cur_log_probs <= .000001).all()), (cur_log_probs)
        cur_cost_matrix = -1*cur_log_probs #k_best_assign_mult_cost_matrices is set up to find minimum cost, not max log prob
        
        cur_min_cost = np.min(cur_cost_matrix)

        if cur_min_cost < min_cost:
            min_cost = cur_min_cost

        perturbed_cost_matrices.append(cur_cost_matrix)
        #add min_cost*2*(M+T) term because T varies between particle groups and we subtract min_cost from all entries in the perturbed cost matrix
        particle_costs.append(-1*np.log(particle.importance_weight) + min_cost*2*(M+T))
        particle_neg_log_probs.append(-1*np.log(particle.importance_weight))


    #make all entries of all cost matrices non-negative
    for idx in range(len(perturbed_cost_matrices)):
        perturbed_cost_matrices[idx] = perturbed_cost_matrices[idx] - min_cost
        assert((perturbed_cost_matrices[idx] >= 0.0).all()), (perturbed_cost_matrices[idx])

    #4. find N_PARTICLES most likely assignments among all assignments in log probs matrices of ALL particle GROUPS
    #best_assignments: (list of triplets) best_assignments[i][0] is the cost of the ith best
    #assignment.  best_assignments[i][1] is the ith best assignment, which is a list of pairs
    #where each pair represents an association in the assignment (1's in assignment matrix),
    #best_assignments[i][2] is the index in the input cost_matrices of the cost matrix used
    #for the ith best assignment
    best_assignments = k_best_assign_mult_cost_matrices(N_PARTICLES*extra_assignments_to_find_factor, perturbed_cost_matrices, particle_costs, M)
    # print "len(best_assignments): ", len(best_assignments)


    #5. For each of the most likely assignments, create a new particle that is a copy of its particle GROUP, 
    # and associate measurements / kill targets according to assignment.
    prv_cost=-np.inf
    for (idx, (cur_cost, cur_assignment, cur_particle_idx)) in enumerate(best_assignments):
        assert(cur_cost >= prv_cost or np.allclose(cur_cost, prv_cost)), (cur_cost, prv_cost, idx) #make sure costs are decreasing in best_assignments
        prv_cost = cur_cost
        #3. create a new particle that is a copy of the max group, and associate measurements / kill
        #targets according to the max x_k from 2.
        cur_assignment_matrix = convert_assignment_pairs_to_matrix3(cur_assignment, M, T)
        np.nan_to_num(log_prob_matrices[cur_particle_idx], copy=False) #convert -infinity assignment costs to very negative number to avoid nan importance weights
        assignment_log_prob = np.trace(np.dot(log_prob_matrices[cur_particle_idx], cur_assignment_matrix.T))
        new_particle_log_importance_weight = assignment_log_prob - particle_neg_log_probs[cur_particle_idx] #log prob
        
        # print "log_prob_matrices[cur_particle_idx]:"
        # print log_prob_matrices[cur_particle_idx]
        # print "cur_assignment_matrix.T:"
        # print cur_assignment_matrix.T
        # print "np.trace(np.dot(log_prob_matrices[cur_particle_idx], cur_assignment_matrix.T)):"
        # print np.trace(np.dot(log_prob_matrices[cur_particle_idx], cur_assignment_matrix.T))
        # print "np.dot(log_prob_matrices[cur_particle_idx], cur_assignment_matrix.T):"
        # print np.dot(log_prob_matrices[cur_particle_idx], cur_assignment_matrix.T)


        # print("new_particle_log_importance_weight =", new_particle_log_importance_weight, 'assignment_log_prob =', assignment_log_prob, 'particle_neg_log_probs[cur_particle_idx] =', particle_neg_log_probs[cur_particle_idx], 'parent_particle_idx =', cur_particle_idx)
        # print("cur_assignment:", cur_assignment[:7])
        # print

        # if math.isnan(new_particle_log_importance_weight):
        #     print np.isnan(log_prob_matrices[cur_particle_idx]).any()
        #     random_number = np.random.random()
        #     matrix_file_name = './matrices_for_debugging/inspect_matrices%f' % random_number
        #     if not os.path.exists(os.path.dirname(matrix_file_name)):
        #         try:
        #             os.makedirs(os.path.dirname(matrix_file_name))
        #         except OSError as exc: # Guard against race condition
        #             if exc.errno != errno.EEXIST:
        #                 raise
        #     print "saving matrices in %s" % matrix_file_name
        #     f = open(matrix_file_name, 'w')
        #     pickle.dump((log_prob_matrices[cur_particle_idx], cur_assignment_matrix.T), f)
        #     f.close()                  

        (meas_grp_associations, dead_target_indices) = convert_assignment_matrix3(cur_assignment_matrix, M, T)
        assert(len(dead_target_indices) == 0), "current implementation assumes constant # targets"
        child_particle = particle_set[cur_particle_idx].create_child_particle(\
            log_child_imprt_weight=new_particle_log_importance_weight,\
            measurements=cur_time_step_measurements,\
            measurement_associations=meas_grp_associations)

        assert(T == len(child_particle.targets))

        new_particle_set.append(child_particle)
        assert(child_particle.parent_particle != None)

    return new_particle_set


def normalize_importance_weights(particle_set, normalize_log_weights=True):
    if normalize_log_weights:
        largest_log_imprt_weight = -np.inf
        for particle in particle_set:
            if particle.log_importance_weight > largest_log_imprt_weight:
                largest_log_imprt_weight = particle.log_importance_weight

        #normalize importance weights so all importance weights in the particle set sum to 1.0
        normalization_constant = 0.0
        for particle in particle_set:
            normalization_constant += np.exp(particle.log_importance_weight - largest_log_imprt_weight)
        assert(normalization_constant != 0.0), normalization_constant
        # print("normalize_importance_weights called, normalization_constant =", normalization_constant)
        importance_weight_sum = 0.0
        for particle in particle_set:
            particle.importance_weight = np.exp(particle.log_importance_weight - largest_log_imprt_weight)/normalization_constant
            # particle.log_importance_weight_normalization += np.log(normalization_constant*np.exp(largest_log_imprt_weight))
            particle.log_importance_weight_normalization += np.log(normalization_constant) + largest_log_imprt_weight
            importance_weight_sum += particle.importance_weight
        assert(np.isclose(importance_weight_sum, 1, rtol=1e-04, atol=1e-04)), importance_weight_sum

    else:
        #normalize importance weights so all importance weights in the particle set sum to 1.0
        normalization_constant = 0.0
        for particle in particle_set:
            # print("particle.importance_weight:", particle.importance_weight)
            normalization_constant += particle.importance_weight
        assert(normalization_constant != 0.0), normalization_constant
        # print("normalize_importance_weights called, normalization_constant =", normalization_constant)
        importance_weight_sum = 0.0
        for particle in particle_set:
            particle.importance_weight /= normalization_constant
            particle.log_importance_weight_normalization += np.log(normalization_constant)
            importance_weight_sum += particle.importance_weight
        assert(np.isclose(importance_weight_sum, 1, rtol=1e-04, atol=1e-04)), importance_weight_sum

def convert_measurements_by_target_to_by_time(all_measurements, randomize_order=False):
    '''
    Inputs:
    - all_measurements: (list of list of measurements) all_measurements[i][j] is the ith target's measurement at the jth time instance
    - randomize_order: (bool) whether to permute the order of measurements at each time step in all_measurements_by_time
    Outputs:
    - all_measurements_by_time: (list of list of measurements) all_measurements_by_time[i][j] is the jth measurement from
                             the ith time step
    '''
    assert(randomize_order==False), randomize_order
    all_measurements_by_time = []
    number_time_steps = len(all_measurements[0])
    number_targets = len(all_measurements)
    for target_idx in range(number_targets):
        assert(len(all_measurements[target_idx]) == number_time_steps)
    for time_idx in range(number_time_steps):
        cur_time_measurements = []
        for target_idx in range(number_targets):
            cur_time_measurements.append(all_measurements[target_idx][time_idx])
        all_measurements_by_time.append(cur_time_measurements) 
    return all_measurements_by_time

def get_gt_association_likelihood(gen_params, all_measurements):
    '''
    Outputs:
    - all_log_likelihoods: (list of floats) all_log_likelihoods[i] is the ground truth log likelihood at the ith timestep
    '''
    log_prob_of_all_targets = 0
    all_log_likelihoods = None
    for target_idx in range(gen_params.num_targets):
        with io.capture_output() as captured:
            cur_gen_params = copy.copy(gen_params)
            cur_gen_params.initial_position_means = [gen_params.initial_position_means[target_idx]]
            cur_gen_params.initial_velocity_means = [gen_params.initial_velocity_means[target_idx]]
            cur_gen_params.spring_constants = [gen_params.spring_constants[target_idx]]
            (all_target_posteriors, all_target_priors, most_probable_particle, cur_target_all_log_likelihoods, log_likelihoods_from_most_probable_particles) = run_tracking([all_measurements[target_idx]], tracking_method='MHT', generative_parameters=cur_gen_params, n_particles=1, use_group_particles='False')
            if all_log_likelihoods is None:
                all_log_likelihoods = [ll[0] for ll in cur_target_all_log_likelihoods]
            else:
                assert(len(all_log_likelihoods) == len(cur_target_all_log_likelihoods))
                for idx, ll in enumerate(cur_target_all_log_likelihoods):
                    all_log_likelihoods[idx] += ll[0]
            print "all_log_likelihoods    :", all_log_likelihoods                        
            log_prob_of_all_targets += most_probable_particle.log_importance_weight_normalization + np.log(most_probable_particle.importance_weight)
            print(most_probable_particle.log_importance_weight_normalization + np.log(most_probable_particle.importance_weight))

    print "log_prob_of_all_targets:", log_prob_of_all_targets
    print "all_log_likelihoods    :", all_log_likelihoods    

    # assert(log_prob_of_all_targets == all_log_likelihoods[-1])
    return log_prob_of_all_targets, all_log_likelihoods


def exp_normalize(x):
    #https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    b = x.max()
    y = np.exp(x - b)
    return y / y.sum()

def perform_resampling(particle_set):
    # print "memory used before resampling: %d" % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert(len(particle_set) == N_PARTICLES)
    weights = []
    for particle in particle_set:
        weights.append(particle.importance_weight)
    assert(abs(sum(weights) - 1.0) < .0000001)

    new_particles = stratified_resample(weights)
    new_particle_set = []
    for index in new_particles:
        USE_CREATE_CHILD = False
        if USE_CREATE_CHILD:
            new_particle_set.append(particle_set[index].create_child())
        else:
            new_particle_set.append(copy.deepcopy(particle_set[index]))
    del particle_set[:]
    for particle in new_particle_set:
        particle.importance_weight = 1.0/N_PARTICLES
        particle_set.append(particle)
    assert(len(particle_set) == N_PARTICLES)
    #testing
    weights = []
    for particle in particle_set:
        weights.append(particle.importance_weight)
        assert(particle.importance_weight == 1.0/N_PARTICLES)
    assert(abs(sum(weights) - 1.0) < .01), sum(weights)
    # print "memory used after resampling: %d" % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    #done testing

def run_tracking(all_measurements, tracking_method, generative_parameters, n_particles, use_group_particles):
    """
    Measurement class designed to only have 1 measurement/time instance
    Input:
    - all_measurements: (list of list of measurements) all_measurements[i][j] is the ith target's measurement at the jth time instance
    - tracking_method: (string) either 'exact_sampling' or 'MHT'
    - generative_parameters: (GenerativeParameters) object storing parameters used for data generation and inference
    - n_particles: (int) the number of particles (samples) used in exact sampling or hypotheses in multiple hypothesis tracking
    - use_group_particles: (bool) During tracking we maintain a particle set where each particle represents a unique set of measurement 
        target associations throughout time.  However, it's possible that two particles could differ only in an association that took 
        place long ago and the distributions over current target states are indistinguishable.  If True, merge particles whose 
        distributions over current target states are close.
    Output:
    - all_target_posteriors: (list of list of np.array's) all_target_posteriors[i][j] is the posterior distribution of the 
                         ith target's state at the jth time step
    - all_log_likelihoods: (list of list of floats) all_log_likelihoods[i][j] is the jth sample's (or hypotheses') log likelihood
                           at the ith time step                         
    """
    global N_PARTICLES
    N_PARTICLES = n_particles

    global GEN_PARAMS 
    GEN_PARAMS = generative_parameters

    single_initial_particle = Particle(parent_particle=None, generative_parameters=generative_parameters, log_importance_weight=1.0)
    all_measurements_by_time = convert_measurements_by_target_to_by_time(all_measurements=all_measurements)
    print "all_measurements_by_time:"
    print all_measurements_by_time

    #set false for varying spring constants
    global INITIAL_ASSOCIATIONS_KNOWN
    INITIAL_ASSOCIATIONS_KNOWN = False


    T = len(all_measurements_by_time[0]) #number of targets
    for target_idx in range(T):
        root_target_node = TargetStateNode(gen_params=generative_parameters, parent_node=None, target_idx=target_idx)            
        if INITIAL_ASSOCIATIONS_KNOWN:
            root_target_node.initialize_root_target_initialAssociationKnown(measurement=all_measurements_by_time[0][target_idx], target_idx=target_idx)
        else:
            root_target_node.initialize_root_target_initialAssociationNotKnown(target_idx=target_idx)
        single_initial_particle.targets.append(root_target_node)
    particle_set = [single_initial_particle]


    DEBUG_EXACT_SAMPLING = True
    if DEBUG_EXACT_SAMPLING and (tracking_method == 'exact_sampling' or tracking_method == 'sequential_proposal_SMC'):
        while(len(particle_set) < N_PARTICLES):
            particle_set.append(copy.copy(single_initial_particle))

    time_step=1
    if INITIAL_ASSOCIATIONS_KNOWN:
        remaining_measurements_by_time = all_measurements_by_time[1:]
    else:
        remaining_measurements_by_time = all_measurements_by_time

    all_log_likelihoods = []
    for cur_time_step_measurements in remaining_measurements_by_time:    
        print '-'*40, 'timestep', time_step, '-'*40
        time_step += 1
        if tracking_method == 'exact_sampling':
            # particle_set = exact_sampling_step(particle_set, cur_time_step_measurements)
            particle_set = exact_sampling_step_debug(particle_set, cur_time_step_measurements)
            # particle_set = exact_sampling_step_modifiedSIS(particle_set, cur_time_step_measurements)
        elif tracking_method == 'sequential_proposal_SMC':
            particle_set = sequential_proposal_distribution_sampling_step(particle_set, cur_time_step_measurements)
        elif tracking_method == 'MHT':
            particle_set = MHT_step(particle_set, cur_time_step_measurements, extra_assignments_to_find_factor=1)
        elif tracking_method == 'gt_assoc':
            particle_set = gt_assoc_step(particle_set, cur_time_step_measurements)
        else:
            assert(False)
           
        normalize_importance_weights(particle_set)
        print "len(particle_set) before grouping particles:",len(particle_set)

        if use_group_particles:
            # particle_set = group_particles(particle_set)
            grouped_particle_set = []
            for extra_idx in range(1):
                print "extra_idx:", extra_idx, "len(grouped_particle_set):", len(grouped_particle_set)
                grouped_particle_set = group_particles(grouped_particle_set + particle_set[extra_idx*N_PARTICLES:(extra_idx+1)*N_PARTICLES])

                if len(grouped_particle_set) >= N_PARTICLES:
                    break

            particle_set = grouped_particle_set[:N_PARTICLES]
            print "len(particle_set) after grouping particles:",len(particle_set)



        PRINT_CUR_LOG_LIKELIHOOD = True
        if PRINT_CUR_LOG_LIKELIHOOD:
            cur_time_step_log_likelihoods = []
            cur_most_probable_particle = None
            first_particle_log_importance_weight_normalization = None
            for particle in particle_set:
                cur_time_step_log_likelihoods.append(particle.log_importance_weight_normalization + np.log(particle.importance_weight))
                # cur_time_step_log_likelihoods.append(particle.log_importance_weight_debug)
                if first_particle_log_importance_weight_normalization is None:
                    first_particle_log_importance_weight_normalization = particle.log_importance_weight_normalization
                else:
                    assert(first_particle_log_importance_weight_normalization == particle.log_importance_weight_normalization)
                if cur_most_probable_particle is None or particle.importance_weight > cur_most_probable_particle.importance_weight:
                    cur_most_probable_particle = particle
            all_log_likelihoods.append(cur_time_step_log_likelihoods)
            print("current most probable particle log_prob:", cur_most_probable_particle.log_importance_weight_normalization + np.log(cur_most_probable_particle.importance_weight))

            cur_least_probable_particle = None
            for particle in particle_set:
                if cur_least_probable_particle is None or particle.importance_weight < cur_least_probable_particle.importance_weight:
                    cur_least_probable_particle = particle

            print("current least probable particle log_prob:", cur_least_probable_particle.log_importance_weight_normalization + np.log(cur_least_probable_particle.importance_weight))
        
        unnormalized_log_importance_weights = []
        for particle in particle_set:
            unnormalized_log_importance_weights.append(particle.log_importance_weight_debug)
        normalized_importance_weights = exp_normalize(np.array(unnormalized_log_importance_weights))
        for idx, particle in enumerate(particle_set):
            particle.normalized_importance_weight_debug = normalized_importance_weights[idx]
        print "unnormalized_log_importance_weights:", unnormalized_log_importance_weights
        print "normalized_importance_weights:", normalized_importance_weights
        effective_num_particles = 1/np.sum(np.power(normalized_importance_weights, 2))
        print "effective number of particles =", effective_num_particles
        # if effective_num_particles < N_PARTICLES/10:
        # if effective_num_particles < 1.001:
        if False:
            print "resampling particles"
            perform_resampling(particle_set)         


    #we sample without replacement with exact sampling, so just find the highest weight sample, no marginalization
    most_probable_particle = None
    log_likelihoods_from_most_probable_particles = []
    for particle in particle_set:
        cur_particle_log_likelihoods = particle.get_log_likelihoods_over_time()
        log_likelihoods_from_most_probable_particles.append(cur_particle_log_likelihoods)
        if most_probable_particle is None or particle.importance_weight > most_probable_particle.importance_weight:
            most_probable_particle = particle
    all_target_posteriors = most_probable_particle.get_all_target_posteriors()
    all_target_priors = most_probable_particle.get_all_target_priors()

    return (all_target_posteriors, all_target_priors, most_probable_particle, all_log_likelihoods, log_likelihoods_from_most_probable_particles)

def target_distance_metric(targetA, targetB):
    '''
    *** may not be a valid distance metric ***
    define distance between two targets
    '''
    distance = np.sum((targetA.posterior[0] - targetB.posterior[0])**2) \
             + np.sum((targetA.posterior[1] - targetB.posterior[1])**2) \
             + (targetA.spring_constant - targetB.spring_constant)**2
    return distance

def min_cost_assignment(cost_matrix):
    '''
    Inputs:
    - cost_matrix: (np.array)(CHECK THIS IS CORRECT!!)

    Outputs:
    - association_list: (list of pairs) each represents an assignment (CHECK THIS IS CORRECT!!)
    '''
    lin_assign = linear_assignment.LinearAssignment(cost_matrix)
    solution = lin_assign.solution
    association_list = zip([i for i in range(len(solution))], solution)
    return association_list

def particles_are_similar(particleA, particleB, distance_threshold=.1):
    '''
    During tracking we maintain a particle set where each particle represents a unique set of measurement target associations 
    throughout time.  However, it's possible that two particles could differ only in an association that took place long ago
    and the distributions over current target states are indistinguishable.  This function decides whether two particles 
    distributions over current target states are sufficiently close so that they can be merged.
    '''
    if len(particleA.targets) != len(particleB.targets): #particles don't have the same number of targets
        return False 

    T = len(particleA.targets)
    # print("T =", T)
    #1. construct a cost matrix of size TxT where the cost_matrix_ij is (distribution) distance between the distribution
    #of particleA's ith target and particleB's jth target
    cost_matrix = np.zeros((T,T))
    for a_idx, targetA in enumerate(particleA.targets):
        for b_idx, targetB in enumerate(particleB.targets):
            cost_matrix[a_idx, b_idx] = target_distance_metric(targetA, targetB)

    #2. find the minimum cost matching between particle A's targets and particle B's targets
    min_cost_association_list = min_cost_assignment(cost_matrix)
    # print("min_cost_association_list:", min_cost_association_list)
    #3. if all distances are sufficiently small in the minimum cost matching return True, otherwise False
    for cur_association in min_cost_association_list:
        if cost_matrix[cur_association] > distance_threshold:
            return False

    if False:
        print '-' * 80
        print "particles are similar"
        print "min_cost_association_list:", min_cost_association_list
        for idx, cur_association in enumerate(min_cost_association_list):
            # print "association index:", idx
            # print "cur_association", cur_association, "cur_association cost", cost_matrix[cur_association]
            # if cur_association[0] != cur_association[1]:
            #     print 'target from particle a posterior and associations'
            #     print particleA.targets[cur_association[0]].posterior
            #     print particleA.targets[cur_association[0]].print_measurement_associations(leaf_node = True)

            #     print 'target from particle b posterior and associations'
            #     print particleB.targets[cur_association[1]].posterior
            #     print particleB.targets[cur_association[1]].print_measurement_associations(leaf_node = True)

            target_1 = particleA.targets[cur_association[0]]
            target_2 = particleB.targets[cur_association[1]]
            list_of_measurements1 = target_1.get_associated_measurements()
            list_of_measurements2 = target_2.get_associated_measurements()
            list_of_measurement_indices1 = target_1.get_associated_measurement_indices()
            list_of_measurement_indices2 = target_2.get_associated_measurement_indices()

            if list_of_measurements2 == list_of_measurements1:
                print "targets in association", idx, "have same measurement associations"
            else:
                print "targets in association", idx, "have different measurement associations"
                print "target 1 associations:", list_of_measurements1
                print "target 2 associations:", list_of_measurements2

            if list_of_measurement_indices1 == list_of_measurement_indices2:
                print "targets in association", idx, "have same measurement association indices"
            else:
                print "targets in association", idx, "have different measurement associations"
                print "target 1 measurement association indices:", list_of_measurement_indices1
                print "target 2 measurement association indices:", list_of_measurement_indices2


            if not np.allclose(particleA.targets[cur_association[0]].posterior, particleB.targets[cur_association[1]].posterior):
                print "first posterior:"
                print particleA.targets[cur_association[0]].posterior
                print "second posterior:"
                print particleB.targets[cur_association[1]].posterior

        # for t_idx, target in enumerate(particleA.targets):
        #     list_of_measurements1 = target.get_associated_measurements()
        #     # print "posterior", target.posterior
        #     list_of_measurements2 = particleB.targets[t_idx].get_associated_measurements()
        #     if list_of_measurements2 == list_of_measurements1:
        #         print "targets", t_idx, "have same measurement associations"
        #     else:
        #         print "targets", t_idx, "have different measurement associations"
        #         print "target 1 associations:", list_of_measurements1
        #         print "target 2 associations:", list_of_measurements2
        #     # print "posterior", particleB.targets[t_idx].posterior
        #     print
        print
        print

    return True


def group_particles(particle_set, verbose=False):
    '''
    During tracking we maintain a particle set where each particle represents a unique set of measurement target associations 
    throughout time.  However, it's possible that two particles could differ only in an association that took place long ago
    and the distributions over current target states are indistinguishable.  This function merges particles whose distributions
    over current target states are sufficiently close.
    '''
    grouped_particle_set = [particle_set[0]]
    for old_particle in particle_set[1:]:
        old_particle_added_to_group = False
        for new_particle_idx, new_particle in enumerate(grouped_particle_set):
            if particles_are_similar(new_particle, old_particle):
                if verbose:
                    print('particles are similar!!')
                    print("old_particle target associations:")
                    for target in old_particle.targets:
                        print target.debug_info
                    print
                    print("old_particle target parent priors:")
                    for targ_idx, target in enumerate(old_particle.targets):
                        print "target", targ_idx, "prior max =", np.max(target.parent_node.prior), "at index", np.argmax(target.parent_node.prior), "posterior max =", np.max(target.posterior), "at index", np.argmax(target.posterior)
                    print                
                    print("new_particle target associations:")
                    for target in new_particle.targets:
                        print target.debug_info   
                    print 
                    print("new_particle target parent priors:")
                    for targ_idx, target in enumerate(new_particle.targets):
                        print "target", targ_idx, "prior max =", np.max(target.parent_node.prior), "at index", np.argmax(target.parent_node.prior), "posterior max =", np.max(target.posterior), "at index", np.argmax(target.posterior)
                    print                 
                    print

                if old_particle.importance_weight > new_particle.importance_weight:
                    grouped_particle_set[new_particle_idx] = old_particle
                old_particle_added_to_group = True
                break
        if old_particle_added_to_group == False:
            grouped_particle_set.append(old_particle)

    #check that new importance weight sum to 1
    sum_of_new_particle_importance_weights = 0.0
    for particle in grouped_particle_set:
        sum_of_new_particle_importance_weights += particle.importance_weight
    # assert(np.isclose(sum_of_new_particle_importance_weights, 1, rtol=1e-04, atol=1e-04))

    return grouped_particle_set



def get_assoc_likelihood(target, measurement):
    """
    Inputs:
    """
    global GEN_PARAMS

    R = GEN_PARAMS.r_matrix
    H = GEN_PARAMS.h_matrix

    target_X = target.prior[0]
    target_P = target.prior[1]

    S = np.dot(np.dot(H, target_P), H.T) + R
    assert(target_X.shape == (3, 1)), (target_X.shape, target_X)


    state_mean_meas_space = np.dot(H, target_X)
    state_mean_meas_space = np.squeeze(state_mean_meas_space)

    use_python_gaussian = True
    if use_python_gaussian:
        distribution = multivariate_normal(mean=state_mean_meas_space, cov=S)
        assoc_likelihood = distribution.pdf(measurement)
    else: #probably not correct for the current state space size
        S_det = S[0][0]*S[1][1] - S[0][1]*S[1][0] # a little faster
        S_inv = matrix_inverse(S)
        assert(S_det > 0), (S_det, S, target_P, R)
        LIKELIHOOD_DISTR_NORM = 1.0/math.sqrt((2*math.pi)**2*S_det)

        offset = measurement - state_mean_meas_space
        a = -.5*np.dot(np.dot(offset, S_inv), offset)
        assoc_likelihood = LIKELIHOOD_DISTR_NORM*math.exp(a)

    # assert(assoc_likelihood >= 0.0 and assoc_likelihood <= 1.0), (assoc_likelihood, state_mean_meas_space, S, measurement)

    return assoc_likelihood



def construct_exact_sampling_matrix(targets, measurements):
    '''
    M = #measurements
    T = #targets


    Inputs:
    - particle: type Particle, we will perform sampling and importance reweighting on this particle         
    - meas_groups: a list of detection groups, where each detection group is a dictionary of detections 
        in the group, key='det_name', value=detection (should be numpy array of [x,y,width,height])
    - total_target_count: the number of living targets on the previous time instace
    - p_target_deaths: a list of length len(total_target_count) where 
        p_target_deaths[i] = the probability that target i has died between the last
        time instance and the current time instance
    - params: type Parameters, gives prior probabilities and other parameters we are using

    Outputs:
    - probs: numpy matrix with dimensions (M+T)x(M+T) with probabilities:
        [a_11    ...     a_1T   um_1 0   ...  0]
        [.               .      0   .          ]
        [.               .      .      .       ]
        [.               .      .         .    ]
        [a_M1    ...     a_MT   0    ...   um_M]
        [ut_1    ...     ut_T   1    ...      1]
        [.               .      .             .]
        [.               .      .             .]
        [ut_1    ...     ut_T   1    ...      1]    
        - upper left quadrant is a MxT submatrix and composed of a_ij = the association probability of
          measurement i to target j
        - upper right quadrant is an MxM submatrix.  Row i is composed of M repetitions of 
          um_i = the probability that measurement i is unassociated with a target (marginalized over whether the
          measurement is clutter or a birth)
        - lower left quadrant is a TxT submatrix.  It is a diagonal matrix with elements ut_i = the
          probability that target i doesn't emit a measuremnt (marginalized over
          whether it lives or dies)
        - lower right quadrant is an TxM submatrix of all 1's



    '''
    M = len(measurements)
    T = len(targets)
    probs = np.zeros((M + T, M + T))
    
    conditional_birth_probs = np.ones(M)
    conditional_death_probs = np.ones(T)
    clutter_probs_conditioned_unassoc = []

    #calculate log probs for target association entries in the log-prob matrix
    for t_idx, target in enumerate(targets):
        for m_idx, measurement in enumerate(measurements):
        #calculate log probs for measurement-target association entries in the log-prob matrix
            likelihood = get_assoc_likelihood(target, measurement)
            probs[m_idx][t_idx] = likelihood

    #set bottom right quadrant to 1's
    for row_idx in range(M, M+T):
        for col_idx in range(T, T+M):
            probs[row_idx][col_idx] = 1.0


    return probs

def construct_exact_sampling_matrix_constantNumTargets(targets, measurements):
    '''
    association matrix for a constant number of targets with no clutter, number of measurements
    equals number of targets

    M = #measurements
    T = #targets
    require M=T

    Inputs:

    Outputs:
    - probs: numpy matrix with dimensions (M)x(T) with probabilities:
        [a_11    ...     a_1T]
        [.               .   ]
        [.               .   ]
        [.               .   ]
        [a_M1    ...     a_MT]  
        - upper left quadrant is a MxT submatrix and composed of a_ij = the association probability of
          measurement i to target j (M=T for this function)


    '''
    M = len(measurements)
    T = len(targets)
    assert(M == T)

    probs = np.zeros((M, M))
    #calculate log probs for target association entries in the log-prob matrix
    for t_idx, target in enumerate(targets):
        for m_idx, measurement in enumerate(measurements):
        #calculate log probs for measurement-target association entries in the log-prob matrix
            likelihood = get_assoc_likelihood(target, measurement)
            probs[m_idx][t_idx] = likelihood

    # print "check probs:"
    # print probs

    return probs

def ln_zero_approx(x):
    #return a small number instead of -infinity for ln(0)
    if x == 0:
        return -sys.maxint
    else:
        return math.log(x)
        
def construct_MHT_matrix(targets, measurements):
    '''
    M = #measurements
    T = #targets

    Inputs:
    - particle: type Particle, we will perform sampling and importance reweighting on this particle         
    - meas_groups: a list of detection groups, where each detection group is a dictionary of detections 
        in the group, key='det_name', value=detection (should be numpy array of [x,y,width,height])
    - total_target_count: the number of living targets on the previous time instace
    - p_target_deaths: a list of length len(total_target_count) where 
        p_target_deaths[i] = the probability that target i has died between the last
        time instance and the current time instance
    - params: type Parameters, gives prior probabilities and other parameters we are using

    Outputs:
    - log_probs: numpy matrix with dimensions (2*M+2*T)x(2*M+2*T) of log probabilities.
        np.trace(np.dot(log_probs,A.T) will be the log probability of an assignment A, given our
        Inputs.  (Where an assignment defines measurement associations to targets, and is marginalized

    - clutter_probs_conditioned_unassoc: (list of floats) length M, the probability that each measurement
        is clutter conditioned on it being unassociated with any target
    '''
    M = len(measurements)
    T = len(targets)
    assert(M==T)
    log_probs = np.ones((2*M + 2*T, 2*T + 2*M))
    log_probs *= -np.inf #setting all entries to very negative value

    # print 'target state means:'
    # for target in targets:
    #     print target.prior[0][0],
    # print
    # print 'measurements:'
    # for measurement in measurements:
    #     print measurement,
    # print


    for t_idx, target in enumerate(targets):
        for m_idx, measurement in enumerate(measurements):
        #calculate log probs for measurement-target association entries in the log-prob matrix
            likelihood = get_assoc_likelihood(target, measurement)
            log_probs[m_idx][t_idx] = ln_zero_approx(likelihood)
            assert(not math.isnan(log_probs[m_idx][t_idx])), (log_probs[m_idx][t_idx], likelihood)

    #set bottom right quadrant to 0's
    for row_idx in range(M, 2*M+2*T):
        for col_idx in range(T, 2*T+2*M):
            log_probs[row_idx][col_idx] = 0.0

    # print "log_probs:", log_probs

    return log_probs


def associate_measurements_sequentially(targets, measurements):

    """
    Try sampling associations with each measurement sequentially
    Input:

    Output:
    - list_of_measurement_associations: list of associations for each measurement group list_of_measurement_associations[i] = j
        means that measurement i is associated with target j
    - proposal_probability: proposal probability of the sampled associations
        
    """
    list_of_measurement_associations = []
    proposal_probability = 1.0

    for (meas_idx, measurement) in enumerate(measurements):
        #create proposal distribution for the current measurement
        #compute target association proposal probabilities
        proposal_distribution_list = []


        for (target_idx, target) in enumerate(targets):
            cur_target_likelihood = get_assoc_likelihood(target, measurement)
            # targ_likelihoods_summed_over_meas = 0.0
            # for (meas_idx2, measurement2) in enumerate(measurements):
            #     targ_likelihoods_summed_over_meas += get_assoc_likelihood(target, measurement2)

            # if (targ_likelihoods_summed_over_meas != 0.0) and (not target_idx in list_of_measurement_associations):
            #     cur_target_prior = cur_target_likelihood / targ_likelihoods_summed_over_meas
            # else:
            #     cur_target_prior = 0.0

            # proposal_distribution_list.append(cur_target_likelihood*cur_target_prior)

            if (cur_target_likelihood != 0.0) and (not target_idx in list_of_measurement_associations):
                proposal_distribution_list.append(cur_target_likelihood)
            else:
                proposal_distribution_list.append(0.0)



        #normalize the proposal distribution
        proposal_distribution = np.asarray(proposal_distribution_list)
        # assert(np.sum(proposal_distribution) != 0.0)
        if (np.sum(proposal_distribution) == 0.0):
            proposal_distribution = np.ones(len(proposal_distribution))
            for target_idx in list_of_measurement_associations:
                proposal_distribution[target_idx] = 0
        proposal_distribution /= float(np.sum(proposal_distribution))
        assert(len(proposal_distribution) == len(targets)), len(proposal_distribution)

        sampled_assoc_idx = np.random.choice(len(proposal_distribution),
                                                p=proposal_distribution)


        list_of_measurement_associations.append(sampled_assoc_idx)

        proposal_probability *= proposal_distribution[sampled_assoc_idx]


    return(list_of_measurement_associations, proposal_probability)



def plot_generated_data(all_xs, all_zs):
    ######################## PLOT DATA ########################
    for target_idx in range(gen_params.num_targets):
        xs = all_xs[target_idx]
        zs = all_zs[target_idx]
        print "states:", [x[0] for x in xs]
        plt.plot([x[0] for x in xs], label='states', marker='+', linestyle="None")
        print "measurements:", zs
        plt.plot([z[0] for z in zs], label='measurements', marker='x', linestyle="None")
#     plt.ylabel('some numbers')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    np.random.seed(3)

    N_PARTICLES = 1
    use_group_particles = False
    # method = 'exact_sampling'
    method = 'MHT'

    # (all_states, all_measurements, gen_params) = generate_data.get_parameters_and_data(num_time_steps=30, num_targets=15,\
    #                                   initial_position_variance=500, initial_vel_variance=0, measurement_variance=.01, spring_k=15, dt=.1)

    num_time_steps=20
    num_targets=7
    initial_position_means=[10*np.random.rand() for i in range(num_targets)]
    initial_position_variance=6
    initial_vel_variance=30
    measurement_variance=1
    spring_constants=[100*np.random.rand() for i in range(num_targets)]
    dt=.1

    (all_states, all_measurements, gen_params) = generate_data.get_parameters_and_data(num_time_steps, num_targets, \
        initial_position_means, initial_position_variance, initial_vel_variance, measurement_variance, spring_constants, dt)

    # with io.capture_output() as captured:
    # gen_params.num_targets = 1
    # all_measurements = all_measurements[0:1]
    (all_target_posteriors, all_target_priors, most_probable_particle, all_log_likelihoods, log_likelihoods_from_most_probable_particles) = run_tracking(all_measurements, tracking_method=method, generative_parameters=gen_params, n_particles=N_PARTICLES, use_group_particles=use_group_particles)

    print("most_probable_particle.importance_weight:", most_probable_particle.importance_weight)
    print("most_probable_particle.log_importance_weight_normalization:", most_probable_particle.log_importance_weight_normalization)
    print("most probable particle log_prob:", most_probable_particle.log_importance_weight_normalization + np.log(most_probable_particle.importance_weight))
    print("most probable particle prob:", np.exp(most_probable_particle.log_importance_weight_normalization + np.log(most_probable_particle.importance_weight)))

    print("ground truth, log_prob_of_all_targets =", get_gt_association_likelihood(gen_params, all_measurements))

    plot_generated_data(all_states, all_measurements)
    ######################## PLOT DATA ########################
    for target_idx in range(gen_params.num_targets):
        print 'hi!'
        xs = all_target_posteriors[target_idx]
        # print "states:", [x[0,0] for x in xs]
        plt.plot(xs, label='inferred states', marker='+', linestyle="None")
        plt.plot([x[0] for x in all_states[target_idx]], label='true states', marker='x', linestyle="None")
        plt.plot([z[0] for z in all_measurements[target_idx]], label='measurements', marker='+', linestyle="None")
    #     plt.ylabel('some numbers')
        plt.title('%s target %d' % (method, target_idx))
        plt.legend()
        # plt.show()
        plt.savefig('%s target %d' % (method, target_idx))
        plt.close()

