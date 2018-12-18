from __future__ import division
import sys
import numpy as np
# from filterpy.discrete_bayes import update as discrete_bayes_update
import math
from pymatgen.optimization import linear_assignment
from IPython.utils import io

import generate_data

RBPF_HOME_DIRECTORY = "/Users/jkuck/tracking_research/rbpf_fireworks/"
sys.path.insert(0, RBPF_HOME_DIRECTORY)
from rbpf_sampling_manyMeasSrcs import construct_log_probs_matrix3, convert_assignment_pairs_to_matrix3, convert_assignment_matrix3, construct_log_probs_matrix4

sys.path.insert(0, "%smht_helpers" % RBPF_HOME_DIRECTORY)
from k_best_assign_birth_clutter_death_matrix import k_best_assign_mult_cost_matrices

sys.path.insert(0, '/Users/jkuck/research/gumbel_sample_permanent')
# from tracking_specific_nestingUB_gumbel_sample_permanent import associationMatrix, multi_matrix_sample_associations_without_replacement
from constant_num_targets_sample_permenant import associationMatrix, multi_matrix_sample_associations_without_replacement

# N_PARTICLES = 5
INFEASIBLE_COST = 9999999999999999

class TargetStateNode:
    '''
    Represents the state of a target at one instance in time
    TargetStateNode's are linked to form a tree
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
        self.posterior = np.empty(gen_params.state_space_tuple)

        #prior distribution of the target's predicted state at the next time step
        self.prior = np.empty(gen_params.previous_dependent_states_shape)

        self.emission_probabilities = gen_params.emission_probabilities[target_idx]
        self.transition_probabilities = gen_params.transition_probabilities[target_idx]
        self.target_idx = target_idx

        self.gen_params = gen_params
        self.debug_info = debug_info

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


    def initialize_root_target_with_known_association(self, measurement, target_idx):
        '''
        set the prior and posterior distributions of a root target (created on this time step) when the initial association is known
        '''
        print 'target initialized with measurement:', measurement
        likelihood = self.get_likelihood(measurement)
        # print 'likelihood =', likelihood
        # print 'self.gen_params.all_initial_state_probabilities[target_idx] =', self.gen_params.all_initial_state_probabilities[target_idx]
        self.posterior = discrete_bayes_update(likelihood, self.gen_params.all_initial_state_probabilities[target_idx])
        # print 'initial self.posterior:', list(self.posterior).index(1)
        #resize posterior to previous_dependent_states_shape,
        resized_posterior = np.zeros(self.gen_params.previous_dependent_states_shape)
        for cur_state_index in np.ndindex(*self.gen_params.state_space_tuple):
            repeated_state_space_tuple = cur_state_index
            for i in range(self.gen_params.markov_order-1):
                repeated_state_space_tuple += cur_state_index
            resized_posterior[repeated_state_space_tuple] = self.posterior[cur_state_index]
        self.posterior = resized_posterior

        # print 'posterior =', self.posterior
        assert(self.posterior.shape == self.gen_params.previous_dependent_states_shape)
        self.prior = self.predict(self.posterior)
        assert(self.prior.shape == self.gen_params.previous_dependent_states_shape)

    def initialize_root_target_unknown_association(self, target_idx):
        '''
        set the prior and posterior distributions of a root target (created on this time step) with only a prior 
        (when the initial association is unknown)
        '''
        prior = self.gen_params.all_initial_state_probabilities[target_idx]
        print("prior initialized with max =", np.max(prior), "at index", np.argmax(prior))
        #resize prior to previous_dependent_states_shape,
        resized_prior = np.zeros(self.gen_params.previous_dependent_states_shape)
        for cur_state_index in np.ndindex(*self.gen_params.state_space_tuple):
            repeated_state_space_tuple = cur_state_index
            for i in range(self.gen_params.markov_order-1):
                repeated_state_space_tuple += cur_state_index
            resized_prior[repeated_state_space_tuple] = prior[cur_state_index]
        self.prior = resized_prior
        print("resized prior initialized with max =", np.max(self.prior), "at index", np.argmax(self.prior), 'resized_prior.shape:', self.prior.shape)

        self.posterior = np.ones(self.gen_params.previous_dependent_states_shape) #dummy posterior
        # print 'prior =', self.prior
        assert(self.prior.shape == self.gen_params.previous_dependent_states_shape)


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
            #the likelihood of each state given the measurement=the probability of the measurement given each state 
            likelihood = self.get_likelihood(measurement)
            new_child_node.posterior = discrete_bayes_update(likelihood, self.prior)
            new_child_node.debug_info['likelihood'] = likelihood
            new_child_node.debug_info['prior'] = self.prior

            new_child_node.prior = self.predict(new_child_node.posterior)
            self.child_nodes[meas_idx] = new_child_node
            return new_child_node

    def get_likelihood(self, z):
        '''
        
        The likelihood of a state given a measurement is the same as the 
        probability of a measurement given the state. 
        
        Inputs:
        - z: (np.array) a measurement, has the same shape as the state space
        - emission_probabilities: (np.array) has shape concatenate(state_space, state_space) e.g.
            (10, 5, 3, 10, 5, 3) for the above example.  The element (a, b, c, d, e, f) specifies the
            probability of emitting the measurement (d, e, f) when the state is (a, b, c).  Summing over
            emission_probabilities(a,b,c, :, :, :) must be 1 to be a properly normalized distribution
        
        Outputs:
        - likelihood: (np.array) specifies the likelihood of each state given the measurement.
                      has the same shape as the state space
        '''
        likelihood = self.emission_probabilities[..., z].squeeze()
        # print('-'*80)
        # print("measurement:", z, "likelihood[0]:", likelihood[0])
        # print('-'*80)
        assert(likelihood.shape == self.gen_params.state_space_tuple), (likelihood.shape, self.gen_params.state_space_tuple, self.emission_probabilities.shape)
        return likelihood

    def predict(self, posterior):
        '''
        Inputs:
        - posterior: (np.array) specifies a probability for each discrete element in the state space.
                 Has the same shape as the state space. (this is the posterior distribution after
                 incorporating the measurement from the current time step)
                             
        Outputs:
        - prior: (np.array) specifies a probability for each discrete element in the state space.
                 Has the same shape as the state space. (this is the prior distribution for the next time step)
        '''
        assert(posterior.shape == self.gen_params.previous_dependent_states_shape)
        prior = np.zeros(posterior.shape)
        
        #iterate over each state
        for cur_state_index in np.ndindex(*self.gen_params.previous_dependent_states_shape):
            # print "posterior[cur_state_index]:", posterior[cur_state_index]
            # print "self.transition_probabilities[cur_state_index]:", self.transition_probabilities[cur_state_index]

            # prior += posterior[cur_state_index]*self.transition_probabilities[cur_state_index]

            # prior has shape previous_dependent_states_shape, we keep markov_order - 1 previous states
            prior[cur_state_index[len(self.gen_params.state_space_tuple):]] += posterior[cur_state_index]*self.transition_probabilities[cur_state_index]
        # if np.max(posterior) > .98:
        #     print 'poster is concentrated np.max(np.sum(posterior, axis=0)) =', np.max(np.sum(posterior, axis=0)) 
        #     print 'poster is concentrated np.max(posterior) =', np.max(posterior) 
        #     print 'np.max(np.sum(prior, axis=0)) =', np.max(np.sum(prior, axis=0)) 
        #     assert(np.max(np.sum(prior, axis=0)) > .97) 
        # else:
        #     print 'poster is NOT concentrated np.max(np.sum(posterior, axis=0)) =', np.max(np.sum(posterior, axis=0))
        #     print 'poster is concentrated np.max(posterior) =', np.max(posterior)             
        #     print 'np.max(np.sum(prior, axis=0)) =', np.max(np.sum(prior, axis=0))

        assert(prior.shape == self.gen_params.previous_dependent_states_shape)
        # print "prior=", prior
        return prior

    def get_posteriors(self):
        '''
        Output:
        - posteriors: (list of np.array's) posteriors[j] is the posterior distribution of the 
                      target's state at the jth time step

        '''
        if self.parent_node is None:
            # posteriors = [self.posterior]
            #marginalize over previous states
            assert((self.gen_params.markov_order)*len(self.gen_params.state_space_tuple) == len(self.posterior.shape)), ((self.gen_params.markov_order)*len(self.gen_params.state_space_tuple), len(self.posterior.shape))
            axes_to_marginalize_over = tuple(range((self.gen_params.markov_order - 1)*len(self.gen_params.state_space_tuple)))
            marginal_posterior = np.sum(self.posterior, axis=axes_to_marginalize_over)
            posteriors = [marginal_posterior]
        else:
            posteriors = self.parent_node.get_posteriors()
            # posteriors.append(self.posterior)
            #marginalize over previous states
            assert((self.gen_params.markov_order)*len(self.gen_params.state_space_tuple) == len(self.posterior.shape)), ((self.gen_params.markov_order)*len(self.gen_params.state_space_tuple), len(self.posterior.shape))
            axes_to_marginalize_over = tuple(range((self.gen_params.markov_order - 1)*len(self.gen_params.state_space_tuple)))
            marginal_posterior = np.sum(self.posterior, axis=axes_to_marginalize_over)    
            posteriors.append(marginal_posterior)
                    
        return posteriors

    def get_priors(self):
        '''
        Output:
        - priors: (list of np.array's) priors[j] is the prior distribution of the 
                      target's state at the jth time step

        '''
        if self.parent_node is None:
            # priors = [self.prior]
            #marginalize over previous states
            assert((self.gen_params.markov_order)*len(self.gen_params.state_space_tuple) == len(self.prior.shape)), ((self.gen_params.markov_order)*len(self.gen_params.state_space_tuple), len(self.prior.shape))
            axes_to_marginalize_over = tuple(range((self.gen_params.markov_order - 1)*len(self.gen_params.state_space_tuple)))
            marginal_prior = np.sum(self.prior, axis=axes_to_marginalize_over)
            priors = [marginal_prior]


        else:
            priors = self.parent_node.get_priors()
            # priors.append(self.prior)
            #marginalize over previous states
            assert((self.gen_params.markov_order)*len(self.gen_params.state_space_tuple) == len(self.prior.shape)), ((self.gen_params.markov_order)*len(self.gen_params.state_space_tuple), len(self.prior.shape))
            axes_to_marginalize_over = tuple(range((self.gen_params.markov_order - 1)*len(self.gen_params.state_space_tuple)))
            marginal_prior = np.sum(self.prior, axis=axes_to_marginalize_over)    
            priors.append(marginal_prior)            
        return priors

def normalize(pdf):
    pdf /= np.sum(pdf)
    return pdf


def discrete_bayes_update(likelihood, prior):
    """ 
    from https://github.com/rlabbe/filterpy/blob/master/filterpy/discrete_bayes/discrete_bayes.py
    Computes the posterior of a discrete random variable given a
    discrete likelihood and prior. In a typical application the likelihood
    will be the likelihood of a measurement matching your current environment,
    and the prior comes from discrete_bayes.predict().
    Parameters
    ----------
    likelihood : (np.array with shape state_space_tuple)
         array of likelihood values
    prior : (np.array with shape previous_dependent_states_shape)
        prior pdf.
    Returns
    -------
    posterior : ndarray, dtype=float
        Returns array representing the posterior.
    Examples
    --------
    .. code-block:: Python
        # self driving car. Sensor returns values that can be equated to positions
        # on the road. A real likelihood compuation would be much more complicated
        # than this example.
        likelihood = np.ones(len(road))
        likelihood[road==z] *= scale_factor
        prior = predict(posterior, velocity, kernel)
        posterior = update(likelihood, prior)
    """
    #prior and likelihood may not have shame shape (prior will have more dimensions when markov_order > 1),
    #but element-wise multiplication in this case iterates over first dimension and multiplies to last dimensions as desired

    posterior = prior * likelihood
    return normalize(posterior)


class Particle:
    def __init__(self, parent_particle, generative_parameters, importance_weight=None, log_importance_weight_normalization=0.0):
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

        if importance_weight is None:
            self.importance_weight = 1.0/N_PARTICLES
        else:
            self.importance_weight = importance_weight

        #log of the product of each importance weight normalization 
        self.log_importance_weight_normalization = log_importance_weight_normalization

        #GenerativeParameters object storing parameters used for data generation and inference
        self.generative_parameters = generative_parameters

    def create_child_particle(self, child_imprt_weight, measurements, measurement_associations):
        '''
        Inputs:
        - child_imprt_weight: (float) the importance weight of the child particle
        - measurements: (list of measurements) contains measurements to update target states with
                        NOTE: this list of measurements should be passed in the same order to every
                        particle because particles may share targets and the measurement indices 
                        must be consistent for TargetStateNode updates
        - measurement_associations: (list of ints) measurement_associations[i] = j specifies that
            specifies that the ith measurement in measurements should be used to update the jth
            target in self.targets
        '''
        child_particle = Particle(parent_particle=self, generative_parameters=self.generative_parameters, importance_weight=child_imprt_weight,\
                                  log_importance_weight_normalization=self.log_importance_weight_normalization)
        assert(len(measurements) == len(measurement_associations))
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

        probs = construct_exact_sampling_matrix(targets=particle.targets, measurements=cur_time_step_measurements)
        print 'original probs:'
        print probs
        for m_idx in range(M):
            for t_idx in range(T):
                if m_idx != t_idx:
                    probs[m_idx][t_idx] = .0000000000000000000001
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
        child_particle = particle_set[sampled_association.matrix_index].create_child_particle(\
            child_imprt_weight=sampled_association.complete_assoc_probability,\
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

        # print "particle_prior_prob =", particle_prior_prob
        cur_a_matrix = associationMatrix(matrix=probs, M=M, T=T,\
            conditional_birth_probs=[0.0 for i in range(M)], conditional_death_probs=[0.0 for i in range(T)],\
            prior_prob=particle_prior_prob)
        all_association_matrices.append(cur_a_matrix)

    sampled_associations = multi_matrix_sample_associations_without_replacement(num_samples=N_PARTICLES, all_association_matrices=all_association_matrices)


    new_particle_set = []
    for sampled_association in sampled_associations:
        child_particle = particle_set[sampled_association.matrix_index].create_child_particle(\
            child_imprt_weight=sampled_association.complete_assoc_probability,\
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
        assert((cur_log_probs <= .000001).all()), (cur_log_probs)

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

    #5. For each of the most likely assignments, create a new particle that is a copy of its particle GROUP, 
    # and associate measurements / kill targets according to assignment.
    prv_cost=-np.inf
    for (idx, (cur_cost, cur_assignment, cur_particle_idx)) in enumerate(best_assignments):
        assert(cur_cost >= prv_cost or np.allclose(cur_cost, prv_cost)), (cur_cost, prv_cost, idx) #make sure costs are decreasing in best_assignments
        prv_cost = cur_cost
        assert(cur_cost < INFEASIBLE_COST)
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
            child_imprt_weight=np.exp(new_particle_log_importance_weight),\
            measurements=cur_time_step_measurements,\
            measurement_associations=meas_grp_associations)

        assert(T == len(child_particle.targets))

        new_particle_set.append(child_particle)
        assert(child_particle.parent_particle != None)

    return new_particle_set


def normalize_importance_weights(particle_set):
    # print 'HI!!!'
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

def get_gt_likelihood(gen_params, all_measurements):
    log_prob_of_all_targets = 0
    print 'get_gt_likelihood called'
    for target_idx in range(gen_params.num_targets):
        cur_gen_params = generate_data.GenerativeParameters(gen_params.num_time_steps, gen_params.state_space, \
            gen_params.previous_dependent_states_shape, gen_params.all_initial_state_probabilities[target_idx:target_idx+1], \
            gen_params.transition_probabilities[target_idx:target_idx+1], gen_params.emission_probabilities[target_idx:target_idx+1], \
            gen_params.markov_order, gen_params.num_targets)
        with io.capture_output() as captured:
            (all_target_posteriors, all_target_priors, most_probable_particle) = run_tracking([all_measurements[target_idx]], tracking_method='MHT', generative_parameters=cur_gen_params, n_particles=1, use_group_particles='False')
        log_prob_of_all_targets += most_probable_particle.log_importance_weight_normalization + np.log(most_probable_particle.importance_weight)
        print(most_probable_particle.log_importance_weight_normalization + np.log(most_probable_particle.importance_weight))
    return log_prob_of_all_targets


def run_tracking(all_measurements, tracking_method, generative_parameters, n_particles, use_group_particles):
    """
    Measurement class designed to only have 1 measurement/time instance
    Input:
    - all_measurements: (list of list of measurements) all_measurements[i][j] is the ith target's measurement at the jth time instance
        NOTE: all_measurements[0] should be in the same order as all_initial_state_probabilities so that each target has the correct prior
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
    """
    global N_PARTICLES
    N_PARTICLES = n_particles
    single_initial_particle = Particle(parent_particle=None, generative_parameters=generative_parameters, importance_weight=1.0)
    all_measurements_by_time = convert_measurements_by_target_to_by_time(all_measurements=all_measurements)
    print "all_measurements_by_time:"
    print all_measurements_by_time

    initial_associations_unknown=True
    if initial_associations_unknown:
        T = len(all_measurements_by_time[0]) #number of targets
        for target_idx in range(T):
            root_target_node = TargetStateNode(gen_params=generative_parameters, parent_node=None, target_idx=target_idx)            
            root_target_node.initialize_root_target_unknown_association(target_idx=target_idx)
            single_initial_particle.targets.append(root_target_node)
        particle_set = [single_initial_particle]

        for target in particle_set[0].targets:
            print("check prior initialized with max =", np.max(target.prior), "at index", np.argmax(target.prior))


        for cur_time_step_measurements in all_measurements_by_time:    
            if tracking_method == 'exact_sampling':
                particle_set = exact_sampling_step(particle_set, cur_time_step_measurements)
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

    else:
        for target_idx, cur_measurment in enumerate(all_measurements_by_time[0]):
            root_target_node = TargetStateNode(gen_params=generative_parameters, parent_node=None, target_idx=target_idx)
            root_target_node.initialize_root_target_with_known_association(measurement=cur_measurment, target_idx=target_idx)
            single_initial_particle.targets.append(root_target_node)
        particle_set = [single_initial_particle]

        for cur_time_step_measurements in all_measurements_by_time[1:]:    
            if tracking_method == 'exact_sampling':
                particle_set = exact_sampling_step(particle_set, cur_time_step_measurements)
            elif tracking_method == 'MHT':
                particle_set = MHT_step(particle_set, cur_time_step_measurements)
            elif tracking_method == 'gt_assoc':
                particle_set = gt_assoc_step(particle_set, cur_time_step_measurements)
            else:
                assert(False)
               
            normalize_importance_weights(particle_set)
            if use_group_particles:
                particle_set = group_particles(particle_set)

    #we sample without replacement with exact sampling, so just find the highest weight sample, no marginalization
    most_probable_particle = None
    for particle in particle_set:
        if most_probable_particle is None or particle.importance_weight > most_probable_particle.importance_weight:
            most_probable_particle = particle
    all_target_posteriors = most_probable_particle.get_all_target_posteriors()
    all_target_priors = most_probable_particle.get_all_target_priors()

    return (all_target_posteriors, all_target_priors, most_probable_particle)

def distribution_distance_metric(distributionA, distributionB):
    '''
    For each discrete element compute the difference between probabilities for distributionA and distributionB
    and divide by the larger of the two probabilties.  Return the largest of these values over all elements
    '''
    # return np.max(np.abs(distributionA-distributionB)/np.maximum(distributionA,distributionB))
    return np.max(np.abs(distributionA-distributionB))

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

def particles_are_similar(particleA, particleB, distance_threshold=.01):
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
            cost_matrix[a_idx, b_idx] = distribution_distance_metric(targetA.posterior, targetB.posterior)

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
    # print "target.get_likelihood(measurement):", target.get_likelihood(measurement)
    # print "target.prior:", target.prior

    association_likelihood = np.sum(target.prior * target.get_likelihood(measurement)) #element wise product, then sum elements
    return association_likelihood

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
    - probs: numpy matrix with dimensions (M+T)x(M+T) with probabilities:
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
    # log_probs *= -1*INFEASIBLE_COST #setting all entries to very negative value
    log_probs *= -np.inf #setting all entries to very negative value


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


    return log_probs



if __name__ == "__main__":
    np.random.seed(0)


    num_time_steps = 4
    state_space = np.array((20))
    measurement_space = np.array((20))
    markov_order = 1
    num_targets = 10

    N_PARTICLES = 50
    use_group_particles = True
    method = 'exact_sampling'
    # method = 'MHT'

    (all_states, all_measurements, gen_params) = generate_data.get_parameters_and_data(num_time_steps, state_space,\
        measurement_space, markov_order, num_targets)

    # with io.capture_output() as captured:
    (all_target_posteriors, all_target_priors, most_probable_particle) = run_tracking(all_measurements, tracking_method=method, generative_parameters=gen_params, n_particles=N_PARTICLES, use_group_particles=use_group_particles)

    print("most_probable_particle.importance_weight:", most_probable_particle.importance_weight)
    print("most_probable_particle.log_importance_weight_normalization:", most_probable_particle.log_importance_weight_normalization)
    print("most probable particle log_prob:", most_probable_particle.log_importance_weight_normalization + np.log(most_probable_particle.importance_weight))
    print("most probable particle prob:", np.exp(most_probable_particle.log_importance_weight_normalization + np.log(most_probable_particle.importance_weight)))

    print("ground truth, log_prob_of_all_targets =", get_gt_likelihood(gen_params, all_measurements))




