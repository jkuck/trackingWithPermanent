import numpy as np
import perform_tracking
import generate_data
import plotting
import os

from IPython.utils import io
import pickle
import matplotlib.pyplot as plt
COLORMAP = plt.cm.gist_ncar



def run_experiment_over_parameter_set(num_time_steps, state_space, hidden_state_space, observed_state_space, measurement_space, markov_order, num_targets):
	# (all_states, all_measurements, gen_params) = generate_data.get_parameters_and_data(num_time_steps, state_space, measurement_space,
	# 	markov_order, num_targets)

	(all_states, all_measurements, gen_params) = generate_data.get_parameters_and_data_targets_identical_plus_noise(num_time_steps, state_space, hidden_state_space, observed_state_space, measurement_space,
		markov_order, num_targets)


	experiment_name = 'compare_group_particles_lowTransitionNoise_hiddenStateSize2_genDataTargetsIdentical_%dtargets2' % num_targets
	experiment_folder = './' + experiment_name + '/'
	os.mkdir(experiment_folder)

	f = open(experiment_folder + 'input_data.pickle', 'w')
	pickle.dump((gen_params, all_measurements), f)
	f.close()  

	gt_likelihood = perform_tracking.get_gt_likelihood(gen_params, all_measurements)
	f = open(experiment_folder + 'log_likelihoods.txt', 'a')
	f.write('ground truth log_likelihood = %f\n' % gt_likelihood)
	f.close()


	# for (n_particles, method) in [(10, 'MHT'), (10, 'exact_sampling')]:
	for (n_particles, method) in [(10, 'MHT'), (10, 'exact_sampling'), (100, 'MHT'), (25, 'exact_sampling'), (1000, 'MHT'), (50, 'exact_sampling'), (10000, 'MHT'), (51, 'exact_sampling'), (50000, 'MHT'), (53, 'exact_sampling'), (100000, 'MHT'), (51, 'exact_sampling'), (500000, 'MHT'), (54, 'exact_sampling')]:
	    for use_group_particles in [True, False]:
	        cur_experiment = "%s_particles=%d_use_group_particles=%s" % (method, n_particles, use_group_particles)
	        print("cur_experiment:", cur_experiment)
	        with io.capture_output() as captured:
	            (all_target_posteriors, all_target_priors, most_probable_particle) = perform_tracking.run_tracking(all_measurements, tracking_method=method, generative_parameters=gen_params, n_particles=n_particles, use_group_particles=use_group_particles)

	        most_probable_particle_log_prob = most_probable_particle.log_importance_weight_normalization + np.log(most_probable_particle.importance_weight)

	        f = open(experiment_folder + '%s_results.pickle'%cur_experiment, 'w')
	        pickle.dump((all_target_posteriors, all_target_priors, most_probable_particle, most_probable_particle_log_prob), f)
	        f.close()

	        f = open(experiment_folder + 'log_likelihoods.txt', 'a')
	        f.write(cur_experiment + ' log_likelihood = %f\n' % most_probable_particle_log_prob)
	        f.close()

	#         for time_step in [5, 11, 17]:
	        for time_step in range(1, num_time_steps):            
	            for target_idx in range(num_targets):
	                plotting.bar_plot(np.sum(all_target_priors[target_idx][time_step], axis=0), ylim=(0,1.1), title='Prior Distribution over Target States', c=COLORMAP(target_idx/num_targets))
	                plt.axvline(all_states[target_idx][time_step][1] -.3 + .5*target_idx/num_targets, lw=1.5, c=COLORMAP(target_idx/num_targets))
	            plt.tight_layout()
	            plt.savefig(experiment_folder + cur_experiment + 'Priors_timeStep%d.pdf'%time_step)
	            plt.close()

	            for target_idx in range(num_targets):
	                plotting.bar_plot(np.sum(all_target_posteriors[target_idx][time_step+1], axis=0), ylim=(0,1.1), title='Posterior Distribution over Target States', c=COLORMAP(target_idx/num_targets))
	                plt.axvline(all_states[target_idx][time_step][1] -.3 + .5*target_idx/num_targets, lw=1.5, c=COLORMAP(target_idx/num_targets))
	            plt.tight_layout()
	            plt.savefig(experiment_folder + cur_experiment + 'Posteriors_timeStep%d.pdf'%time_step)

def replot_previous_experiment_data():
	experiment_name = 'test_experiment'
	experiment_folder = './' + experiment_name + '/'

	f = open(experiment_folder + 'input_data.pickle', 'r')
	(gen_params, all_measurements) = pickle.load(f)
	f.close()  

	# for (n_particles, method) in [(10, 'MHT'), (10, 'exact_sampling')]:
	for (n_particles, method) in [(10, 'MHT'), (10, 'exact_sampling'), (100, 'MHT'), (20, 'exact_sampling'), (1000, 'MHT'), (50, 'exact_sampling'), (10000, 'MHT'), (100, 'exact_sampling')]:
	    cur_experiment = "%s_particles=%d" % (method, n_particles)

	    f = open(experiment_folder + '%s_results.pickle'%cur_experiment, 'r')
	    (all_target_posteriors, all_target_priors, most_probable_particle, most_probable_particle_log_prob) = pickle.load(f)
	    f.close()
	    
	    print(cur_experiment, most_probable_particle.importance_weight)
	    
	#     for time_step in [5, 11, 17]:
	#         for target_idx in range(NUM_TARGETS):
	#             plotting.bar_plot(all_target_priors[target_idx][time_step], ylim=(0,1.1), title='Prior Distribution over Target States', c=COLORMAP(target_idx/NUM_TARGETS))
	#             plt.axvline(all_states[target_idx][time_step], lw=1.5, c=COLORMAP(target_idx/NUM_TARGETS))
	#         plt.tight_layout()
	#         plt.savefig(experiment_folder + cur_experiment + 'Priors_timeStep%d.pdf'%time_step)
	#         plt.close()

	#         for target_idx in range(NUM_TARGETS):
	#             plotting.bar_plot(all_target_posteriors[target_idx][time_step], ylim=(0,1.1), title='Posterior Distribution over Target States', c=COLORMAP(target_idx/NUM_TARGETS))
	#             plt.axvline(all_states[target_idx][time_step], lw=1.5, c=COLORMAP(target_idx/NUM_TARGETS))
	#         plt.tight_layout()
	#         plt.savefig(experiment_folder + cur_experiment + 'Posteriors_timeStep%d.pdf'%time_step)




if __name__ == "__main__":
	np.random.seed(4)

	# num_time_steps = 6
	# state_space = np.array((20,20))
	# measurement_space = np.array((20))
	# markov_order = 1
	# num_targets = 20

	num_time_steps = 4
	hidden_state_space = np.array((2))
	observed_state_space = np.array((20))
	state_space = np.array((2, 20))
	measurement_space = np.array((20))
	markov_order = 1
	num_targets = 15	

	run_experiment_over_parameter_set(num_time_steps, state_space, hidden_state_space, observed_state_space, measurement_space, markov_order, num_targets)
	        