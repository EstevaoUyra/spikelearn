.PHONY: all data clean

data : data/raw/spikesorted \
				data/external/selected_neurons
			python src/data/0.initialize_shortcuts.py
			python src/data/1.raw_to_spikes_behavior.py
			python src/data/2.behav_stats.py
			python src/data/3.epoch_spikes.py
			python src/data/4.kernel_smoothing.py

test-data :
			echo NotImplemented

test-package :
			echo NotImplemented

temp :
			python src/data/kernel_smoothing.py
			python src/models/tune_hyperparameters.py -clf XGboost elasticnet_SGD -o models/hyperopt/harmonic -ow -vvv
