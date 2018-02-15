data : data/raw/spikesorted \
				data/external/selected_neurons
			python src/data/initialize_shortcuts.py
			python src/data/raw_to_spikes_behavior.py
			python src/data/behav_stats.py
			python src/data/epoch_spikes.py
			python src/data/kernel_smoothing.py

test-data :
			echo NotImplemented

test-package :
			echo NotImplemented

temp :
			python src/models/tune_hyperparameters.py -clf XGboost elasticnet_SGD -o models/hyperopt/harmonic -ow -vvv

clear-cache :
			find . | \
			grep -E "(__pycache__|\.pyc$)" | \
			xargs rm -rf
