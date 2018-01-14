data : data/raw/spikesorted \
				data/external/selected_neurons
			python initialize_shortcuts.py
			python raw_to_spikes_behavior.py
			python epoch_spikes.py
