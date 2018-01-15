data : data/raw/spikesorted \
				data/external/selected_neurons
			python src/data/initialize_shortcuts.py
			python src/data/raw_to_spikes_behavior.py
			python src/data/epoch_spikes.py
