# start_molecule = None
start_molecule = 'CC(C)NCc1cccc(-c2cccc(-c3nc4cc(F)ccc4[nH]3)c2)c1'
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 2000
optimizer = "Adam"
# optimizer = "SGD"
polyak = 0.995
atom_types = ["C", "O", "N"]
max_steps_per_episode = 40
# max_steps_per_episode = 20
allow_removal = True
allow_no_modification = True
allow_bonds_between_rings = False
# allowed_ring_sizes = [3, 4, 5, 6]
allowed_ring_sizes = [3, 5, 6]
replay_buffer_size = 4000
learning_rate = 1e-4
gamma = 0.95
fingerprint_radius = 3
fingerprint_length = 2048
discount_factor = 0.9
bde_factor = 0.9
ip_factor = 0.8
lru_cache_capacity = 128
# etkdg_max_attempts_cache = 7
# etkdg_max_attempts_uncache = 7
reward_of_invalid_mol = -1000

transformer_params = {
    'd_atom': 27, # 26 MAT + remaining_steps
    'd_model': 1024,
    'N': 8,
    'h': 16,
    'N_dense': 1,
    'lambda_attention': 0.33, 
    'lambda_distance': 0.33,
    'leaky_relu_slope': 0.1, 
    'dense_output_nonlinearity': 'relu', 
    'distance_matrix_kernel': 'exp', 
    'dropout': 0.0,
    'aggregation_type': 'mean'
}

