from sacred import Experiment
from sacred.observers import FileStorageObserver

from ml_collections import ConfigDict

from polaris.experience import EpisodeCallbacks

# Create a new experiment
exp_name = 'VMPO_lunarlander'
exp_path = "experiments/" + exp_name
ex = Experiment(exp_name)

# Add an observer to save the experiment's results
#ex.observers.append(FileStorageObserver('sacred_runs'))

# Define the default configuration
@ex.config
def my_config():
    from polaris.environments.example import PolarisLunarLander
    env_obj = PolarisLunarLander()
    env_obj.register()
    env = env_obj.env_id
    del env_obj

    num_workers = 64
    policy_path = 'polaris.policies.VMPO'
    model_path = 'polaris.models.fc_model'
    policy_class = 'VMPO'
    model_class = 'FCModel'
    trajectory_length = 32
    train_batch_size = 1024
    max_queue_size = train_batch_size * 10
    max_seq_len = 32

    default_policy_config = {
            'discount'    : 0.999, #0.997
            'entropy_cost': 1e-2,
            'popart_std_clip': 1e-2,
            'popart_lr': 2e-2,
            'grad_clip': 4.,
            'lr'              : 0.0005,
            'rms_prop_rho'    : 0.99,
            'rms_prop_epsilon': 1e-5,
            'fc_dims'         : [128, 128],

            # VMPO
            'trust_region_speed': 10.,
            'initial_trust_region_coeff': 5.,
            'trust_region_eps'          : 1e-2,

            'temperature_speed': 10.,
            'initial_temperature'       : 1.,
            'temperature_eps'           : 0.1,

            'target_update_freq'        : 1000,
            'top_sample_frac'           : 0.5,

            'baseline_weight'           : 0.5, #np.log(dummy_ssbm.action_space.n)
        }

    policy_params = [dict(
        name="VMPO_policy1",
        config=default_policy_config.copy()
    ),
    ]

    tensorboard_logdir = 'lunarlander_VMPO'
    report_freq = 50
    episode_metrics_smoothing = 0.9

    env_config = {}

    checkpoint_config = dict(
        checkpoint_frequency=100,
        checkpoint_path=exp_path,
        stopping_condition={"environment_steps": 1e6},
        keep=5,
    )

    episode_callback_class = EpisodeCallbacks

# Define a simple main function that will use the configuration
@ex.main
def main(_config):
    import tensorflow as tf
    tf.compat.v1.enable_eager_execution()
    from polaris.trainers import AsyncTrainer


    # TODO: seeding
    # Access the configuration using _config
    c = ConfigDict(_config)
    print("Experiment Configuration:")
    print(c)
    trainer = AsyncTrainer(c, restore=False)
    trainer.run()


# Run the experiment
if __name__ == '__main__':
    ex.run_commandline()