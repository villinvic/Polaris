from sacred import Experiment
from sacred.observers import FileStorageObserver

from ml_collections import ConfigDict

# Create a new experiment
ex = Experiment('VMPO on cartpole')

# Add an observer to save the experiment's results
ex.observers.append(FileStorageObserver('sacred_runs'))

# Define the default configuration
@ex.config
def my_config():
    from polaris.environments.example import PolarisCartPole
    env_obj = PolarisCartPole()
    env_obj.register()
    env = env_obj.env_id
    del env_obj

    num_workers = 6
    policy_path = 'polaris.policies.VMPO'
    model_path = 'polaris.models.fc_model'
    policy_class = 'VMPO'
    model_class = 'FCModel'
    trajectory_length = 256
    train_batch_size = 1024
    max_queue_size = train_batch_size * 10
    max_seq_len = 256

    default_policy_config = {
            'discount'    : 0.99,
            'gae_lambda'  : 1.0,
            'entropy_cost': 0.0,
            'popart_std_clip': 1e-2,
            'popart_lr': 0.2,
            'grad_clip': 5.,
            'lr'              : 0.0005,
            'rms_prop_rho'    : 0.99,
            'rms_prop_epsilon': 1e-5,
            'fc_dims'         : [64, 64],

            # VMPO
            'trust_region_speed': 10.,
            'initial_trust_region_coeff': 3.,
            'trust_region_eps': 0.01,

            'initial_temperature': 1.,
            'temperature_eps': 0.1,

            'target_update_freq': 100,

            'top_sample_frac': 0.5,
        }

    policy_params = [dict(
        name="IMPALA_policy",
        config=default_policy_config
    )]
    tensorboard_logdir = 'cartpole_VMPO'
    report_freq = 5
    episode_metrics_smoothing = 0.9

# Define a simple main function that will use the configuration
@ex.main
def main(_config):
    import tensorflow as tf
    tf.compat.v1.enable_eager_execution()
    from polaris.trainers.async_trainer import Trainer


    # TODO: seeding
    # Access the configuration using _config
    c = ConfigDict(_config)
    print("Experiment Configuration:")
    print(c)
    trainer = Trainer(c)
    trainer.run()


# Run the experiment
if __name__ == '__main__':
    ex.run_commandline()