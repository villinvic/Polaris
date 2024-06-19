from sacred import Experiment
from sacred.observers import FileStorageObserver

from ml_collections import ConfigDict

# Create a new experiment
ex = Experiment('A3C on cartpole')

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

    num_workers = 1
    policy_path = 'polaris.policies.A3C'
    model_path = 'polaris.models.fc_model'
    policy_class = 'A3CPolicy'
    model_class = 'FCModel'
    policy_config = {}
    batch_size = 16
    max_queue_size = 10
    train_batch_size = 256
    policy_params = [dict(
        name="A3C_policy"
    )]
    tensorboard_logdir = '~/polaris_logs'
    report_freq = 5
    model_config = {
        'lr': 0.0001,
        'rms_prop_rho': 0.99,
        'rms_prop_epsilon': 1e-5,
        'fc_dims': [32, 32]
    }
    policy_config = {
        'discount'    : 0.99,
        'gae_lambda'  : 1.0,
        'entropy_cost': 1e-3
    }

# Define a simple main function that will use the configuration
@ex.main
def main(_config):
    import tensorflow as tf
    tf.compat.v1.enable_eager_execution()
    from polaris.trainers.trainer import Trainer


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