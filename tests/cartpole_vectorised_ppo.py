import wandb
from sacred import Experiment
from ml_collections import ConfigDict

from polaris.experience import EpisodeCallbacks
from polaris.environments.example import PolarisCartPole

exp_name = 'cartpole'
exp_path = "experiments/" + exp_name
ex = Experiment(exp_name)


@ex.config
def cfg():

    env_config = dict()

    env = PolarisCartPole.env_id

    num_workers = 16 # the +1 is for the rendering window.
    policy_path = 'polaris.policies.PPO'
    model_path = 'polaris.models.cartpole'
    policy_class = 'PPO'
    model_class = 'CartPoleModel'

    # the episode_length is fixed, we should train over full episodes.
    trajectory_length = 128
    max_seq_len = trajectory_length
    num_envs_per_worker = 16
    inference_batch_size = (num_workers * num_envs_per_worker)

    train_batch_size = trajectory_length * (num_workers * num_envs_per_worker)
    n_epochs=8
    minibatch_size = train_batch_size // 8 # we are limited in GPU RAM ... A bigger minibatch leads to stabler updates.


    default_policy_config = {

        'discount': 0.99,
        'gae_lambda': 1., # coefficient for Bias-Variance tradeoff in advantage estimation. A smaller lambda may speed up learning.
        'entropy_cost': 0.01, # encourages exploration
        'lr': 3e-4, #5e-4

        'grad_clip': 1.,
        'ppo_clip': 0.2, # smaller clip coefficient will lead to more conservative updates.
        'baseline_coeff': 0.01,
        'initial_kl_coeff': 1.,
        'kl_target': 0.05,
        "vf_clip": 10.
        }

    policy_params = [{
        "name": "agent",
        "config": default_policy_config
    }]


    compute_advantages_on_workers = True
    wandb_logdir = 'wandb_logs'
    report_freq = 1
    episode_metrics_smoothing = 0.95
    training_metrics_smoothing = 0.8

    checkpoint_config = dict(
        checkpoint_frequency=50,
        checkpoint_path=exp_path,
        stopping_condition={"environment_steps": 1e10},
        keep=4,
    )

    restore = False

    episode_callback_class = EpisodeCallbacks

@ex.automain
def main(_config):
    import ray
    ray.init()

    import tensorflow as tf
    tf.compat.v1.enable_eager_execution()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    config = ConfigDict(_config)
    PolarisCartPole(**config["env_config"]).register()

    wandb.init(
        config=_config,
        project="deepred",
        mode='online',
        group="debug",
        name="cartpole",
        notes=None,
        dir=config["wandb_logdir"]
    )

    from polaris.trainers.batched_inference_trainer import BatchedInferenceTrainer

    trainer = BatchedInferenceTrainer(config, restore=config["restore"])
    trainer.run()