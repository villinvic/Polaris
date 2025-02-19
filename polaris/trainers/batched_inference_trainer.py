import importlib
import time
from typing import Dict, Union, NamedTuple, AnyStr, Any

import numpy as np
import tree
from ml_collections import ConfigDict

from polaris.experience.timestep import TrajectoryBank, TrajectoryStep
from polaris.experience.worker_set import VectorisedWorkerSet
from polaris.checkpointing.checkpointable import Checkpointable

from polaris.environments.polaris_env import PolarisEnv
from polaris.policies.policy import Policy, PolicyParams, ParamsMap

from polaris.utils.metrics import MetricBank, GlobalCounter

import psutil


class PolicyState(NamedTuple):
    state: Any
    prev_action: Any
    prev_reward: Any



class BatchedInferenceTrainer(Checkpointable):
    def __init__(
            self,
            config: ConfigDict,
            restore: Union[bool, str] = False,
    ):
        """
        The BatchedInferenceTrainer class should be used for similar use cases to SyncTrainer's

        # TODO: this only works with one policy for now
        """

        self.config = config

        # Init environment
        self.env = PolarisEnv.make(self.config.env, env_index=-1, **self.config.env_config)

        # We can extend to multiple policy types if really needed, but won't be memory efficient
        policy_params = [
            PolicyParams(**ConfigDict(pi_params)) for pi_params in self.config.policy_params
        ]

        PolicylCls = getattr(importlib.import_module(self.config.policy_path), self.config.policy_class)
        self.policy_map: Dict[str, Policy] = {
            policy_param.name: PolicylCls(
                name=policy_param.name,
                action_space=self.env.action_space,
                observation_space=self.env.observation_space,
                config=self.config,
                policy_config=policy_param.config,
                options=policy_param.options,
                # For any algo that needs to track either we have the online model
                is_online=True,
            )
            for policy_param in policy_params
        }

        self.policy_0 = list(self.policy_map.keys())[0]

        self.params_map = ParamsMap(**{
            name: p.get_params() for name, p in self.policy_map.items()
        })

        self.trajectory_bank = TrajectoryBank(config)
        self.metricbank = MetricBank(
            report_freq=self.config.report_freq
        )
        self.metrics = self.metricbank.metrics

        super().__init__(
            checkpoint_config = config.checkpoint_config,

            components={
                "config": self.config,
                "params_map": self.params_map,
                "metrics": self.metrics,
            }
        )

        if restore:
            if isinstance(restore, str):
                self.restore(restore_path=restore)
            else:
                self.restore()

            # Need to pass the restored references afterward
            self.metricbank.metrics = self.metrics

            env_step_counter = "counters/" + GlobalCounter.ENV_STEPS
            if env_step_counter in self.metrics:
                GlobalCounter[GlobalCounter.ENV_STEPS] = self.metrics["counters/" + GlobalCounter.ENV_STEPS].get()

            for policy_name, policy in self.policy_map.items():
                policy.setup(self.params_map[policy_name])

        # init workers
        self.worker_set = VectorisedWorkerSet(
            config,
        )
        # send dummy stuff to get them to reset
        # TODO: handle multiple policies here
        # W
        # we need to keep track of who is playing on which env...
        self.worker_set.send(
                actions={
                    aid: np.array([self.env.action_space.sample()] * self.worker_set.num_envs())
                    for aid in self.env.get_agent_ids()
                },
                options={
                    aid: self.policy_map[self.policy_0].options
                    for aid in self.env.get_agent_ids()

                }
            )
        self.policy_states = {
            env_index: {
                aid: PolicyState(
                state=self.policy_map[self.policy_0].get_initial_state(),
                prev_action=self.env.action_space.sample(), # TODO: better initialisation maybe ?
                prev_reward=0.
            ) for aid in self.env.get_agent_ids()
            }
            for env_index in range(self.worker_set.num_envs())
        }
        self.startup_time = time.time()

        # self.obs = np.stack([step[0].obs for step in self.timesteps])
        # self.prev_action = np.stack([self.policy_states[step[0].env_index][0].prev_action for step in self.timesteps])
        # self.prev_reward = np.stack([self.policy_states[step[0].env_index][0].prev_reward for step in self.timesteps])
        # self.prev_state = tree.map_structure(lambda v: np.stack(v), [self.policy_states[step[0].env_index][0].state for step in self.timesteps])

    def training_step(self):
        """
        Executes one step of the trainer:
        1. Sends jobs to available environment workers.
        2. Gathers ready experience and metrics.
        3. Updates policies whose batches are ready.
        4. Reports experience and training metrics to the bank.
        """
        iter_start_time = time.time()

        timesteps, episode_metrics = self.worker_set.recv()

        # the policy must implement `compute_action_batch_with_extras`
        # Can we keep timesteps grouped by worker ? shape (w, num_env_per_w, ...)
        outs = {
            aid: self.policy_map[self.policy_0].model.compute_action_batch_with_extras(

                tree.map_structure(lambda *v: np.stack(v), *[step[aid].obs for step in timesteps]),
                np.stack([self.policy_states[step[aid].env_index][aid].prev_action for step in timesteps]),
                np.stack([self.policy_states[step[aid].env_index][aid].prev_reward for step in timesteps]),
                tree.map_structure(lambda *v: np.stack(v),
                                   *[self.policy_states[step[aid].env_index][aid].state for step in timesteps]),
            )
            for aid in self.env.get_agent_ids()
        }

        self.worker_set.send(
            actions={
                aid: outs[aid][0]
                for aid in self.env.get_agent_ids()
            },
            options={
                    aid: self.policy_map[self.policy_0].options
                    for aid in self.env.get_agent_ids()
                })

        infer_end_time = time.time()
        infer_dt = infer_end_time - iter_start_time

        for aid in self.env.get_agent_ids():


            # actions[aid] = self.prev_action
            # states = self.prev_state
            # action_logits = np.full((self.prev_action.shape[0], 2), fill_value=0., dtype=np.float32)
            # action_logps = np.zeros((self.prev_action.shape[0],), dtype=np.float32)
            # values = np.zeros((self.prev_action.shape[0],), dtype=np.float32)

            action, state, logits, logps, values = outs[aid]

            trajectory_steps = [TrajectoryStep(
                action=action[i],
                prev_action=self.policy_states[step[aid].env_index][aid].prev_action,
                action_logp=logps[i],
                action_logits=logits[i],
                prev_reward=self.policy_states[step[aid].env_index][aid].prev_reward,
                policy_id=self.policy_0, # TODO
                version=self.policy_map[self.policy_0].version,
                state=self.policy_states[step[aid].env_index][aid].state,
                values=values[i],
                obs=step[aid].obs,
                agent_id=step[aid].agent_id,
                episode_id=step[aid].episode_id,
                env_index=step[aid].env_index,
                episode_state=step[aid].episode_state,
                t=step[aid].t,
                reward=step[aid].reward,
                done=step[aid].done,
                #**step[aid]._asdict()
            ) for i, step in enumerate(timesteps)]

            for i, step in enumerate(timesteps):
                self.policy_states[step[aid].env_index][aid] = PolicyState(
                    state=state[i],
                    prev_action=action[i],
                    prev_reward=step[aid].reward
                )

            self.trajectory_bank.add(trajectory_steps)


        store_end_time = time.time()
        store_dt = store_end_time - infer_end_time


        # Training
        # Train once a batch is ready
        training_metrics = {}
        for pid, policy in self.policy_map.items():

            if self.trajectory_bank.is_ready(self.config.train_batch_size, policy):
                batch = self.trajectory_bank.get_batch(self.config.train_batch_size, policy)
                # TODO: check if we have of policy data here
                train_results = policy.train(batch)
                training_metrics[f"{pid}"] = train_results
                GlobalCounter.incr(GlobalCounter.STEP)
                self.params_map[pid] = policy.get_params()

        train_end_time = time.time()
        train_dt = train_end_time - store_end_time

        env_step_time = time.time()
        env_step_dt = env_step_time - train_end_time

        GlobalCounter[GlobalCounter.ENV_STEPS] += len(timesteps)
        GlobalCounter[GlobalCounter.NUM_EPISODES] += len(episode_metrics)

        if len(training_metrics)> 0:
            for pid, policy_training_metrics in training_metrics.items():
                self.metricbank.update(tree.flatten_with_path(policy_training_metrics), prefix=f"training/{pid}/",
                                       smoothing=self.config.training_metrics_smoothing)
        if len(episode_metrics) > 0:
            for metrics in episode_metrics:
                self.metricbank.update(tree.flatten_with_path(metrics), prefix=f"experience/",
                                       smoothing=self.config.episode_metrics_smoothing)

        metrics_dt = time.time() - env_step_time

        ram_info = psutil.virtual_memory()
        misc_metrics =  [
            ('trajectory_bank_size', self.trajectory_bank.size()),
            ('RAM_percent_usage', ram_info.percent),
            ('FPS', GlobalCounter[GlobalCounter.ENV_STEPS] / (time.time() - self.startup_time)),
            ('infer_time_ms', infer_dt * 1000),
            ('batching_time_ms', store_dt * 1000),
            ('train_time_ms', train_dt * 1000),
            ('env_step_time_ms', env_step_dt * 1000),
            ('metrics_time_ms', metrics_dt * 1000),
        ]

        self.metricbank.update(
            misc_metrics
            , prefix="misc/", smoothing=0.9
        )

    def run(self):
        """
        Runs the trainer in a loop, until the stopping condition is met or a C^ is caught.
        """

        try:
            while not self.is_done(self.metricbank):
                self.training_step()
                self.metricbank.report()
                self.checkpoint_if_needed()
        except KeyboardInterrupt:
            print("Caught C^. Terminating...")









