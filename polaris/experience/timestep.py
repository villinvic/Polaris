import time
from collections import defaultdict
from typing import NamedTuple, Any, List, Dict, Mapping, Iterator, Tuple

from colorama import Fore
import numpy as np
import tree
from ml_collections import ConfigDict
from tqdm import tqdm

from polaris.experience import SampleBatch
from polaris.policies import Policy
from polaris.policies.utils.gae import batched_gae


class TimeStep(Mapping):

    def __init__(
            self,
            *args,
            obs: Any = None,
            agent_id: Any = None,
            episode_id: int = None,
            env_index: int = None,
            episode_state: "EpisodeState" = None,
            t: int = None,
            reward: float = 0,
            done: bool = False,
            **kwargs
    ):
        if len(args) == 1:
            for k, v in args[0]:
                setattr(self, k, v)
        else:
            self.obs = obs
            self.agent_id = agent_id
            self.episode_id = episode_id
            self.env_index = env_index
            self.episode_state = episode_state
            self.t = t
            self.reward = reward
            self.done = done

            self.action = None
            self.prev_action = None
            self.action_logp = None
            self.action_logits = None
            self.prev_reward = None
            self.policy_id = None
            self.version = None
            self.state = None
            self.values = None

            self.add_policy_data(**kwargs)

        #print(list((key, self[key]) for key in self))


    def add_policy_data(
            self,
            action: Any | None = None,
            prev_action: Any | None = None,
            action_logp: Any | None = None,
            action_logits: Any | None = None,
            prev_reward: Any | None = None,
            policy_id: Any | None = None,
            version: Any | None = None,
            state: Any | None = None,
            values: Any | None = None,
    ):
        self.action = action
        self.prev_action = prev_action
        self.action_logp = action_logp
        self.action_logits = action_logits
        self.prev_reward = prev_reward
        self.policy_id = policy_id
        self.version = version
        self.state = state
        self.values = values

    # Mapping interface methods
    def __getitem__(self, key: str) -> Any:
        # Return the value for the given key
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"{key} not found in TimeStep")

    def __iter__(self) -> Iterator[str]:
        # Iterate over all attributes
        return (key for key in self.__dict__ if not key.startswith("_") and self[key] is not None)

    def __len__(self) -> int:
        # Count all attributes
        return len(self.__dict__)

    def __repr__(self) -> str:
        return f"TimeStep({self.__dict__})"

    def build_buffer(
            self,
            batch_dims: Tuple,
    ):
        def leaves_to_numpy(
                value,
        ):
            if isinstance(value, np.ndarray):
                return np.zeros(batch_dims + value.shape, dtype=value.dtype)
            elif isinstance(value, (np.float32, float)) or value is None:
                return np.zeros(batch_dims, dtype=np.float32)
            elif isinstance(value, (np.int32, np.int16, int)):
                return np.zeros(batch_dims, dtype=np.int32)
            elif isinstance(value, bool):
                return np.zeros(batch_dims, dtype=np.bool_)
            elif isinstance(value, str):
                return np.zeros(batch_dims, dtype='<U20')
            else:
                return np.zeros(batch_dims, dtype=object)

        return TimeStepBuffer(tree.map_structure(
            leaves_to_numpy,
            self
        ))

    def to_dict(self) -> Dict:
        return self.__dict__


class TimeStepBuffer(TimeStep):

    def __init__(
            self,
            buffer
    ):
        super().__init__(**buffer)

        # keeps track of the index for each trajectory
        self.indices = tree.map_structure(lambda v: 0, buffer)

        # buffer of shape (traj_len, num_workers * num_envs_per_worker, ...)

    def store(
            self,
            timestep: List[TimeStep]
    ):
        # timestep
        pass


class Trajectory(NamedTuple):
    steps: TimeStep # list of stacked trajectory_step
    seq_lens: List[int]
    bootstrap_value: Any

    @classmethod
    def from_steps(
            cls,
            steps,
            seq_lens,
            bootstrap_value,
    ):
        return cls(
            tree.map_structure(lambda *arrays: np.stack(arrays), *steps),
            seq_lens=seq_lens,
            bootstrap_value=bootstrap_value
        )

    @staticmethod
    def stack(
            trajectories,
            time_major=True,
    ) -> Dict:
        axis = 1 if time_major else 0
        def stack(*arr):
            if isinstance(arr[0], np.ndarray):
                return np.stack(arr, axis=axis)
            return np.stack(arr)
        return tree.map_structure(stack, *trajectories)

    def to_sample_batch(
            self,
    ) -> Dict:
        # we assume we already have a stack of trajectories here
        batch = self.steps.to_dict()
        batch[SampleBatch.SEQ_LENS] = self.seq_lens
        batch[SampleBatch.BOOTSTRAP_VALUE] = self.bootstrap_value
        return batch


class TrajectoryBankVisualiser:
    def __init__(
            self,
            config: ConfigDict,
    ):
        self.config = config

        # Progress bar for sample collection speed
        self.sample_speed_bar = tqdm(
            total=config.train_batch_size,
            desc="Samples Collected",
            position=0,
            leave=True,
            unit=" frames",
            bar_format="%s{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed} | {rate_fmt}]%s" % (Fore.CYAN, Fore.RESET)
        )

        # Progress bars for trajectories per policy
        self.num_policy_bars = 0
        self.policy_bars = {}

    def make_policy_bar(self, policy_name):
        self.policy_bars[policy_name] = tqdm(
                total=self.config.train_batch_size // self.config.trajectory_length,
                desc=f"Train Batch [{policy_name}]",
                position=self.num_policy_bars + 1,
                leave=True,
                bar_format="%s{l_bar}{bar:20}| {n_fmt}/{total_fmt} Trajectories [{percentage:3.0f}]%s" % (Fore.LIGHTBLUE_EX, Fore.RESET)
            )
        self.num_policy_bars += 1

    def update_sample_speed(self, num_samples):
        self.sample_speed_bar.update(num_samples)
        if self.sample_speed_bar.n >= self.sample_speed_bar.total:
            self.sample_speed_bar.reset()

    def update_policy_bars(self, ready):
        for policy_id, trajectories in ready.items():
            if policy_id not in self.policy_bars:
                # TODO: remove unused policy ids as well.
                self.make_policy_bar(policy_id)

            self.policy_bars[policy_id].n = len(trajectories)

            self.policy_bars[policy_id].refresh()

    def close_bars(self):
        self.sample_speed_bar.close()
        for bar in self.policy_bars.values():
            bar.close()

    def reset(self):
        self.close_bars()
        self.__init__(self.config)



class TrajectoryBank:
    """
    - stack data w.r.t. policy id, episode id
    - once we reached the truncation limit, or a done, move to next bit
    - can be used to grab (and optionally remove) a batch

    # TODO: implement for recurrent policies !!! We implemented this supposing policies were not recurrent
    """

    def __init__(
            self,
            config: ConfigDict,
    ):
        self.trajectory_length = config.trajectory_length
        # (pid, eid) -> [
        #  [batch 1],
        #  [batch 2],
        #  etc...
        #]
        # lets try with lists for now
        self.data = defaultdict(list)
        #self.data = defaultdict(lambda: np.empty((self.trajectory_length,), dtype=object))
        self.ready = defaultdict(list)

        self.timeout = 60 * 2
        self.timestamps = {}  # Tracks last update time per key, some episodes may crash, so we give up their batch.

        self.progress = TrajectoryBankVisualiser(config)

    def add(
            self,
            step: TimeStep
    ):
        """
        Add a list of `TrajectoryStep` objects to the trajectory buffer.
        """
        key = (step.policy_id, step.agent_id, step.env_index)
        if len(self.data[key]) >= self.trajectory_length:
            # trajectory is filled
            # TODO: see if boostrapping like that is fine computationally
            steps = self.data.pop(key)

            if steps[-1].done:
                boostrap_value = 0.
            else:
                boostrap_value = step.values

            # TODO: rrn seq lens
            self.ready[step.policy_id].append(Trajectory.from_steps(steps, self.trajectory_length, bootstrap_value=boostrap_value))

            del self.timestamps[key]


        self.data[key].append(step)
        self.timestamps[key] = time.time()  # Update the timestamp for this key

    def cleanup_stale_entries(self):
        """
        Remove keys from the data dictionary if no data has been received for `timeout` seconds.
        """
        current_time = time.time()
        stale_keys = [key for key, last_update in self.timestamps.items()
                      if current_time - last_update > self.timeout]

        for key in stale_keys:
            del self.data[key]
            del self.timestamps[key]

    def num_ready_samples(self, policy):
        return len(self.ready[policy.name]) * self.trajectory_length

    def is_ready(
            self,
            batch_size: int,
            policy: Policy,
    ):
        return self.num_ready_samples(policy) == batch_size

    def get_batch(
            self,
            batch_size: int,
            policy: Policy,
            cleanup: bool = True
    ):
        """
        Retrieves a batch of size `batch_size`. Optionally removes the retrieved data from the buffer.
        returns a SampleBatch of shape [T, B, ...] with T * B = batch_size,
        with advantages, values, and value targets computed
        """
        if batch_size % self.trajectory_length != 0:
            raise ValueError(f"trajectory_length must divide batch_size ! {batch_size} % {self.trajectory_length} != 0.")
        if not self.is_ready(batch_size, policy):
            raise ValueError(f"Batch is not ready !")

        # first build the batch
        batch = Trajectory.stack(self.ready.pop(policy.name)).to_sample_batch()

        # cleanup remaining stuff related to this policy in data as well (avoid off)

        batched_gae(batch, policy)

        if cleanup:
            keys_to_remove = {k for k in self.data.keys() if k[0] == policy.name}
            for key in keys_to_remove:
                del self.data[key]

        return batch

    def size(self):
        return len(self.ready) + len(self.data)

    def refresh_bars(self, num_samples):
        self.progress.update_sample_speed(num_samples)
        self.progress.update_policy_bars(self.ready)


class TrajectoryBankV2:
    """
    - stack data w.r.t. policy id, episode id
    - once we reached the truncation limit, or a done, move to next bit
    - can be used to grab (and optionally remove) a batch

    # TODO: implement for recurrent policies !!! We implemented this supposing policies were not recurrent
    """

    def __init__(
            self,
            config: ConfigDict,
    ):
        self.num_workers = config.num_workers
        self.num_envs_per_worker = config.num_envs_per_worker
        self.trajectory_length = config.trajectory_length

        self.batch: TimeStep | None = None


    def initialise_batch(
            self,
            timestep: TimeStep
    ):
        if self.batch is None:
            self.batch = timestep.build_buffer((self.trajectory_length, self.num_workers * self.num_envs_per_worker))


    def add(self):
        pass



if __name__ == '__main__':

    t = TimeStep(
                    obs={"screen":np.zeros((3, 3)), "t": 0},
                    agent_id=0,
                    episode_id=1,
                    episode_state=2,
                    t=0,
                    values=1.,
                    env_index=0,
                    reward=1,
                    done=0 == 19,
                    action=1,
                    prev_action=0-1,
                    action_logp=0.5,
                    action_logits=np.zeros(5),
                    prev_reward=0-1,
                    policy_id="bob",
                    version=0,
                    state= None,

    )


    # from polaris.experience.vectorised_episode import EpisodeState
    # class DummyPolicy:
    #     name = "name"
    #     version = 0
    #     policy_config = ConfigDict(
    #         {
    #             "discount": 0.99,
    #             "gae_lambda": 0.95
    #         }
    #     )
    #
    # policy = DummyPolicy
    #
    #
    # batch_size = 16 * 3
    #
    # config = ConfigDict({
    #     "trajectory_length": 16,
    # })
    #
    # data = TrajectoryBank(config)
    #
    # for episode in range(5):
    #     for i in range(20):
    #         if i == 0:
    #             episode_state = EpisodeState.RESET
    #         elif i == 19:
    #             episode_state = EpisodeState.TERMINATED
    #         else:
    #             episode_state = EpisodeState.RUNNING
    #
    #         t = TrajectoryStep(
    #             obs={"screen":np.zeros((3, 3)), "t": i},
    #             agent_id=0,
    #             episode_id=episode,
    #             episode_state=episode_state,
    #             t=i,
    #             values=1.,
    #             env_index=0,
    #             reward=1,
    #             done=i == 19,
    #             action=i,
    #             prev_action=i-1,
    #             action_logp=0.5,
    #             action_logits=np.zeros(5),
    #             prev_reward=i-1,
    #             policy_id=policy.name,
    #             version= policy.version,
    #             state= None,
    #         )
    #
    #         data.add([t])
    #
    #         if data.is_ready(batch_size, policy):
    #             b = data.get_batch(batch_size, policy)
    #             print(b[SampleBatch.DONE])
    #             print(b[SampleBatch.ADVANTAGES])
    #             exit()

