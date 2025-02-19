import itertools
from typing import List, Tuple, Union, Dict

import tree
from ml_collections import ConfigDict
import ray
from polaris.policies import PolicyParams

from .environment_worker import EnvWorker, SyncEnvWorker, VectorisedEnvWorker
from .episode import EpisodeMetrics
from .sampling import SampleBatch
from .timestep import TimeStep


class AsyncWorkerSet:
    def __init__(
            self,
            config: ConfigDict,
    ):
        """
        TODO: out-of-date
        Workers do not update model weights mid episode.
        """

        self.workers = {
            wid: EnvWorker.remote(
                worker_id=wid,
                config=config
            )
            for wid in range(config.num_workers)
        }

        self.available_workers = set(list(self.workers.keys()))

    def get_num_worker_available(self):
        return len(self.available_workers)

    def push_jobs(
            self,
            jobs: List,
    ):
        job_refs = []
        hired_workers = set()
        for wid, job in zip(self.available_workers, jobs):
            hired_workers.add(wid)
            job_refs.append(self.workers[wid].run_episode_for.remote(
                job
            ))
        self.available_workers -= hired_workers

        return job_refs

    def wait(self, job_refs) -> List[Union[SampleBatch,EpisodeMetrics]]:
        samples = []
        ready_jobs, _ = ray.wait(
            job_refs,
            num_returns=len(self.workers),
            timeout=0.01,
            fetch_local=False,
        )
        for job in ready_jobs:
            try:
                wid, ret = ray.get(next(job))
                if ret is None or isinstance(ret, EpisodeMetrics):
                    self.available_workers.add(wid)
                    ret = [ret]
                samples.extend(ret)
            except StopIteration:
                job_refs.remove(job)
        return samples



class SyncWorkerSet:
    def __init__(
            self,
            config: ConfigDict,
            with_spectator=False,
    ):
        """
        Synchronous set of environment workers.
        The set of workers update their model weights each time they communicate with the main process.
        """

        self.workers = {
            wid: SyncEnvWorker.remote(
                worker_id=wid,
                config=config,
                spectator=with_spectator and wid == 0
            )
            for wid in range(config.num_workers)
        }

        self.databag = []
        self.waiting = False
        self.num_workers = len(self.workers)
        self.batch_count = 0

        self.available_workers = set(list(self.workers.keys()))
        self.waiting_workers = set(list(self.workers.keys()))

        self.running_jobs = {}

    def get_num_worker_available(self) -> int:
        return len(self.available_workers)

    def push_jobs(
            self,
            params_map: Dict[str, PolicyParams],
            jobs: List[Dict[str, PolicyParams]],
            push_back=False
    ) -> List:
        """
        Push jobs to available environment workers.
        """

        job_refs = []

        if not push_back:
            if self.waiting:
                return job_refs

            for wid, job in self.running_jobs.items():
                if wid in self.waiting_workers:
                    for aid, params in job.items():
                        if params.name in params_map:
                            job[aid] = params_map[params.name]
                    self.waiting_workers -= {wid}

                    job_refs.append(self.workers[wid].get_next_batch_for.remote(
                        job
                    ))

        hired_workers = set()
        for wid, job in zip(self.available_workers, jobs):
            hired_workers.add(wid)
            self.running_jobs[wid] = job
            job_refs.append(self.workers[wid].get_next_batch_for.remote(
                job
            ))
        self.available_workers -= hired_workers
        self.waiting_workers -= hired_workers

        return job_refs

    def wait(
            self,
            params_map,
            job_refs,
            timeout=1e-2
    ) -> List[Union[SampleBatch,EpisodeMetrics]]:
        """
        Blocks timeout seconds before receiving samples from workers.
        """

        returns = []
        ready_jobs, njob_refs = ray.wait(
            job_refs,
            num_returns=len(job_refs),
            timeout=timeout,
            fetch_local=True,
        )

        job_refs = njob_refs
        for job in ready_jobs:
            wid, rets = ray.get(job)
        
            for ret in rets:
                if isinstance(ret, EpisodeMetrics):
                    # Episode is finished if we are here
                    self.available_workers.add(wid)
                    self.waiting_workers.add(wid)
                    try:
                        del self.running_jobs[wid]
                    except Exception as e:
                        print("WAT?")
                        raise

                elif ret is None:
                    self.available_workers.add(wid)
                    self.waiting_workers.add(wid)

                    job = self.running_jobs.pop(wid)

                    # push back the job that failed, this should be no problem to send to a different worker
                    job_refs.extend(self.push_jobs(params_map, [job], push_back=True))
                else:
                    self.waiting_workers.add(wid)
                    #self.batch_count += 1
                if ret is not None:
                    returns.append(ret)

        return returns, job_refs


def map_slice(batched, slice):
    return tree.map_structure(lambda v: v[slice], batched)



class VectorisedWorkerSet:
    def __init__(
            self,
            config: ConfigDict,
    ):
        """
        Vectorised set of environment workers.
        We assume the main GPU process does all computations related to models
        """
        self.num_workers = config.num_workers
        self.inference_batch_size = config.inference_batch_size
        self.num_envs_per_worker = config.num_envs_per_worker


        self.workers = {
            wid: VectorisedEnvWorker.remote(
                worker_id=wid,
                config=config,
            )
            for wid in range(config.num_workers)
        }
        # if (config.num_workers * config.num_envs_per_worker) % self.inference_batch_size != 0:
        #     raise ValueError(f"inference_batch_size must divide (num_workers * num_envs_per_worker) !"
        #                      f" {config.num_workers * config.num_envs_per_worker} % {self.inference_batch_size} != 0.")
        if self.inference_batch_size % config.num_envs_per_worker != 0:
            raise ValueError(f"num_envs_per_worker must divide inference_batch_size !"
                             f" {self.inference_batch_size} % {self.num_envs_per_worker} != 0.")

        self.ongoing_job_refs = []
        self.available_workers = set(self.workers.keys())

    def num_envs(self):
        return self.num_envs_per_worker * self.num_workers

    def requires_options(self):
        return [wid in self.available_workers for wid in self.workers]


    def send(
            self,
            actions,
            options
    ):
        self.ongoing_job_refs.extend([
            self.workers[wid].step.remote(
                map_slice(actions, slice(i * self.num_envs_per_worker, (i + 1) * self.num_envs_per_worker, 1)),
                options,
                # options[wid * self.num_envs_per_worker: (wid + 1) * self.num_envs_per_worker]
            )
            for i, wid in enumerate(self.available_workers)])

        self.available_workers = set()

    def recv(self) -> Tuple[List[TimeStep], List[EpisodeMetrics]]:
        ready_jobs, self.ongoing_job_refs = ray.wait(self.ongoing_job_refs,
                                                     num_returns=self.inference_batch_size // self.num_envs_per_worker,
                                                     timeout=None)

        data = ray.get(ready_jobs)

        samples = []
        metrics = []

        for s, m, w in data:
            self.available_workers.add(w)

            samples.extend(s)
            metrics.extend(m)

        return samples, metrics


    def step(
            self,
            actions,
            options
    ) -> Tuple[List[TimeStep], List[EpisodeMetrics]]:

        self.send(actions, options)
        return self.recv()



