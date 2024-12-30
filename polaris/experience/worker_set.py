from typing import List, Tuple, Union, Dict

from ml_collections import ConfigDict
import ray
from polaris.policies import PolicyParams

from .environment_worker import EnvWorker, SyncEnvWorker
from .episode import EpisodeMetrics
from .sampling import SampleBatch


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
        print(len(jobs), len(hired_workers))
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



