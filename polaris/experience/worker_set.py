from typing import List, Tuple, Union

from ml_collections import ConfigDict
import ray

from .environment_worker import EnvWorker
from .episode import EpisodeMetrics
from .sampling import SampleBatch


class WorkerSet:
    def __init__(
            self,
            config: ConfigDict,
    ):
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
        for wid, (job, episode_options) in zip(self.available_workers, jobs):
            hired_workers.add(wid)
            job_refs.append(self.workers[wid].run_episode_for.remote(
                job,
                episode_options
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




