from typing import List

from ml_collections import ConfigDict
import ray

from .environment_worker import EnvWorker

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

    def wait(self, job_refs):
        samples = []
        ready_jobs, _ = ray.wait(
            job_refs,
            num_returns=1,
            timeout=None,
            fetch_local=False,
        )
        for job in ready_jobs:
            wid, sample_batch = ray.get(next(job))
            if sample_batch is None:
                # TODO episode metrics here
                job_refs.remove(job)
                self.available_workers.add(wid)
            else:
                samples.append(sample_batch)
        return samples




