import time

import numpy as np
import psutil
import ray
from gymnasium.envs.classic_control import CartPoleEnv


def print_avg_core_usage(interval=1.0, duration=10.0):
    """
    Prints the average CPU usage per core over a specified duration.

    :param interval: Time in seconds to wait between each measurement.
    :param duration: Total time in seconds for which to compute the average usage.
    """
    core_count = psutil.cpu_count(logical=True)
    usage_per_core = [0] * core_count
    samples = int(duration / interval)

    print("Monitoring CPU usage... (press Ctrl+C to stop)\n")
    try:
        for _ in range(samples):
            core_usages = psutil.cpu_percent(percpu=True, interval=interval)
            for i, usage in enumerate(core_usages):
                usage_per_core[i] += usage
        # Calculate average usage per core
        avg_usage = [usage / samples for usage in usage_per_core]

        # Print results
        print("\nAverage CPU usage per core:")
        for i, usage in enumerate(avg_usage):
            print(f"Core {i}: {usage:.2f}%")
        print(f"\nOverall CPU usage: {sum(avg_usage) / core_count:.2f}%")
    except KeyboardInterrupt:
        print("\nMonitoring interrupted by user.")

@ray.remote(num_cpus=1, num_gpus=0)
def play_env(i):

    env = CartPoleEnv()
    while True:
        _ = np.random.rand(512, 512).dot(np.random.rand(512, 512))  # Dummy workload
        # _, _, d, _, _= env.step(env.action_space.sample())
        # if d:
        #     env.reset()





if __name__ == '__main__':


    runs = [play_env.remote(i) for i in range(64)]


    time.sleep(20)
    print_avg_core_usage()


    for r in runs:
        ray.cancel(r)

    exit()
