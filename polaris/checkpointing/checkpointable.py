import pickle
import os

from ml_collections import ConfigDict

from polaris.utils import GlobalCounter
from polaris.utils.metrics import Metrics


def pickle_paths(obj, path: str):
    if isinstance(obj, dict) and not isinstance(obj, Metrics):
        for k, v in obj.items():
            pickle_paths(v, os.path.join(path, k))
    else:
        parent_path = path.rsplit(os.sep, maxsplit=1)[0]
        os.makedirs(parent_path, exist_ok=True)
        with open(path + ".pkl", "wb") as f:
            pickle.dump(obj, f)
            print("saved", path)

def unpickle_from_dir(path):
    """
    TODO: does not support dirs of dirs
    """
    unpickled = {}
    for full_path, _, files in  os.walk(path):
        sub_path = full_path[len(path) + 1:]
        if len(sub_path) > 0 and sub_path not  in unpickled:
            unpickled[sub_path] = {}
            to_fill = unpickled[sub_path]
        else:
            to_fill = unpickled
        for file in files:
            if file.endswith(".pkl"):
                with open(os.path.join(full_path, file), "rb") as f:
                    to_fill[file[:-4]] = pickle.load(f)

    return unpickled

def delete_empty_dirs(root):
   for dirpath, dirnames, filenames in os.walk(root, topdown=False):
      for dirname in dirnames:
         full_path = os.path.join(dirpath, dirname)
         if not os.listdir(full_path):
             os.rmdir(full_path)
   os.rmdir(root)


class Checkpointable:

    def __init__(self,
                 checkpoint_config: ConfigDict,
                 #checkpoint_path="polaris_checkpoint",
                 #stopping_condition = lambda metrics: metrics["step"] > 1e9,
                 components = {}
                ):

        self.checkpoint_frequency = checkpoint_config.checkpoint_frequency
        self.checkpoint_path = checkpoint_config.checkpoint_path
        self.stopping_condition = checkpoint_config.stopping_condition
        self.keep = checkpoint_config.keep
        self.last_checkpoint = -1

        self.prev_checkpoints = []

        self.components = components


    def is_done(self, metrics):
        done = False
        for m, v in  self.stopping_condition.items():
            done = metrics.get(done, 0) >= v
            if done:
                return done
        return done

    def checkpoint_if_needed(self):
        c = GlobalCounter[GlobalCounter.STEP]
        if self.last_checkpoint != c and c % self.checkpoint_frequency == 0:
            self.last_checkpoint = c
            self.save()


    def roll_checkpoints(self, new_path):

        self.prev_checkpoints.append(new_path)
        if len(self.prev_checkpoints) > self.keep:
            to_remove = self.prev_checkpoints.pop(0)
            try:
                for full_path, _, files in os.walk(to_remove):
                    for file in files:
                        os.remove(os.path.join(full_path, file))
                delete_empty_dirs(to_remove)
            except Exception as e:
                print(f"Tried to remove ckpt {e}. Got error:", e)
        os.makedirs(new_path, exist_ok=True)

    def save(self):
        print(self.components)
        curr_path = os.path.join(self.checkpoint_path, "checkpoint_" + str(GlobalCounter[GlobalCounter.STEP]))
        self.roll_checkpoints(curr_path)
        pickle_paths(self.components, curr_path)

    def restore(self, restore_path=None):
        if restore_path is None:
            restore_path = self.checkpoint_path

        _, checkpoints, _ = next(os.walk(restore_path))

        def get_ckpt_num(ckpt):
            return int(ckpt.split("_")[-1])

        self.prev_checkpoints = [
            os.path.join(self.checkpoint_path, ckpt) for ckpt in sorted(checkpoints, key=get_ckpt_num)
        ]


        last_checkpoint = self.prev_checkpoints[-1]

        restored_components = unpickle_from_dir(last_checkpoint)

        print(last_checkpoint, restored_components)
        loaded_keys = set(restored_components.keys())
        component_keys = set(self.components.keys())

        if loaded_keys != component_keys:
            print(f"Loaded a checkpoint with different components: had {component_keys}, loaded {loaded_keys}")

        self.__dict__.update(restored_components)

        self.components = {
            k: getattr(self, k)
            for k in self.components
        }

        GlobalCounter[GlobalCounter.STEP] = get_ckpt_num(last_checkpoint)

        print('Restored from checkpoint:', restore_path)







