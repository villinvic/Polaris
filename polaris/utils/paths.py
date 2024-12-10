import os

# TODO: Deprecated

class PathManager:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        #self.policies_dir = os.path.join(self.base_dir, 'policies')
        #self.models_dir = os.path.join(self.base_dir, 'models')
        self.tensorboard_logs_dir = os.path.join(self.base_dir, 'tensorboard_logs')

        self._create_dirs()

    def _create_dirs(self):
        """Create the necessary directories if they do not exist."""
        #os.makedirs(self.policies_dir, exist_ok=True)
        #os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.tensorboard_logs_dir, exist_ok=True)

    # def get_policies_path(self, filename: str = "") -> str:
    #     """Get the path for a file in the policies directory."""
    #     return os.path.join(self.policies_dir, filename)
    #
    # def get_models_path(self, filename: str = "") -> str:
    #     """Get the path for a file in the models directory."""
    #     return os.path.join(self.models_dir, filename)

    def get_tensorboard_logdir(self, dir_name: str = "") -> str:
        """Get the path for a dir in the tensorboard logs directory."""
        path = os.path.join(self.tensorboard_logs_dir, dir_name)
        os.makedirs(path, exist_ok=True)
        return path