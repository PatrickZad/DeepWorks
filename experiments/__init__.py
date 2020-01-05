platform_linux = 0
platform_kaggle = 1
platform_win = 2
log_bases = ['/home/patrick/PatrickWorkspace/DeepWorks/log', '/kaggle/working', '']
out_bases = ['/home/patrick/PatrickWorkspace/DeepWorks/out', '/kaggle/working', '']


class ExperimentConfig:
    def __init__(self, platform):
        self.log_base = log_bases[platform]
        self.out_base = out_bases[platform]
        self.platform = platform
