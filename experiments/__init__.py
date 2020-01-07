platform_linux = 0
platform_kaggle = 1
platform_win = 2
platform_kaggle_test = 3
log_bases = ['/home/patrick/PatrickWorkspace/DeepWorks/log',
             '/kaggle/working', '', '/home/patrick/PatrickWorkspace/Datasets/kaggle_test/working']
out_bases = ['/home/patrick/PatrickWorkspace/DeepWorks/out',
             '/kaggle/working', '', '/home/patrick/PatrickWorkspace/Datasets/kaggle_test/working']


# kaggle test

class ExperimentConfig:
    def __init__(self, platform):
        self.log_base = log_bases[platform]
        self.out_base = out_bases[platform]
        self.platform = platform
