import abc

# constant
batch_norm = 0
inst_norm = 1


class NNconfig:
    def __init__(self):
        self.lr = None

    @abc.abstractclassmethod
    def optimizer(self, parameters):
        pass

    @abc.abstractclassmethod
    def main_loss(self):
        pass
