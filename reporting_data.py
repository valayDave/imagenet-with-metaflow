class HyperParams():
    def __init__(self):
        self.learning_rate = None 
        self.momentum = None
        self.weight_decay = None
        self.batch_size = None

class ModelAnalytics():
    def __init__(self):
        self.architecture = None # This will hold name of architecture. 
        self.epoch_histories = None # This will hold the data about the losses at each Epoch 
        self.hyper_params = HyperParams() #
        self.model = None 