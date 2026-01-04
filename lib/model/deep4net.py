import torch
from braindecode.models.deep4 import Deep4Net

class Deep4NetModel:
    def __init__(self, config):
        self.config = config
        self.model = self._build_model()
    
    def _build_model(self):
        if isinstance(self.config, dict):
            in_chans = self.config.get("in_chans", 8)
            n_classes = self.config.get("n_classes",2)
            n_times = self.config.get("n_times", 750)
            final_conv_length = self.config.get("final_conv_length", "auto")
        else:
            in_chans = self.config.in_chans
            n_classes = self.config.n_classes
            n_times = self.config.n_times
            final_conv_length = self.config.final_conv_length
        
        model = Deep4Net(
            in_chans,
            n_classes,
            n_times,
            final_conv_length=final_conv_length
        )
        return model
    
    def get_model(self):
        return self.model