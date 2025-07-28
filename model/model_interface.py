import importlib
from torch import nn

class ModelInterface:
    def __init__(self, **kwargs):
        self.model : nn.Module = None
        modelname = kwargs.get('model')
        self.load_model(modelname=modelname, **kwargs)
    
    def load_model(self, modelname = None, **kwargs):
        if modelname is None:
            modelname = kwargs.get('model')
        camel_name = ''.join([word.capitalize() for word in modelname.split('_')])
        try:
            Model = getattr(importlib.import_module(f"model.{modelname}", package=__package__), camel_name)
        except:
            raise ValueError(f"Model {modelname} not found")
        self.model = Model(**kwargs)
        
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
        
    