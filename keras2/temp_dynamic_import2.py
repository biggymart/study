import importlib
from tensorflow.keras import applications as ap

tmp = dir(ap)[-14:]
tmp.pop(2)
MODULE_NAME = tmp # 모듈 13개
print(tmp)
from os import getcwd
print(ap.__file__)
print(getcwd())
# C:/Study
importlib.import_module(MODULE_NAME[0], package='../Users/ai/Anaconda3/lib/site-packages/tensorflow/keras/applications/')

import sys
from importlib.machinery import PathFinder

def find_and_load_module(complete_name):
    """
        Given a name and a path it will return a module instance
        if found.
        When the module could not be found it will raise ImportError
    """
    if complete_name in sys.modules:
        return sys.modules[complete_name]
    module = None
    module_spec = PathFinder.find_spec(complete_name)
    if module_spec:
        loader = module_spec.loader
        module = loader.load_module()
        sys.modules[complete_name] = module
    return module

    # https://stackoverflow.com/questions/25808364/can-not-call-dynamically-imported-python-modules
