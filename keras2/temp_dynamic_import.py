# 방법 1

import importlib

def load_module_func(module_name):
    mod = importlib.import_module(module_name)
    return mod

# 방법 2

def laod_module_func2(module_name):
    mod = __import__('%s' %(module_name), fromlist=[module_name])
    return mod

'''
def load_module_func(module_name, class_name):
    mod = __import__('%s' %(module_name), fromlist=[module_name])
    cls = getattr(mod, class_name)(class_args)

    cls.func1()
    cls.func2(arg1, arg2)
    ...
'''


# 출처: https://bluese05.tistory.com/31 [ㅍㅍㅋㄷ]


#########################################################################
# How to iteratively import python modules
# https://www.tutorialspoint.com/Can-we-iteratively-import-python-modules-inside-a-for-loop
# import importlib
# modnames = ["Xception", "VGG16", "VGG19", "ResNet50"]
# for lib in modnames:
#     globals()[lib] = importlib.import_module(lib)

#######################################################################
# 한 프로젝트의 서비스 내부에 여러 개의 모듈이 존재할 때 가내수공업을 피하는 방법
# https://www.jungyin.com/30

# flask_test_project 안의 test 폴더 안에 module이 여러 개 존재하고,
# 각 모듈별로 메소드들이 있을 때, 동적 임포팅 하는 방법

# 동적인 모듈을 생성해준다
# def make_modules(module_name):
#   command_module = __import__("프로젝트이름.모듈폴더.%s" %module_name, fromlist=["프로젝트이름.모듈폴더.%s"])
#   모듈을 리턴해준다
#   return command_module

# def get_modules(modul_name):
#   모듈을 리턴 받는다
#   command_module = make_modules(module_name)

#   동적으로 생성된 모듈에 접근
#   load_func = getattr(command_module, 클래스나 메소드 이름)

#   리턴값 = load_func(파라미터)

#   return 최종리턴값


# How to iteratively import python modules
# https://www.tutorialspoint.com/Can-we-iteratively-import-python-modules-inside-a-for-loop
# import importlib
# modnames = ['densenet', 'efficientnet', 'imagenet_utils', 'inception_resnet_v2', 'inception_v3', 'mobilenet', 'mobilenet_v2','nasnet', 'resnet', 'resnet50', 'resnet_v2', 'vgg16', 'vgg19', 'xception']
# for lib in modnames:
#     globals()[lib] = importlib.import_module(lib, package='tensorflow.keras.applications')


# 비슷하게 설명해주는 블로그
### https://codechacha.com/ko/dynamic-import-and-call/


# 핵심은 __import__ 이후에 getattr 하는 것 같은데

##################################################

# 내가 하고 싶은 것:
# Input: modnames = ["Xception", "VGG16", "VGG19", "ResNet50"]

# user configurable

# one.py
"""
Plugin #1
"""


class Plugin:
    def __init__(self, *args, **kwargs):
        print('Plugin init one :', args, kwargs)
    
    def execute(self, a, b):
        return a + b

# =======
# two.py
"""
Plugin #1
"""


class Plugin:
    def __init__(self, *args, **kwargs):
        print('Plugin init two :', args, kwargs)
    
    def execute(self, a, b):
        return a - b

# =====
# app.py

import importlib

PLUGIN_NAME = "plugins.one"

plugin_module = importlib.import_module(PLUGIN_NAME, ".")

print(plugin_module)

plugin = plugin_module.Plugin("hello", key=123)

result = plugin.execute(5, 3)
print(result)



