# -*- coding: UTF-8 -*-
import functools
from time import time


def log_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time()
        func_return = func(*args, **kwargs)
        print("%s execute time: %ds" % (func.__name__, time()-start))
        return func_return
    return wrapper
