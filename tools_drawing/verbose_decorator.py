import functools
import os

verbose_functions = set()

def verbose(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        # Do something before
        name = func.__module__ + "." +func.__name__
        if name in verbose_functions:
            #print(name, "in verbose mode")
            value = func(*args, **kwargs, VERBOSE=True)
        else:
            value = func(*args, **kwargs)
        # Do something after
        return value
    return wrapper_decorator

def read_verbose_functions(file_name="verbose_functions.txt"):
    with open(file_name, "r") as fh:
        lines = fh.readlines()
    for line in lines:
        if "." in line:
            if "\n" in line:
                line = line.split("\n")[0]
            verbose_functions.add("pysbm."+line)
    #print(verbose_functions)