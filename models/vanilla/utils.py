import joblib
from pdb import set_trace as breakpoint
import os, sys, logging
import functools
import time


class ErrorFlag:
    """Default class for error catching"""
    _number_of_errors = 0
    
    def __init__(self):
        self.increase_errors_by_one()
        
    def __repr__(self):
        return 'ErrorFlag'
    
    def __str__(self):
        return self.__repr__()
    
    @classmethod
    def get_errors(cls):
        return cls._number_of_errors
    
    @classmethod
    def increase_errors_by_one(cls):
        cls._number_of_errors += 1
        
    @classmethod
    def reset_errors(cls):
        cls._number_of_errors = 0

def get_error_logger():
    """Log errors in same logger"""
    logger = logging.getLogger('Error Logger')
    logger.setLevel(logging.ERROR)
    return logger

def protect(func):
    """Protect function from errors and exceptions"""
    logger = get_error_logger()
    @functools.wraps(func)
    def protect_wrap(*args, **kwargs):
        try:
            value = func(*args, **kwargs)
            if not isinstance(value, ErrorFlag):
                return value
            else:
                error_msg = "WARNING: '{}' returned with ErrorFlag.".format(func.__name__)
                logger.error(error_msg)
                return value
        except Exception as e:
            args_repr = [repr(a) for a in args] #TODO: functions should habe their name, not repr
            kwargs_repr = ["{}={}".format(k,v) for k, v in kwargs.items()]
            signature = ", ".join(args_repr + kwargs_repr)
            error_msg = "ERROR in '{0}'. Method was called as '{0}({1})'.\n>> ERROR: {2}".format(func.__name__, signature, repr(e))
            logger.error(error_msg)
            return ErrorFlag() 
    return protect_wrap

def timer(func):
    """Measure time it takes to complete function"""
    logger = get_error_logger()
    @functools.wraps(func)
    def time_wrap(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        #logger.info("Finished {} in {:.4f} secs".format(func.__name__, run_time))
        print("Finished {} in {:.4f} secs".format(func.__name__, run_time))
        return value
    return time_wrap

class Environment:
    def __init__(self, schedule):
        self._internal_state = None
        self._initialized = False # This is to avoid adding None the first time we enter the state.setter
        self._past_states = []
        self.schedule = schedule

    @property
    def state(self):
        return self._internal_state

    @state.setter
    def state(self, newState):
        if self._initialized:
            self._past_states.append(self._internal_state)
        else:
            self._initialized = True
        self._internal_state = newState

    @state.deleter
    def state(self):
        if len(self._past_states) == 0:
            self._internal_state = None
            self._initialized = False
        else:
            self._internal_state = self._past_states[-1]
            self._past_states = self._past_states[:-1]

    def _run_job(self, job):
        """Runs a single job
           A job is of the form :(func, args, kwargs)"""
        func, args, kwards = job
        values = func(*args, **kwards)
        return values

    @protect
    def run_job(self, job):
        return self._run_job(job)

    def run(self):
        """Run all of the jobs in the schedule
           Runs sequentially"""
        for job in self.schedule:
            value = self.run_job(job)
            newState = {'job':job, 'value':value}
            if not isinstance(value, ErrorFlag):
                self.state = newState

    def save_states(self, filename):
        joblib.dump(self._past_states+[self._internal_state], filename)