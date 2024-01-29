from functools import wraps

def make_cacheable(func):
    """
    :param func: function to cache

    :return: wrapper function that caches the result of func and only actually calls the function when the result is not in the cache

    Note that this function assumes the cache is the input state
    object which should be indexable.
    """
    @wraps(func)
    def wrapper(state):
        cache_key = func.__name__
        if cache_key not in state:
            state[cache_key] = func(state)
        return state[cache_key]
    return wrapper
