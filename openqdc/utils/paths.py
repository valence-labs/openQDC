import os

def get_local_cache():
    fname = os.path.abspath(__file__)
    base ='/'.join(fname.split('/')[:-3])
    cache_dir=os.path.join(base, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def get_remote_cache():
    fname = os.path.abspath(__file__)
    base ='/'.join(fname.split('/')[:-3])
    cache_dir=os.path.join(base, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir