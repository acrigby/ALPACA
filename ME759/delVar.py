for name in dir():
    if not name.startswith('_'):
        del globals()[name]

del replay_mem
del policy_net
del target_net
del shared_deque