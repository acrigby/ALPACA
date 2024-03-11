import collections
from multiprocessing import Process
from multiprocessing.managers import BaseManager
import random


class DequeManager(BaseManager):
    pass

class DequeProxy(object):
    def __init__(self, capacity):
        self.deque = collections.deque(maxlen=capacity)
    def __len__(self):
        return self.deque.__len__()
    def appendleft(self, x):
        self.deque.appendleft(x)
    def append(self, x):
        self.deque.append(x)
    def pop(self):
        return self.deque.pop()
    def popleft(self):
        return self.deque.popleft()
    def sample(self,batch_size):
        batch_size = min(batch_size, len(self))
        return random.sample(self.deque,batch_size)

# Currently only exposes a subset of deque's methods.
DequeManager.register('DequeProxy', DequeProxy,
                      exposed=['__len__', 'append', 'appendleft',
                               'pop', 'popleft', 'sample'])


process_shared_deque = None  # Global only within each process.

def map_fn(q,i):
    print("here")
    global process_shared_deque
    process_shared_deque = q
    q.append("Hello world")
    process_shared_deque.append(i)  # deque's don't have a "put()" method.
    print(len(process_shared_deque))


if __name__ == "__main__":
    manager = DequeManager()
    manager.start()
    shared_deque = manager.DequeProxy(5)

    processes = []
    for episode in range(3):
        # Launch the first round of tasks, building a list of ApplyResult objects
        process = Process(target = map_fn,args = (shared_deque,episode))
        processes.append(process)
            
    for p in processes:
        p.start()

    for p in processes:
        p.join()


    for p in range(len(shared_deque)):  # Show left-to-right contents.
        print(shared_deque)

    print(shared_deque.sample(2))