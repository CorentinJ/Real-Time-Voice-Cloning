from time import perf_counter as timer
from collections import OrderedDict
import numpy as np


class Profiler:
    def __init__(self, summarize_every=5, disabled=False):
        self.last_tick = timer()
        self.logs = OrderedDict()
        self.summarize_every = summarize_every
        self.disabled = disabled

    def tick(self, name):
        if self.disabled:
            return

        # Log the time needed to execute that function
        if not name in self.logs:
            self.logs[name] = []
        if len(self.logs[name]) >= self.summarize_every:
            self.summarize()
            self.purge_logs()
        self.logs[name].append(timer() - self.last_tick)

        self.reset_timer()

    def purge_logs(self):
        for name in self.logs:
            self.logs[name].clear()

    def reset_timer(self):
        self.last_tick = timer()

    def summarize(self):
        n = max(map(len, self.logs.values()))
        assert n == self.summarize_every
        print(f"Average execution time over {n} steps:")

        name_msgs = [f"{name} ({len(deltas)}/{n}):" for name, deltas in self.logs.items()]
        pad = max(map(len, name_msgs))
        for name_msg, deltas in zip(name_msgs, self.logs.values()):
            print(f"  {name_msg.ljust(pad)}  mean: {np.mean(deltas) * 1000}ms   std: {np.std(deltas) * 1000}ms")
        print("", flush=True)
