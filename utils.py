import os
import sys
import time
import shutil
import subprocess
from collections import defaultdict
from deep_rl.utils.misc import get_time_str, mkdir

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.sum_meter = defaultdict(float)
        self.lasts = defaultdict(float)
        self.counts_meter = defaultdict(int)
    def update(self, key, val, n=1):
        self.lasts[key] = val
        self.sum_meter[key] += val
        self.counts_meter[key] += n
    def last(self, key):
        return self.lasts[key]
    def avg(self, key):
        return self.sum_meter[key] / self.counts_meter[key]
    def __repr__(self):
        s = ""
        for k in self.sum_meter:
            s += "{}={:.4f} ".format(k, self.avg(k))
        return s.strip()


class MultiTimer(object):
    """Count the time for each part of training."""
    def __init__(self):
        self.reset()
    def reset(self):
        self.timer_starts = defaultdict(float)
        self.timer_total = defaultdict(float)
    def start(self, key):
        if self.timer_starts[key] != 0:
            raise RuntimeError("start() is called more than once")
        self.timer_starts[key] = time.time()
    def stop(self, key):
        if key not in self.timer_starts:
            raise RuntimeError("Key does not exist; please call start() before stop()")
        self.timer_total[key] += time.time() - self.timer_starts[key]
        self.timer_starts[key] = 0
    def total(self, key):
        return self.timer_total[key]
    def __repr__(self):
        s = ""
        for k in self.timer_total:
            s += "{}_time={:.3f} ".format(k, self.timer_total[k])
        return s.strip()


def save_runtime(agent, models_path):
    runtime_dir = os.path.join(models_path, "runtime", agent.config.tag + "-" +  get_time_str())
    mkdir(runtime_dir)
    with open(os.path.join(runtime_dir, "cmd.txt"), 'w') as f:
        f.write(" ".join(sys.argv))
    config_file = ""
    for i in range(len(sys.argv)):
        if sys.argv[i] == '--config':
            config_file = sys.argv[i+1]
            break
    if config_file:
        shutil.copyfile(config_file, os.path.join(runtime_dir, "config.json"))
    else:
        agent.logger.warn('cannot find config file to save!')
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"])
    with open(os.path.join(runtime_dir, "commit"), 'wb') as f:
        f.write(commit)
    diff = subprocess.check_output(["git", "diff"])
    with open(os.path.join(runtime_dir, "dirty.diff"), 'wb') as f:
        f.write(diff)
