from collections import defaultdict
import tree
import numpy as np

def merge(args):
    if any(not isinstance(i, dict) for i in args):
        return np.array([args[0]]) if len(args) == 1 else np.array(args)
    d = defaultdict(list)
    for i in args:
        for a, b in i.items():
            d[a].append(b)
    return {a: merge(b) for a, b in d.items()}

def average_dict(data: list):
    return tree.map_structure(
        lambda v: np.mean(v),
        merge(data)
    )

d1 = {"pi1": {"r": 2, "done": 5, "action": 0}, "other": 9}
d2 = {"pi1": {"r": 2, "done": 1, "action": 2}, "other": 0}
d3 = {"pi2": {"done": 0, "action": 3}}
d4 = {"pi2": {"r": 3.5, "action": 1}, "other": 0}



print(average_dict([d4, d3, d1, d2]))