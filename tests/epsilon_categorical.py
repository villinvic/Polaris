from polaris.models.utils import EpsilonCategorical, CategoricalDistribution
import numpy as np

logits = np.float32(np.random.randint(-100, 5, (1, 4)))
print(logits)
eps_dist = EpsilonCategorical(logits)
orig_dist = CategoricalDistribution(logits)

eps_dist._compute_dist()
orig_dist._compute_dist()

print(eps_dist.dist.probs_parameter())
print(orig_dist.dist.probs_parameter())

