import numpy as np


b = 32
a = 51

advantages = np.random.normal(0, 1, (b))
action_probs = np.random.uniform(0.1, 0.9, (b, a))

action_probs = action_probs / action_probs.sum(axis=-1, keepdims=True)

logits = np.log(action_probs)[np.arange(b), np.random.randint(0, a, b)]
print("entropy", np.mean(np.sum(-action_probs * np.log(action_probs), axis=-1)))

for exptemp in np.logspace(1e-2, 5, num=20):
    temp = np.log(exptemp)

    print("max", np.max(advantages / temp))
    exp_adv = np.exp(advantages / temp)
    softmax_adv = exp_adv / np.sum(exp_adv)

    print("soft", softmax_adv)

    pi_loss = -np.sum(logits * softmax_adv)


    print("loss, temp", pi_loss, temp)
