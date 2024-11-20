import jax.numpy as jnp
from numpy import genfromtxt
import numpy as np

import constants
import dnd5e
import default_config
from constants import ConfigItems, Actions


policy = genfromtxt('policy.csv', delimiter=',', dtype=np.float32)
test_policy = genfromtxt('test_policy.csv', delimiter=',', dtype=np.float32)
test_current_player = genfromtxt('test_current_player.csv', delimiter=',', dtype=np.int32)
policy_raw, test_policy_raw = jnp.array(policy), jnp.array(test_policy)
policy = jnp.argmax(policy_raw, -1)
test_policy = jnp.argmax(test_policy_raw, -1)

print(policy)
print(test_policy)
options = jnp.sum(test_policy_raw > 0, -1)
correct = policy == test_policy
pos = jnp.stack([jnp.arange(3), jnp.arange(3)])


char_names = [list(default_config.default_config[ConfigItems.PARTY][party].keys()) for party in constants.Party]


# def action_to_text(action: dnd5e.ActionTuple, char_names):
#     pass
def name(char_names, character: dnd5e.Character, i):
    return char_names[character.party[i].item()][character.index[i].item()]

def action_repr(char_names, actions, i):
    source = name(char_names, actions.source, i)
    action = Actions(actions.action[i]).name
    target = name(char_names, actions.target, i)
    return f'{source} {action} {target}'


actions = dnd5e.decode_action(policy, test_current_player, pos)
test_action = dnd5e.decode_action(test_policy, test_current_player, pos)
for i in range(len(actions.action)):
    # if not correct[i]:
    policy_repr = action_repr(char_names, actions, i)
    test_policy_repr = action_repr(char_names, test_action, i)
    print(f'{i+1},{correct[i]} {test_current_player[i]} {test_policy[i]} {test_policy_repr} {policy[i]} {policy_repr}')

