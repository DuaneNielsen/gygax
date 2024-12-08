import pytest
import jax.numpy as jnp
import jax

import action_resources
from action_resources import legal_actions_by_player_position, ActionResourceArray
from constants import Actions, N_PLAYERS, N_CHARACTERS, N_ACTIONS, ActionResourceType, TargetParty
import dnd5e
from functools import partial

@pytest.fixture
def pos():
    return jnp.tile(jnp.arange(4), (2, 1))

def test_legal_actions_by_player_position(pos):
    legal_actions = legal_actions_by_player_position(pos)

    assert legal_actions.shape == (2, 4, len(Actions), 1, 1)

    assert legal_actions[0, 0, Actions.END_TURN] == True
    assert legal_actions[0, 0, Actions.MOVE] == True
    assert legal_actions[0, 0, Actions.ATTACK_MELEE_WEAPON] == False
    assert legal_actions[0, 0, Actions.ATTACK_OFF_HAND_WEAPON] == False
    assert legal_actions[0, 0, Actions.ATTACK_RANGED_WEAPON] == True


    assert legal_actions[1, 3, Actions.END_TURN] == True
    assert legal_actions[1, 3, Actions.MOVE] == True
    assert legal_actions[1, 3, Actions.ATTACK_MELEE_WEAPON] == True
    assert legal_actions[1, 3, Actions.ATTACK_OFF_HAND_WEAPON] == True
    assert legal_actions[1, 3, Actions.ATTACK_RANGED_WEAPON] == False


def test_legal_actions_by_target_position():
    legal_actions = jnp.ones((N_PLAYERS, N_CHARACTERS, N_ACTIONS, N_PLAYERS, N_CHARACTERS), dtype=jnp.bool)
    legal_actions = legal_actions & action_resources.legal_target_pos

    assert action_resources.legal_target_pos.shape == (1, 1, N_ACTIONS, N_PLAYERS, N_CHARACTERS)
    assert legal_actions[0, 0, Actions.END_TURN, 0, 0] == True
    assert legal_actions[0, 0, Actions.END_TURN, 1, 0] == False

    assert legal_actions[0, 2, Actions.ATTACK_MELEE_WEAPON, 1, 0] == False
    assert legal_actions[0, 2, Actions.ATTACK_MELEE_WEAPON, 1, 1] == False
    assert legal_actions[0, 2, Actions.ATTACK_MELEE_WEAPON, 1, 2] == True
    assert legal_actions[0, 2, Actions.ATTACK_MELEE_WEAPON, 1, 3] == True

    assert legal_actions[0, 2, Actions.ATTACK_RANGED_WEAPON, 0, 0] == False
    assert legal_actions[0, 2, Actions.ATTACK_RANGED_WEAPON, 1, 1] == True
    assert legal_actions[0, 2, Actions.ATTACK_RANGED_WEAPON, 1, 2] == True
    assert legal_actions[0, 2, Actions.ATTACK_RANGED_WEAPON, 1, 3] == True

    assert legal_actions[0, 0, Actions.ATTACK_MELEE_WEAPON, 0, 0] == False
    assert legal_actions[0, 0, Actions.ATTACK_MELEE_WEAPON, 1, 1] == False
    assert legal_actions[0, 0, Actions.ATTACK_MELEE_WEAPON, 1, 2] == True
    assert legal_actions[0, 0, Actions.ATTACK_MELEE_WEAPON, 1, 3] == True

    assert legal_actions[0, 0, Actions.ATTACK_RANGED_WEAPON, 0, 0] == False
    assert legal_actions[0, 0, Actions.ATTACK_RANGED_WEAPON, 1, 1] == True
    assert legal_actions[0, 0, Actions.ATTACK_RANGED_WEAPON, 1, 2] == True
    assert legal_actions[0, 0, Actions.ATTACK_RANGED_WEAPON, 1, 3] == True


@pytest.fixture
def resources():
    resources = ActionResourceArray()
    tile = partial(jnp.tile, reps=(N_PLAYERS, N_CHARACTERS))
    resources = jax.tree.map(tile, resources)
    return resources


def test_legal_actions_by_action_resources(resources):
    legal_actions = action_resources.legal_actions_by_action_resource(resources.current)
    assert legal_actions.shape == (N_PLAYERS, N_CHARACTERS, N_ACTIONS, 1, 1)

    assert jnp.all(legal_actions[:, :, Actions.END_TURN]) == True
    assert jnp.all(legal_actions[:, :, Actions.MOVE]) == True
    assert jnp.all(legal_actions[:, :, Actions.ATTACK_MELEE_WEAPON]) == True
    assert jnp.all(legal_actions[:, :, Actions.ATTACK_OFF_HAND_WEAPON]) == False
    assert jnp.all(legal_actions[:, :, Actions.ATTACK_RANGED_WEAPON]) == True

    resources.current.action = resources.current.action.at[0, 0].set(0)
    resources.current.bonus_action = resources.current.bonus_action.at[0, 0].set(1)
    legal_actions = action_resources.legal_actions_by_action_resource(resources.current)

    assert jnp.all(legal_actions[0, 0, Actions.END_TURN]) == True
    assert jnp.all(legal_actions[0, 0, Actions.MOVE]) == True
    assert jnp.all(legal_actions[0, 0, Actions.ATTACK_MELEE_WEAPON]) == False
    assert jnp.all(legal_actions[0, 0, Actions.ATTACK_OFF_HAND_WEAPON]) == True
    assert jnp.all(legal_actions[0, 1, Actions.ATTACK_OFF_HAND_WEAPON]) == False
    assert jnp.all(legal_actions[0, 0, Actions.ATTACK_RANGED_WEAPON]) == False


def test_consume_action_resources(resources, pos):
    encoded_action = dnd5e.encode_action(Actions.ATTACK_RANGED_WEAPON, 1, TargetParty.ENEMY, 3)
    action = dnd5e.decode_action(encoded_action ,0, pos)
    resources.current = action_resources.consume_action_resource(resources.current, action)
    assert resources.current.action[0, 0] == 1
    assert resources.current.action[0, 1] == 0

    resources.current.bonus_action = resources.current.bonus_action.at[0, [0, 1]].set(1)

    encoded_action = dnd5e.encode_action(Actions.ATTACK_OFF_HAND_WEAPON, 1, TargetParty.ENEMY, 3)
    action = dnd5e.decode_action(encoded_action, 0, pos)
    resources.current = action_resources.consume_action_resource(resources.current, action)

    assert resources.current.action[0, 0] == 1
    assert resources.current.action[0, 1] == 0
    assert resources.current.bonus_action[0, 0] == 1
    assert resources.current.bonus_action[0, 1] == 0
