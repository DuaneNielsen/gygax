import jax.numpy as jnp
from constants import N_PLAYERS, N_CHARACTERS, N_ACTIONS, Actions, TRUE, ActionResourceType
import chex
from dataclasses import field
import jax

legal_use_pos = [None] * N_ACTIONS
legal_use_pos[Actions.END_TURN] = [1, 1, 1, 1]
legal_use_pos[Actions.MOVE] = [1, 1, 1, 1]
legal_use_pos[Actions.ATTACK_MELEE_WEAPON] = [0, 0, 1, 1]
legal_use_pos[Actions.ATTACK_OFF_HAND_WEAPON] = [0, 0, 1, 1]
legal_use_pos[Actions.ATTACK_RANGED_WEAPON] = [1, 1, 0, 0]
legal_use_pos = jnp.bool(legal_use_pos)

legal_target_pos = [None] * N_ACTIONS
legal_target_pos[Actions.END_TURN] = [[1, 0, 0, 0], [0, 0, 0, 0]]
legal_target_pos[Actions.MOVE] = [[1, 1, 1, 1], [0, 0, 0, 0]]
legal_target_pos[Actions.ATTACK_MELEE_WEAPON] = [[0, 0, 0, 0], [0, 0, 1, 1]]
legal_target_pos[Actions.ATTACK_OFF_HAND_WEAPON] = [[0, 0, 0, 0], [0, 0, 1, 1]]
legal_target_pos[Actions.ATTACK_RANGED_WEAPON] = [[0, 0, 0, 0], [1, 1, 1, 1]]
legal_target_pos = jnp.bool(legal_target_pos)

legal_target_pos = jnp.bool(legal_target_pos)[None, None, ...]


@chex.dataclass
class ActionResources:
    action: jnp.int8 = 1
    bonus_action: jnp.int8 = 0
    attack: jnp.int8 = 0


@chex.dataclass
class ActionResourceArray:
    start_turn: ActionResources = field(default_factory=lambda: ActionResources())
    current: ActionResources = field(default_factory=lambda: ActionResources())


def legal_actions_by_action_resource(action_resources: ActionResources):
    """
    Does the character have the required action resources to perform the action

    Args:
        action_resources: (N_PLAYERS, N_CHARACTERS, N_ACTION_RESOURCES)

    Returns: (N_PLAYERS, N_CHARACTERS, N_ACTIONS, 1, 1)

    """
    legal_actions = jnp.zeros((N_PLAYERS, N_CHARACTERS, N_ACTIONS), dtype=jnp.bool)
    has_resource = jax.tree.map(lambda a: a > 0, action_resources)

    # end turn does not require resources
    legal_actions = legal_actions.at[:, :, Actions.END_TURN].set(TRUE)
    legal_actions = legal_actions.at[:, :, Actions.MOVE].set(TRUE)

    # weapons require an action or an attack resource
    can_attack = has_resource.action | has_resource.attack
    attack_actions = [Actions.ATTACK_MELEE_WEAPON, Actions.ATTACK_RANGED_WEAPON]
    legal_actions = legal_actions.at[:, :, attack_actions].set(can_attack[..., None])

    # offhand attacks require bonus action
    legal_actions = legal_actions.at[:, :, Actions.ATTACK_OFF_HAND_WEAPON].set(has_resource.bonus_action)

    return legal_actions[..., jnp.newaxis, jnp.newaxis]


def consume_action_resource(action_resources: ActionResources, action):
    has_resource = jax.tree.map(lambda a: a > 0, action_resources)

    # first use up attacks, else use up an action
    has_attacks = has_resource.attack
    weapon_attacked = (action.action == Actions.ATTACK_MELEE_WEAPON) | (action.action == Actions.ATTACK_RANGED_WEAPON)
    consume_action = action_resources.action.at[*action.source].set(action_resources.action[*action.source] - 1)
    consume_attack = action_resources.attack.at[*action.source].set(action_resources.attack[*action.source] - 1)
    action_resources.attack = jnp.where(weapon_attacked & has_attacks, consume_attack, action_resources.attack)
    action_resources.action = jnp.where(weapon_attacked & ~has_attacks, consume_action, action_resources.action)

    decrement_bonus_action = action_resources.bonus_action[*action.source] - 1
    consume_bonus_action = action_resources.bonus_action.at[*action.source].set(decrement_bonus_action)
    offhand_attacked = action.action == Actions.ATTACK_OFF_HAND_WEAPON
    action_resources.bonus_action = jnp.where(offhand_attacked, consume_bonus_action, action_resources.bonus_action)

    return action_resources


def legal_actions_by_player_position(pos):
    """
    True if character is in position to perform action
    :param pos: (Party, Characters) : index of player positions
    :return: boolean (Party, Characters, Action, 1, 1)
    """
    legal_pos_for_action = legal_use_pos[..., pos]
    return jnp.transpose(legal_pos_for_action, (1, 2, 0))[..., None, None]


