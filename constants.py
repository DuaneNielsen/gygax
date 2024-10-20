import jax.numpy as jnp
from enum import IntEnum


FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)
MAX_PARTY_SIZE = 4
N_PLAYERS = 2


class Actions(IntEnum):
    END_TURN = 0
    DASH = 1
    ATTACK_MELEE_WEAPON = 3
    ATTACK_RANGED_WEAPON = 4


class ActionResourceUsageType(IntEnum):
    END_TURN = 0
    ACTION = 1
    BONUS_ACTION = 2
    ATTACK = 3
    SPELL_SLOT_1 = 4


class ActionResourceType(IntEnum):
    ACTION = 0
    BONUS_ACTION = 1
    ATTACK = 2


class Abilities(IntEnum):
    STR = 0
    DEX = 1
    CON = 2
    INT = 3
    WIS = 4
    CHA = 5


class WeaponSlots(IntEnum):
    MELEE = 0
    RANGED = 1


N_ACTIONS = len(Actions)
N_ACTION_RESOURCE_TYPES = len(ActionResourceUsageType)
N_ABILITIES = len(Abilities)
N_WEAPON_SLOTS = len(WeaponSlots)

action_resource_table = {
    Actions.END_TURN: ActionResourceUsageType.END_TURN,
    Actions.DASH: ActionResourceUsageType.ACTION,
    Actions.ATTACK_MELEE_WEAPON: ActionResourceUsageType.ATTACK,
    Actions.ATTACK_RANGED_WEAPON: ActionResourceUsageType.ATTACK
}

ACTION_RESOURCE_TABLE = jnp.zeros((N_ACTIONS), dtype=jnp.bool_)
for action, action_resource in action_resource_table.items():
    ACTION_RESOURCE_TABLE.at[action].set(action_resource)
