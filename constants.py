import jax.numpy as jnp
from enum import StrEnum, Enum, auto, IntEnum


FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)

N_CHARACTERS = 4
N_PLAYERS = 2


class Actions(IntEnum):
    END_TURN = 0
    MOVE = auto()
    ATTACK_MELEE_WEAPON = auto()
    ATTACK_OFF_HAND_WEAPON = auto()
    ATTACK_RANGED_WEAPON = auto()


N_ACTIONS = len(Actions)


class ActionResourceType(IntEnum):
    ACTION = 0
    BONUS_ACTION = 1
    ATTACK = 2


N_ACTION_RESOURCE_TYPES=len(ActionResourceType)


class Abilities(IntEnum):
    STR = 0
    DEX = 1
    CON = 2
    INT = 3
    WIS = 4
    CHA = 5


N_ABILITIES = len(Abilities)

# action_resource_table = {
#     Actions.END_TURN: ActionResourceUsageType.END_TURN,
#     Actions.MOVE: ActionResourceUsageType.ACTION,
#     Actions.ATTACK_MELEE_WEAPON: ActionResourceUsageType.ATTACK,
#     Actions.ATTACK_RANGED_WEAPON: ActionResourceUsageType.ATTACK
# }

# ACTION_RESOURCE_TABLE = jnp.zeros((N_ACTIONS), dtype=jnp.bool_)
# for action, action_resource in action_resource_table.items():
#     ACTION_RESOURCE_TABLE.at[action].set(action_resource)


class ConfigItems(StrEnum):
    PARTY = auto()


class Party(IntEnum):
    PC = 0
    NPC = 1


class CharacterSheet(StrEnum):
    ABILITIES = auto()
    HITPOINTS = auto()
    ARMOR = auto()
    MAIN_HAND = auto()
    OFF_HAND = auto()
    RANGED_WEAPON = auto()


class DamageType(IntEnum):
    BLUDGEONING = 0
    SLASHING = 1
    PIERCING = 2
