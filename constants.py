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


N_ACTION_RESOURCE_TYPES = len(ActionResourceType)


class Abilities(IntEnum):
    STR = 0
    DEX = 1
    CON = 2
    INT = 3
    WIS = 4
    CHA = 5


N_ABILITIES = len(Abilities)


class Conditions(IntEnum):
    DEAD = 0
    BLINDED = auto()
    CHARMED = auto()
    EXHAUSTED = auto()
    FRIGHTENED = auto()
    GRAPPLED = auto()
    INCAPACITATED = auto()
    INVISIBLE = auto()
    PARALYSED = auto()
    PETRIFIED = auto()
    POISONED = auto()
    PRONE = auto()
    RESTRAINED = auto()
    STUNNED = auto()
    UNCONSCIOUS = auto()


N_CONDITIONS = len(Conditions)


class ConfigItems(StrEnum):
    PARTY = auto()


class Party(IntEnum):
    PC = 0
    NPC = 1


class TargetParty(IntEnum):
    FRIENDLY = 0
    ENEMY = 1


class CharacterSheet(StrEnum):
    ABILITIES = auto()
    HITPOINTS = auto()
    ARMOR = auto()
    MAIN_HAND = auto()
    OFF_HAND = auto()
    RANGED_WEAPON = auto()


class DamageType(IntEnum):
    ACID = 0
    BLUDGEONING = auto()
    COLD = auto()
    FIRE = auto()
    FORCE = auto()
    LIGHTNING = auto()
    NECROTIC = auto()
    PIERCING = auto()
    POISON = auto()
    RADIANT = auto()
    SLASHING = auto()
    THUNDER = auto()


N_DAMAGE_TYPES = len(DamageType)

# scaling constants for observations
HP_LOWER = -20  # we need this to handle death saves
HP_UPPER = 20
AC_LOWER = 0
AC_UPPER = 20
PROF_BONUS_LOWER = 0
PROF_BONUS_UPPER = 6
ABILITY_MODIFIER_LOWER=-5
ABILITY_MODIFIER_UPPER=10
CONDITION_STACKS_UPPER=5
ACTION_RESOURCES_UPPER=5


# actions
DAMAGE_UPPER = 20