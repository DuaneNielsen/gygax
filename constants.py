import jax.numpy as jnp
from enum import StrEnum, Enum, auto, IntEnum
from dnd_character.SRD import SRD, SRD_endpoints
from typing import List

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


def create_sorted_enum(enum_name: str, keys: List[str]) -> type[IntEnum]:
    """
    Creates an IntEnum with keys sorted alphabetically and mapped to sequential integers.

    Args:
        enum_name: Name for the enum class
        keys: List of strings to convert to enum names

    Returns:
        An IntEnum class where sorted keys map to sequential integers
    """
    # Sort keys alphabetically and create enum members dict
    # Convert keys to uppercase and replace spaces with underscores
    sorted_keys = sorted(keys)
    enum_members = {k.upper().replace(" ", "_"): i for i, k in enumerate(sorted_keys)}

    return IntEnum(enum_name, enum_members)


def list_srd_keys():
    return SRD_endpoints


def lookup_srd_key(endpoint):
    return {result["index"]: SRD(result['url']) for result in SRD(SRD_endpoints[endpoint])['results']}


DamageType = create_sorted_enum('DamageType', lookup_srd_key('damage-types').keys())
CharacterClass = create_sorted_enum('CharacterClass', lookup_srd_key('classes').keys())
Conditions = create_sorted_enum('Conditions', lookup_srd_key('conditions').keys())
Race = create_sorted_enum('Race', lookup_srd_key('races').keys())


class HitrollType(IntEnum):
    SPELL = 0
    MELEE = 1
    FINESSE = 2
    RANGED = 3
