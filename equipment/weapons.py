from dataclasses import dataclass
from enum import StrEnum, auto, IntEnum
from typing import List


class Type(IntEnum):
    MELEE = 0
    RANGED = 1


class Class(IntEnum):
    SIMPLE = 0
    MARTIAL = 1


class DamageType(IntEnum):
    BLUDGEONING = 0
    SLASHING = 1
    PIERCING = 2


class Props(StrEnum):
    LIGHT = auto()
    FINESSE = auto()
    THROWN = auto()
    TWOHANDED = auto()
    VERSATILE= auto()
    AMMUNITION = auto()
    LOADING = auto()
    HEAVY = auto()
    REACH = auto()


@dataclass
class Weapon:
    type: Type
    weapon_class: Class
    damage: str
    damage_type: DamageType
    properties: List[Props]
    range_short: int = 0
    range_long: int = 0


weapons = {
    'club': Weapon(Type.MELEE, Class.SIMPLE, '1d4', DamageType.BLUDGEONING, [Props.LIGHT]),
    'dagger': Weapon(Type.MELEE, Class.SIMPLE, '1d4', DamageType.PIERCING, [Props.FINESSE, Props.LIGHT, Props.THROWN], 20, 60),
    'greatclub': Weapon(Type.MELEE, Class.SIMPLE, '1d8', DamageType.BLUDGEONING, [Props.TWOHANDED]),
    'handaxe': Weapon(Type.MELEE, Class.SIMPLE, '1d6', DamageType.SLASHING, [Props.LIGHT, Props.THROWN], 20, 60),
    'javelin': Weapon(Type.MELEE, Class.SIMPLE, '1d6', DamageType.PIERCING, [Props.THROWN], 30, 120),
    'mace': Weapon(Type.MELEE, Class.SIMPLE, '1d6', DamageType.BLUDGEONING, []),
    'quarterstaff': Weapon(Type.MELEE, Class.SIMPLE, '1d6', DamageType.BLUDGEONING, [Props.VERSATILE]),
    'sickle': Weapon(Type.MELEE, Class.SIMPLE, '1d4', DamageType.SLASHING, [Props.LIGHT]),
    'spear': Weapon(Type.MELEE, Class.SIMPLE, '1d6', DamageType.PIERCING, [Props.VERSATILE, Props.THROWN], 20, 60),
    'crossbow-light': Weapon(Type.RANGED, Class.SIMPLE, '1d8', DamageType.PIERCING, [Props.AMMUNITION, Props.LOADING, Props.TWOHANDED], 80, 320),
    'dart': Weapon(Type.RANGED, Class.SIMPLE, '1d4', DamageType.PIERCING, [Props.FINESSE, Props.THROWN], 20, 60),
    'shortbow': Weapon(Type.RANGED, Class.SIMPLE, '1d6', DamageType.PIERCING, [Props.AMMUNITION, Props.TWOHANDED], 80, 320),
    'sling': Weapon(Type.RANGED, Class.SIMPLE, '1d4', DamageType.BLUDGEONING, [], 80, 320),
    'battleaxe': Weapon(Type.MELEE, Class.MARTIAL, '1d8', DamageType.SLASHING, [Props.VERSATILE]),
    'flail': Weapon(Type.MELEE, Class.MARTIAL, '1d8', DamageType.BLUDGEONING, []),
    'glaive': Weapon(Type.MELEE, Class.MARTIAL, '1d10', DamageType.SLASHING, [Props.HEAVY, Props.REACH, Props.TWOHANDED]),
    'greataxe': Weapon(Type.MELEE, Class.MARTIAL, '1d12', DamageType.SLASHING, [Props.HEAVY, Props.TWOHANDED]),
    'greatsword': Weapon(Type.MELEE, Class.MARTIAL, '2d6', DamageType.SLASHING, [Props.HEAVY, Props.TWOHANDED]),
    'lance': Weapon(Type.MELEE, Class.MARTIAL, '1d12', DamageType.PIERCING, [Props.REACH]),
    'longsword': Weapon(Type.MELEE, Class.MARTIAL, '1d8', DamageType.SLASHING, [Props.VERSATILE]),
    'maul': Weapon(Type.MELEE, Class.MARTIAL, '2d6', DamageType.BLUDGEONING, [Props.HEAVY, Props.TWOHANDED]),
    'morningstar': Weapon(Type.MELEE, Class.MARTIAL, '1d8', DamageType.PIERCING, []),
    'pike': Weapon(Type.MELEE, Class.MARTIAL, '1d10', DamageType.PIERCING, [Props.HEAVY, Props.REACH, Props.TWOHANDED]),
    'rapier': Weapon(Type.MELEE, Class.MARTIAL, '1d8', DamageType.PIERCING, [Props.FINESSE]),
    'scimitar': Weapon(Type.MELEE, Class.MARTIAL, '1d6', DamageType.SLASHING, [Props.FINESSE, Props.LIGHT]),
    'shortsword': Weapon(Type.MELEE, Class.MARTIAL, '1d6', DamageType.PIERCING, [Props.FINESSE, Props.LIGHT]),
    'trident': Weapon(Type.MELEE, Class.MARTIAL, '1d6', DamageType.PIERCING, [Props.VERSATILE, Props.THROWN], 20, 60),
    'war-pick': Weapon(Type.MELEE, Class.MARTIAL, '1d8', DamageType.PIERCING, []),
    'warhammer': Weapon(Type.MELEE, Class.MARTIAL, '1d8', DamageType.BLUDGEONING, [Props.VERSATILE]),
    'whip': Weapon(Type.MELEE, Class.MARTIAL, '1d4', DamageType.SLASHING, [Props.FINESSE, Props.REACH]),
    'blowgun': Weapon(Type.RANGED, Class.MARTIAL, '1', DamageType.PIERCING,[Props.AMMUNITION, Props.LOADING], 25, 100),
    'crossbow-heavy': Weapon(Type.RANGED, Class.MARTIAL, '1d10', DamageType.PIERCING, [Props.HEAVY, Props.AMMUNITION, Props.LOADING, Props.TWOHANDED], 100, 400),
    'crossbow-hand': Weapon(Type.RANGED, Class.MARTIAL, '1d6', DamageType.PIERCING, [Props.LIGHT, Props.AMMUNITION, Props.LOADING], 30, 120),
    'longbow': Weapon(Type.RANGED, Class.MARTIAL, '1d8', DamageType.PIERCING, [Props.HEAVY, Props.AMMUNITION, Props.TWOHANDED], 150, 600),
}