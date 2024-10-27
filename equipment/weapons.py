from dataclasses import dataclass
from enum import StrEnum, auto, IntEnum

from constants import DamageType
from equipment.equipment import EquipmentType, Equipment
from typing import List


class Class(IntEnum):
    SIMPLE = 0
    MARTIAL = 1


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
class Weapon(Equipment):
    type: EquipmentType
    weapon_class: Class
    damage: str
    damage_type: DamageType
    properties: List[Props]
    range_short: int = 0
    range_long: int = 0


weapons = {
    'club': Weapon(EquipmentType.WEAPON_MELEE, Class.SIMPLE, '1d4', DamageType.BLUDGEONING, [Props.LIGHT]),
    'dagger': Weapon(EquipmentType.WEAPON_MELEE, Class.SIMPLE, '1d4', DamageType.PIERCING, [Props.FINESSE, Props.LIGHT, Props.THROWN], 20, 60),
    'greatclub': Weapon(EquipmentType.WEAPON_MELEE, Class.SIMPLE, '1d8', DamageType.BLUDGEONING, [Props.TWOHANDED]),
    'handaxe': Weapon(EquipmentType.WEAPON_MELEE, Class.SIMPLE, '1d6', DamageType.SLASHING, [Props.LIGHT, Props.THROWN], 20, 60),
    'javelin': Weapon(EquipmentType.WEAPON_MELEE, Class.SIMPLE, '1d6', DamageType.PIERCING, [Props.THROWN], 30, 120),
    'mace': Weapon(EquipmentType.WEAPON_MELEE, Class.SIMPLE, '1d6', DamageType.BLUDGEONING, []),
    'quarterstaff': Weapon(EquipmentType.WEAPON_MELEE, Class.SIMPLE, '1d6', DamageType.BLUDGEONING, [Props.VERSATILE]),
    'sickle': Weapon(EquipmentType.WEAPON_MELEE, Class.SIMPLE, '1d4', DamageType.SLASHING, [Props.LIGHT]),
    'spear': Weapon(EquipmentType.WEAPON_MELEE, Class.SIMPLE, '1d6', DamageType.PIERCING, [Props.VERSATILE, Props.THROWN], 20, 60),
    'crossbow-light': Weapon(EquipmentType.WEAPON_RANGED, Class.SIMPLE, '1d8', DamageType.PIERCING, [Props.AMMUNITION, Props.LOADING, Props.TWOHANDED], 80, 320),
    'dart': Weapon(EquipmentType.WEAPON_RANGED, Class.SIMPLE, '1d4', DamageType.PIERCING, [Props.FINESSE, Props.THROWN], 20, 60),
    'shortbow': Weapon(EquipmentType.WEAPON_RANGED, Class.SIMPLE, '1d6', DamageType.PIERCING, [Props.AMMUNITION, Props.TWOHANDED], 80, 320),
    'sling': Weapon(EquipmentType.WEAPON_RANGED, Class.SIMPLE, '1d4', DamageType.BLUDGEONING, [], 80, 320),
    'battleaxe': Weapon(EquipmentType.WEAPON_MELEE, Class.MARTIAL, '1d8', DamageType.SLASHING, [Props.VERSATILE]),
    'flail': Weapon(EquipmentType.WEAPON_MELEE, Class.MARTIAL, '1d8', DamageType.BLUDGEONING, []),
    'glaive': Weapon(EquipmentType.WEAPON_MELEE, Class.MARTIAL, '1d10', DamageType.SLASHING, [Props.HEAVY, Props.REACH, Props.TWOHANDED]),
    'greataxe': Weapon(EquipmentType.WEAPON_MELEE, Class.MARTIAL, '1d12', DamageType.SLASHING, [Props.HEAVY, Props.TWOHANDED]),
    'greatsword': Weapon(EquipmentType.WEAPON_MELEE, Class.MARTIAL, '2d6', DamageType.SLASHING, [Props.HEAVY, Props.TWOHANDED]),
    'lance': Weapon(EquipmentType.WEAPON_MELEE, Class.MARTIAL, '1d12', DamageType.PIERCING, [Props.REACH]),
    'longsword': Weapon(EquipmentType.WEAPON_MELEE, Class.MARTIAL, '1d8', DamageType.SLASHING, [Props.VERSATILE]),
    'maul': Weapon(EquipmentType.WEAPON_MELEE, Class.MARTIAL, '2d6', DamageType.BLUDGEONING, [Props.HEAVY, Props.TWOHANDED]),
    'morningstar': Weapon(EquipmentType.WEAPON_MELEE, Class.MARTIAL, '1d8', DamageType.PIERCING, []),
    'pike': Weapon(EquipmentType.WEAPON_MELEE, Class.MARTIAL, '1d10', DamageType.PIERCING, [Props.HEAVY, Props.REACH, Props.TWOHANDED]),
    'rapier': Weapon(EquipmentType.WEAPON_MELEE, Class.MARTIAL, '1d8', DamageType.PIERCING, [Props.FINESSE]),
    'scimitar': Weapon(EquipmentType.WEAPON_MELEE, Class.MARTIAL, '1d6', DamageType.SLASHING, [Props.FINESSE, Props.LIGHT]),
    'shortsword': Weapon(EquipmentType.WEAPON_MELEE, Class.MARTIAL, '1d6', DamageType.PIERCING, [Props.FINESSE, Props.LIGHT]),
    'trident': Weapon(EquipmentType.WEAPON_MELEE, Class.MARTIAL, '1d6', DamageType.PIERCING, [Props.VERSATILE, Props.THROWN], 20, 60),
    'war-pick': Weapon(EquipmentType.WEAPON_MELEE, Class.MARTIAL, '1d8', DamageType.PIERCING, []),
    'warhammer': Weapon(EquipmentType.WEAPON_MELEE, Class.MARTIAL, '1d8', DamageType.BLUDGEONING, [Props.VERSATILE]),
    'whip': Weapon(EquipmentType.WEAPON_MELEE, Class.MARTIAL, '1d4', DamageType.SLASHING, [Props.FINESSE, Props.REACH]),
    'blowgun': Weapon(EquipmentType.WEAPON_RANGED, Class.MARTIAL, '1', DamageType.PIERCING, [Props.AMMUNITION, Props.LOADING], 25, 100),
    'crossbow-heavy': Weapon(EquipmentType.WEAPON_RANGED, Class.MARTIAL, '1d10', DamageType.PIERCING, [Props.HEAVY, Props.AMMUNITION, Props.LOADING, Props.TWOHANDED], 100, 400),
    'crossbow-hand': Weapon(EquipmentType.WEAPON_RANGED, Class.MARTIAL, '1d6', DamageType.PIERCING, [Props.LIGHT, Props.AMMUNITION, Props.LOADING], 30, 120),
    'longbow': Weapon(EquipmentType.WEAPON_RANGED, Class.MARTIAL, '1d8', DamageType.PIERCING, [Props.HEAVY, Props.AMMUNITION, Props.TWOHANDED], 150, 600),
}