from enum import IntEnum, auto
from dataclasses import dataclass


class EquipmentType(IntEnum):
    EMPTY = auto()
    ARMOR = auto()
    WEAPON_MELEE = auto()
    WEAPON_RANGED = auto()


@dataclass
class Equipment:
    type: EquipmentType


empty = Equipment(EquipmentType.EMPTY)