from dataclasses import dataclass
from enum import StrEnum, auto, IntEnum


class ArmorType(StrEnum):
    CLOTH = auto()
    LIGHT = auto()
    MEDIUM = auto()
    HEAVY = auto()
    SHIELD = auto()


@dataclass
class Armor:
    ac: int
    max_dex_bonus: int
    type: ArmorType
    stealth_disadvantage: bool


armor = {
    'cloth': Armor(ac=10, max_dex_bonus=10, type=ArmorType.CLOTH, stealth_disadvantage=True),
    'padded': Armor(ac=11, max_dex_bonus=10, type=ArmorType.LIGHT, stealth_disadvantage=True),
    'leather': Armor(ac=11, max_dex_bonus=10, type=ArmorType.LIGHT, stealth_disadvantage=False),
    'studded-leather': Armor(ac=12, max_dex_bonus=10, type=ArmorType.LIGHT, stealth_disadvantage=False),
    'hide': Armor(ac=12, max_dex_bonus=2, type=ArmorType.MEDIUM, stealth_disadvantage=False),
    'chain-shirt': Armor(ac=13, max_dex_bonus=2, type=ArmorType.MEDIUM, stealth_disadvantage=False),
    'scale-mail': Armor(ac=14, max_dex_bonus=2, type=ArmorType.MEDIUM, stealth_disadvantage=True),
    'breastplate': Armor(ac=14, max_dex_bonus=2, type=ArmorType.MEDIUM, stealth_disadvantage=False),
    'half-plate': Armor(ac=14, max_dex_bonus=2, type=ArmorType.MEDIUM, stealth_disadvantage=True),
    'ring-mail': Armor(ac=14, max_dex_bonus=0, type=ArmorType.HEAVY, stealth_disadvantage=True),
    'chain-mail': Armor(ac=16, max_dex_bonus=0, type=ArmorType.HEAVY, stealth_disadvantage=True),
    'splint': Armor(ac=17, max_dex_bonus=0, type=ArmorType.HEAVY, stealth_disadvantage=True),
    'plate': Armor(ac=18, max_dex_bonus=0, type=ArmorType.HEAVY, stealth_disadvantage=True),
    'shield': Armor(ac=2, max_dex_bonus=10, type=ArmorType.SHIELD, stealth_disadvantage=False)
}
