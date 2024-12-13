from dataclasses import dataclass
from enum import StrEnum
from typing import Container, List

import constants
from tree_serialization import CumBinType, OneHotType
import chex
from dnd_character import Character
from dnd_character.SRD import SRD
from dnd_character.equipment import _Item, Item
from jax import numpy as jnp
from constants import DamageType, Party
import dice
from functools import partial
import jax
import numpy as np
import dataclasses
from constants import HitrollType, Conditions, Abilities, SaveFreq


@dataclasses.dataclass
class ActionEntry:
    name: str = ''
    damage: float = 0.
    damage_type: int = DamageType.FORCE
    req_hitroll: bool = False
    hitroll_type: HitrollType = HitrollType.SPELL
    ability_mod_damage: bool = False
    inflicts_condition: bool = False
    condition: Conditions = Conditions.POISONED
    duration: int = 0  # rounds ( 6 seconds )
    can_save: bool = False
    save: Abilities = Abilities.CON
    use_save_dc: bool = False
    save_dc: int = 0
    save_mod: float = 0.
    cum_save: float = 0.
    save_freq: SaveFreq = SaveFreq.END_TARGET_TURN
    bonus_attacks: int = 0
    bonus_spell_attacks: int = 0
    recurring_damage: float = 0.
    recurring_damage_save_mod: float = 0.

    def replace(self, **kwargs) -> 'ActionEntry':
        return dataclasses.replace(self, **kwargs)


def fix_length(text: str, target_length: int, fill_char=" ") -> str:
    if len(text) > target_length:
        return text[:target_length]
    return text.ljust(target_length, fill_char)


class JaxStringArray:
    """
    A class to represent strings in jax
    """

    @staticmethod
    def str_to_uint8_array(text: str):
        text = fix_length(text, 20)
        # Convert string to bytes, then to uint8 array
        return jnp.array(list(text.encode('utf-8')), dtype=jnp.uint8)

    @staticmethod
    def uint8_array_to_str(arr):
        # Convert uint8 array to bytes, then to string
        arr_cpu = np.asarray(arr)
        return bytes(arr_cpu).decode('utf-8').strip()


class WeaponRange(StrEnum):
    MELEE = 'Melee'
    RANGED = 'Ranged'


unarmed = _Item(
    index="unarmed",
    name="unarmed",
    equipment_category=SRD('/api/equipment-categories/weapon'),
    weapon_category='Simple',
    weapon_range=WeaponRange.MELEE,
    range={'normal': 5},
    damage={
        'damage_dice': '1d1',
        'damage_type': SRD('/api/damage-types/bludgeoning')
    },
    contents=None,
    cost=0,
    desc='unarmed attack',
    properties=[],
    special=None,
    url=None
)


@chex.dataclass
class AbilitiesArray:
    strength: jnp.int8
    dexterity: jnp.int8
    constitution: jnp.int8
    intelligence: jnp.int8
    wisdom: jnp.int8
    charisma: jnp.int8


# scaling constants for observations
HP_LOWER = 0  # dont worry about negative hitpoints for now
HP_UPPER = 20
AC_LOWER = 5
AC_UPPER = 20
PROF_BONUS_LOWER = 0
PROF_BONUS_UPPER = 6
ABILITY_MODIFIER_LOWER = -5
ABILITY_MODIFIER_UPPER = 10
CONDITION_STACKS_UPPER = 5
ACTION_RESOURCES_UPPER = 5
DAMAGE_UPPER = 20

AbilityModCumBin = CumBinType('AbilityModCumBin', (), {}, upper=ABILITY_MODIFIER_UPPER, lower=ABILITY_MODIFIER_LOWER)
DamageCumBin = CumBinType('AbilityModCumBin', (), {}, upper=DAMAGE_UPPER)
ProfBonusCumBin = CumBinType('AbilityModCumBin', (), {}, upper=PROF_BONUS_UPPER, lower=PROF_BONUS_LOWER)
ArmorClassCumBin = CumBinType('AbilityModCumBin', (), {}, upper=AC_UPPER, lower=AC_LOWER)
HitpointsCumBin = CumBinType('AbilityModCumBin', (), {}, upper=HP_UPPER, lower=HP_LOWER)
DamageTypeOneHot = OneHotType('DamageTypeOneHot', (), {}, n_clessas=len(DamageType))
CharacterClassOneHot = OneHotType('CharacterClassOneHot', (), {}, n_clessas=len(constants.CharacterClass))


@chex.dataclass
class AbilityModifierObservation:
    strength: AbilityModCumBin
    dexterity: AbilityModCumBin
    constitution: AbilityModCumBin
    intelligence: AbilityModCumBin
    wisdom: AbilityModCumBin
    charisma: AbilityModCumBin


@chex.dataclass
class WeaponArray:
    ability_modifier: jnp.int8
    expected_damage: jnp.float32
    damage_type: jnp.int8
    finesse: jnp.bool


@chex.dataclass
class WeaponObservation:
    ability_modifier: AbilityModCumBin
    expected_damage: DamageCumBin
    damage_type: DamageTypeOneHot
    finesse: jnp.bool


@chex.dataclass
class CharacterArray:
    character_class: jnp.int8
    armor_class: jnp.int8
    current_hp: jnp.float16
    max_hp: jnp.float16
    prof_bonus: jnp.int8
    ability_modifier: AbilitiesArray
    main_attack: WeaponArray
    ranged_main_attack: WeaponArray
    dead: jnp.bool


@chex.dataclass
class CharacterObservation:
    character_class: CharacterClassOneHot
    armor_class: ArmorClassCumBin
    current_hp: HitpointsCumBin
    prof_bonus: ProfBonusCumBin
    ability_modifier: AbilityModifierObservation
    main_attack: WeaponObservation
    ranged_main_attack: WeaponObservation
    dead: bool


@dataclass
class Attack:
    ability_modifier: int
    damage_dice: str
    expected_damage: float
    damage_type: int
    weapon_range: str
    finesse: bool = False
    range_normal: int = 5
    range_long: int = 5

    @staticmethod
    def make(weapon: Item,
             strength_ability_bonus: int,
             dexterity_ability_bonus: int,
             off_hand=False, two_hand=False, thrown=False):
        """
        returns a dataclass that holds the attack type and damage of the weapon, including ability bonuses
        where applicable under the rules
        Args:
            weapon:
            strength_ability_bonus:
            dexterity_ability_bonus:
            off_hand:
            two_hand:
            thrown:

        Returns:

        """
        weapon_properties = set([p['index'] for p in weapon.properties])
        ability_bonus = strength_ability_bonus
        finesse = False

        # switch to dex if ranged or it's better
        if weapon.weapon_range == WeaponRange.RANGED:
            ability_bonus = dexterity_ability_bonus
        elif weapon_properties.intersection({"finesse"}):
            if dexterity_ability_bonus > strength_ability_bonus:
                ability_bonus = dexterity_ability_bonus
                finesse = True

        def get_damage(damage, ability_bonus, off_hand):
            ability_bonus = min(ability_bonus, 0) if off_hand else ability_bonus
            damage_dice = damage["damage_dice"]
            expected_damage = max(dice.expected_roll(damage_dice) + ability_bonus, 0)
            damage_type = DamageType[damage['damage_type']['index'].upper()]
            return damage_dice, expected_damage, damage_type

        damage_dice, expected_damage, damage_type = get_damage(weapon.damage, ability_bonus, off_hand)
        if weapon.two_handed_damage is not None:
            if two_hand:
                damage_dice, expected_damage, damage_type = get_damage(weapon.two_handed_damage, ability_bonus, False)

        if weapon.weapon_range == WeaponRange.MELEE:
            return Attack(
                ability_modifier=ability_bonus,
                weapon_range=weapon.weapon_range,
                damage_dice=weapon.damage["damage_dice"],
                expected_damage=expected_damage,
                damage_type=damage_type,
                finesse=finesse
            )
        elif weapon.weapon_range == WeaponRange.RANGED:
            return Attack(
                ability_modifier=ability_bonus,
                weapon_range=weapon.weapon_range,
                damage_dice=weapon.damage["damage_dice"],
                expected_damage=expected_damage,
                damage_type=damage_type,
                range_normal=weapon.range['normal'],
                range_long=weapon.range['long']
            )
        elif thrown:
            return Attack(
                ability_modifier=ability_bonus,
                weapon_range=weapon.weapon_range,
                damage_dice=weapon.damage["damage_dice"],
                expected_damage=expected_damage,
                damage_type=damage_type,
                range_normal=weapon.throw_range['normal'],
                range_long=weapon.throw_range['long']
            )


@dataclass
class Abilities:
    strength: int
    dexterity: int
    constitution: int
    intelligence: int
    wisdom: int
    charisma: int

    def __iter__(self):
        return iter(self.__dict__.items())


class CharacterExtra(Character):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ability_modifier = Abilities(
            strength=self.get_ability_modifier(self.strength),
            dexterity=self.get_ability_modifier(self.dexterity),
            constitution=self.get_ability_modifier(self.constitution),
            intelligence=self.get_ability_modifier(self.intelligence),
            wisdom=self.get_ability_modifier(self.wisdom),
            charisma=self.get_ability_modifier(self.charisma)
        )

        self._main_hand = None
        self._off_hand = None
        self._two_hand = None

        # for our rules, the weapon used will depend on the slot
        self._ranged_main_hand = None
        self._ranged_off_hand = None
        self._ranged_two_hand = None
        self._armor = None

        self.conditions = [False] * len(constants.Conditions)
        self.effect_active = [False] * constants.N_EFFECTS
        self.effects = [ActionEntry()] * constants.N_EFFECTS

    @property
    def hp(self):
        return self.current_hp

    @property
    def ability_mods(self):
        return [
            self.ability_modifier.strength,
            self.ability_modifier.dexterity,
            self.ability_modifier.constitution,
            self.ability_modifier.intelligence,
            self.ability_modifier.wisdom,
            self.ability_modifier.charisma
        ]

    @property
    def ac(self):
        return self.armor_class

    @property
    def spell_ability_mod(self):
        spell_ability_mod = self.ability_modifier.charisma
        if self.character_class == constants.CharacterClass.CLERIC:
            spell_ability_mod = self.ability_modifier.wisdom
        if self.character_class == constants.CharacterClass.DRUID:
            spell_ability_mod = self.ability_modifier.wisdom
        if self.character_class == constants.CharacterClass.MONK:
            spell_ability_mod = self.ability_modifier.wisdom
        if self.character_class == constants.CharacterClass.RANGER:
            spell_ability_mod = self.ability_modifier.wisdom
        if self.character_class == constants.CharacterClass.WIZARD:
            spell_ability_mod = self.ability_modifier.intelligence
        return spell_ability_mod

    @property
    def save_bonus(self):
        save_bonus = [0] * 6
        bonus_saves = set(self.saving_throws)
        for i, save in enumerate(['STR', 'DEX', 'CON', 'INT', 'WIS', 'CHA']):
            if save in bonus_saves:
                save_bonus[i] = 1
        return save_bonus

    @property
    def attack_ability_mods(self):
        attack_ability_mods = [None] * len(constants.HitrollType)
        attack_ability_mods[constants.HitrollType.SPELL] = self.spell_ability_mod
        attack_ability_mods[constants.HitrollType.MELEE] = self.ability_modifier.strength
        attack_ability_mods[constants.HitrollType.RANGED] = self.ability_modifier.dexterity
        attack_ability_mods[constants.HitrollType.FINESSE] = self.ability_modifier.dexterity
        return attack_ability_mods

    @property
    def character_class(self):
        return constants.CharacterClass[self.class_name.upper()]

    @property
    def armor(self):
        return self._armor

    @armor.setter
    def armor(self, armor: Item):
        if armor is None:
            self._armor = None
            return
        assert armor.equipment_category['index'] == 'armor', "only armor can be equipped in armor slot"
        assert armor.armor_category != 'Shield', "shields should be equipped in the off_hand slot"
        self._armor = armor

    @property
    def armor_class(self):
        base_armor, dex_bonus, shield = 10, True, 0
        if self._armor is not None:
            base_armor = self._armor.armor_class['base']
            dex_bonus = self._armor.armor_class['dex_bonus']

        if self.off_hand is not None:
            if self._off_hand.armor_class is not None:
                shield = self._off_hand.armor_class['base']

        return base_armor + shield + self.ability_modifier.dexterity * dex_bonus

    @armor_class.setter
    def armor_class(self, ac):
        # this does nothing by design, it's just to keep the parent class happy
        pass

    @property
    def damage_type_mul(self):
        return [1.] * len(DamageType)

    @property
    def main_hand(self) -> Item:
        return self._main_hand

    @main_hand.setter
    def main_hand(self, item):
        if item is not None:
            weapon_properties = set([p['index'] for p in item.properties])
            assert item.weapon_range == "Melee", "must equip melee weapon, use ranged_main_hand to equip ranged or thrown"
            assert not weapon_properties.intersection(
                {'two-handed'}), "two handed weapons must be equipped using two hands"
            self._main_hand = item
            self._two_hand = None
        else:
            self._main_hand = unarmed
            self._two_hand = None

    @property
    def off_hand(self) -> Item:
        return self._off_hand

    @off_hand.setter
    def off_hand(self, item):
        if item is not None:
            if item.armor_category == 'Shield':
                self._off_hand = item
                self._two_hand = None
            elif item.equipment_category['index'] == "weapon":
                weapon_properties = set([p['index'] for p in item.properties])
                assert item.weapon_range == "Melee", "must equip melee weapon, use ranged_off_hand to equip ranged or thrown"
                assert weapon_properties.intersection({'light'}), "only light weapons can be equipped in off hand"
                assert not weapon_properties.intersection(
                    {'two-handed'}), "two handed weapons must be equipped using two hands"
                self._off_hand = item
                self._two_hand = None
            else:
                assert False, "only shields and light weapons can be off hand equipped in the current implementation"
        else:
            self._off_hand = None

    @property
    def two_hand(self):
        return self._two_hand

    @two_hand.setter
    def two_hand(self, item):
        if item is not None:
            weapon_properties = set([p['index'] for p in item.properties])
            assert item.weapon_range == "Melee", "must equip melee weapon, use ranged_two_hand to equip ranged or thrown"
            assert {'two-handed', 'versatile'} & weapon_properties, "weapon was not versatile or two handed"
            self._two_hand = item
            self._main_hand = None
            self._off_hand = None

    @property
    def main_attack(self):
        if self.main_hand is not None:
            return Attack.make(self.main_hand, self.ability_modifier.strength, self.ability_modifier.dexterity)
        elif self.two_hand is not None:
            return Attack.make(self.two_hand, self.ability_modifier.strength, self.ability_modifier.dexterity,
                               two_hand=True)
        else:
            return Attack.make(unarmed, self.ability_modifier.strength, self.ability_modifier.dexterity)

    @property
    def offhand_attack(self):
        if self.off_hand is not None:
            return Attack.make(self.off_hand, self.ability_modifier.strength, self.ability_modifier.dexterity,
                               off_hand=True)

    @property
    def ranged_main_hand(self) -> Item:
        return self._ranged_main_hand

    @ranged_main_hand.setter
    def ranged_main_hand(self, item):
        if item is not None:
            weapon_properties = set([p['index'] for p in item.properties])
            thrown = {"thrown"} & weapon_properties
            assert item.weapon_range == "Ranged" or thrown, "can only equip ranged or thrown in ranged_main_hand"
            assert not weapon_properties & {'two-handed'}, "two handed weapons must be equipped using two hands"
        self._ranged_main_hand = item
        self._ranged_two_hand = None

    @property
    def ranged_off_hand(self) -> Item:
        return self._ranged_off_hand

    @ranged_off_hand.setter
    def ranged_off_hand(self, item):
        if item.index == 'shield':
            self._ranged_off_hand = item
            self._ranged_two_hand = None
        elif item.equipment_category['index'] == "weapon":
            weapon_properties = set([p['index'] for p in item.properties])
            thrown = weapon_properties & {"thrown"}
            assert item.weapon_range == "Ranged" or thrown, "can only equip ranged or thrown in ranged_off_hand"
            assert weapon_properties & {'light'}, "only light weapons can be equipped in off hand"
            assert not weapon_properties.intersection(
                {'two-handed'}), "two handed weapons must be equipped using two hands"
            self._ranged_off_hand = item
            self._ranged_two_hand = None
        else:
            assert False, "only shields and light weapons can be off hand equipped in the current implementation"

    @property
    def ranged_two_hand(self):
        return self._ranged_two_hand

    @ranged_two_hand.setter
    def ranged_two_hand(self, item):
        if item is not None:
            weapon_properties = set([p['index'] for p in item.properties])
            thrown = weapon_properties.intersection({"thrown"})
            assert item.weapon_range == "Ranged" or thrown, "can only equip ranged or thrown in ranged_off_hand"
            assert weapon_properties.intersection({'two-handed', 'versatile'}), "weapon was not versatile or two handed"
        self._ranged_two_hand = item
        self._ranged_main_hand = None
        self._ranged_off_hand = None

    @property
    def ranged_main_attack(self):
        if self.ranged_main_hand is not None:
            thrown = set([p['index'] for p in self.ranged_main_hand.properties]).intersection({"thrown"})
            return Attack.make(self.ranged_main_hand, self.ability_modifier.strength, self.ability_modifier.dexterity,
                               thrown=thrown)
        elif self.ranged_two_hand is not None:
            thrown = set([p['index'] for p in self.ranged_two_hand.properties]).intersection({"thrown"})
            return Attack.make(self.ranged_two_hand, self.ability_modifier.strength, self.ability_modifier.dexterity,
                               two_hand=True, thrown=thrown)
        else:
            return None

    @property
    def ranged_offhand_attack(self):
        if self.ranged_off_hand is not None:
            thrown = set([p['index'] for p in self.ranged_off_hand.properties]).intersection({"thrown"})
            return Attack.make(self.ranged_off_hand, self.ability_modifier.strength, self.ability_modifier.dexterity,
                               off_hand=True, thrown=thrown)
        else:
            return None


from typing import TypeVar

C = TypeVar('C')


def default_values(clazz: C) -> C:
    kwargs = {}
    for field_name, field_info in clazz.__dataclass_fields__.items():
        if issubclass(field_info.type, Container):
            kwargs[field_name] = default_values(field_info.type)
        else:
            kwargs[field_name] = jnp.array(jnp.zeros(1, dtype=field_info.type))
    return clazz(**kwargs)




def convert(character: CharacterExtra, clazz: C) -> C:
    kwargs = {}
    for field_name, field_info in clazz.__dataclass_fields__.items():

        if issubclass(field_info.type, Container):
            field_value = getattr(character, field_name)
            if field_value is not None:
                kwargs[field_name] = convert(field_value, field_info.type)
            else:
                kwargs[field_name] = default_values(field_info.type)
        elif issubclass(field_info.type, JaxStringArray):
            if type(character) is list:
                expanded_arrays = [JaxStringArray.str_to_uint8_array(getattr(s, field_name)) for s in character]
                kwargs[field_name] = jnp.stack(expanded_arrays)
            else:
                field_value = getattr(character, field_name)
                kwargs[field_name] = JaxStringArray.str_to_uint8_array(field_value)
        else:
            if type(character) is list:
                expanded_arrays = [getattr(x, field_name) for x in character]
                kwargs[field_name] = jnp.array(expanded_arrays, dtype=field_info.type)
            else:
                kwargs[field_name] = jnp.array(getattr(character, field_name), dtype=field_info.type)

    return clazz(**kwargs)


from typing import Tuple, Dict, List


def stack_party(party: Dict[str, List[CharacterExtra]], clazz: C) -> C:
    """
    Converts a next dict of CharactersExtras into a character array
    Args:
        {
            Party.PC: [fizban, jimmy, goldmoon, riverwind],
            Party.NPC: [raistlin, joffrey, clarion, pikachu]
        }

        where each value is a CharacterExtra
    Returns: converted to the parameterized class

    """

    party = [
        party[Party.PC],
        party[Party.NPC]
    ]

    jax_party = jax.tree.map(partial(convert, clazz=clazz), party)

    def stack(*args):
        return jnp.stack(args)

    parties = []
    for i in range(2):
        parties += [jax.tree.map(stack, *jax_party[i])]
    return jax.tree.map(stack, *parties)
