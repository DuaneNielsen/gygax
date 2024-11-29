import pytest
from dnd_character import Character, CLASSES
from dnd_character.equipment import Item, _Item
import jax
import jax.numpy as jnp
from typing import NamedTuple, Any
import chex
from enum import StrEnum, auto
import inspect
from collections.abc import Container
from dataclasses import dataclass
import dice
from dnd_character.SRD import SRD
from constants import DamageType


class WeaponRange(StrEnum):
    MELEE = 'Melee'
    RANGED = 'Ranged'


class WeaponCategory(StrEnum):
    SIMPLE = 'Simple'
    MARTIAL = 'Martial'


unarmed = _Item(
    index="unarmed",
    name="unarmed",
    equipment_category=SRD('/api/equipment-categories/weapon'),
    weapon_category=WeaponCategory.SIMPLE,
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


def simplify_dict_with_index(data, is_root=True):
    """
    Recursively processes a dictionary and replaces any nested dictionaries
    containing an 'index' key with just their index value, except for the root dictionary.

    Args:
        data: Input dictionary or list to process
        is_root: Boolean indicating if this is the root dictionary

    Returns:
        Processed dictionary or list with simplified structure
    """
    if isinstance(data, dict):
        # If it's a dictionary with 'index' and not the root, return just the index value
        if 'index' in data:
            if is_root:
                data['name'] = data['index']
            else:
                return data['index']

        # Otherwise, process each value in the dictionary
        return {key: simplify_dict_with_index(value, is_root=False)
                for key, value in data.items()}

    elif isinstance(data, list):
        # If it's a list, process each element
        return [simplify_dict_with_index(item, is_root=False) for item in data]

    else:
        # Return non-dict and non-list values as is
        return data


map_damage_type = {}
for damage_type in DamageType:
    map_damage_type[damage_type.name.lower()] = damage_type


@chex.dataclass
class AbilitiesArray:
    strength: jnp.int8
    dexterity: jnp.int8
    constitution: jnp.int8
    intelligence: jnp.int8
    wisdom: jnp.int8
    charisma: jnp.int8


@chex.dataclass
class CharacterArray:
    current_hp: jnp.float16
    prof_bonus: jnp.int8
    armor_class: jnp.int8
    ability_modifier: AbilitiesArray


@dataclass
class Weapon:
    ability_modifier: int
    damage_dice: str
    expected_damage: float
    damage_type: int
    weapon_range: str
    finesse: bool = False
    range_normal: int = 5
    range_long: int = 5

    @staticmethod
    def make(weapon, strength_ability_bonus, dexterity_ability_bonus, off_hand=False, two_hand=False, thrown=False):

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
            damage_type = map_damage_type[damage['damage_type']['index']]
            return damage_dice, expected_damage, damage_type

        damage_dice, expected_damage, damage_type = get_damage(weapon.damage, ability_bonus, off_hand)
        if weapon.two_handed_damage is not None:
            if two_hand:
                damage_dice, expected_damage, damage_type = get_damage(weapon.two_handed_damage, ability_bonus, False)

        if weapon.weapon_range == WeaponRange.MELEE:
            return Weapon(
                ability_modifier=ability_bonus,
                weapon_range=weapon.weapon_range,
                damage_dice=weapon.damage["damage_dice"],
                expected_damage=expected_damage,
                damage_type=damage_type,
                finesse=finesse
            )
        elif weapon.weapon_range == WeaponRange.RANGED:
            return Weapon(
                ability_modifier=ability_bonus,
                weapon_range=weapon.weapon_range,
                damage_dice=weapon.damage["damage_dice"],
                expected_damage=expected_damage,
                damage_type=damage_type,
                range_normal=weapon.range['normal'],
                range_long=weapon.range['long']
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

    @property
    def main_hand(self) -> Item:
        return self._main_hand

    @main_hand.setter
    def main_hand(self, item):
        weapon_properties = set([p['index'] for p in item.properties])
        assert not weapon_properties.intersection({'two-handed'}), "two handed weapons must be equipped using two hands"
        self._main_hand = item
        self._two_hand = None

    @property
    def off_hand(self) -> Item:
        return self._off_hand

    @off_hand.setter
    def off_hand(self, item):
        if item.index == 'shield':
            self._off_hand = item
            self._two_hand = None
        elif item.equipment_category['index'] == "weapon":
            weapon_properties = set([p['index'] for p in item.properties])
            assert weapon_properties.intersection({'light'}), "only light weapons can be equipped in off hand"
            self._off_hand = item
            self._two_hand = None
        else:
            assert False, "only shields and light weapons can be off hand equipped in the current implementation"

    @property
    def two_hand(self):
        return self._two_hand

    @two_hand.setter
    def two_hand(self, item):
        weapon_properties = set([p['index'] for p in item.properties])
        assert weapon_properties.intersection({'two-handed', 'versatile'}), "weapon was not versatile or two handed"
        self._two_hand = item
        self._main_hand = None
        self._off_hand = None

    @property
    def main_attack(self):
        if self.main_hand is not None:
            return Weapon.make(self.main_hand, self.ability_modifier.strength, self.ability_modifier.dexterity)
        elif self.two_hand is not None:
            return Weapon.make(self.two_hand, self.ability_modifier.strength, self.ability_modifier.dexterity, two_hand=True)
        else:
            return Weapon.make(unarmed, self.ability_modifier.strength, self.ability_modifier.dexterity)

    @property
    def offhand_attack(self):
        if self.off_hand is not None:
            return Weapon.make(self.off_hand, self.ability_modifier.strength, self.ability_modifier.dexterity, off_hand=True)


def convert_to_character_array(character: CharacterExtra, clazz: type):
    character_dict = dict(character)
    kwargs = {}
    for field_name, field_info in clazz.__dataclass_fields__.items():

        if issubclass(field_info.type, Container):
            kwargs[field_name] = convert_to_character_array(character_dict[field_name], field_info.type)
        else:
            kwargs[field_name] = jnp.array(character_dict[field_name])
            print(f"Field Name: {field_name}, Type: {field_info.type}")

    return clazz(**kwargs)


def test_simplify_dict():
    longsword = Item("longsword")
    longsword_simplified = simplify_dict_with_index(dict(longsword))
    print(longsword_simplified)


def test_equip_longsword():
    fighter = CharacterExtra(
        classs=CLASSES["fighter"],
        strength=16,
        dexterity=12,
        constitution=16,
        intelligence=8,
        wisdom=12,
        charisma=8
    )

    fighter.main_hand = Item("longsword")
    weapon = fighter.main_attack
    assert weapon.expected_damage == 4.5 + fighter.ability_modifier.strength
    assert weapon.damage_type == DamageType.SLASHING
    assert weapon.ability_modifier == fighter.ability_modifier.strength
    assert weapon.weapon_range == WeaponRange.MELEE

    fighter.two_hand = Item("longsword")
    weapon = fighter.main_attack
    assert weapon.expected_damage == 5.5 + fighter.ability_modifier.strength
    assert weapon.damage_type == DamageType.SLASHING
    assert weapon.ability_modifier == fighter.ability_modifier.strength
    assert weapon.weapon_range == WeaponRange.MELEE


def test_equip_longbow():
    fighter = CharacterExtra(
        classs=CLASSES["fighter"],
        strength=16,
        dexterity=12,
        constitution=16,
        intelligence=8,
        wisdom=12,
        charisma=8
    )

    fighter.two_hand = Item("longbow")
    weapon = fighter.main_attack
    assert weapon.expected_damage == 4.5 + fighter.ability_modifier.dexterity
    assert weapon.damage_type == DamageType.PIERCING
    assert weapon.ability_modifier == fighter.ability_modifier.dexterity
    assert weapon.weapon_range == WeaponRange.RANGED
    assert weapon.range_normal == 150
    assert weapon.range_long == 600


def test_equip_finesse_weapon():
    fighter = CharacterExtra(
        classs=CLASSES["fighter"],
        strength=16,
        dexterity=12,
        constitution=16,
        intelligence=8,
        wisdom=12,
        charisma=8
    )
    ranger = CharacterExtra(
        classs=CLASSES["ranger"],
        strength=12,
        dexterity=16,
        constitution=16,
        intelligence=8,
        wisdom=12,
        charisma=8
    )

    fighter.main_hand = Item('rapier')
    ranger.main_hand = Item('rapier')

    assert fighter.main_attack.finesse == False
    assert fighter.main_attack.ability_modifier == 3
    assert fighter.main_attack.expected_damage == 4.5 + fighter.ability_modifier.strength

    assert ranger.main_attack.finesse == True
    assert ranger.main_attack.ability_modifier == 3
    assert ranger.main_attack.expected_damage == 4.5 + ranger.ability_modifier.dexterity


def test_equip_two_weapon():
    ranger = CharacterExtra(
        classs=CLASSES["ranger"],
        strength=12,
        dexterity=16,
        constitution=16,
        intelligence=8,
        wisdom=12,
        charisma=8
    )

    ranger.main_hand = Item('rapier')
    ranger.off_hand = Item('dagger')

    assert ranger.main_attack.finesse == True
    assert ranger.main_attack.ability_modifier == 3
    assert ranger.main_attack.expected_damage == 4.5 + ranger.ability_modifier.dexterity

    assert ranger.offhand_attack.finesse == True
    assert ranger.offhand_attack.ability_modifier == 3
    assert ranger.offhand_attack.expected_damage == 2.5

    ranger = CharacterExtra(
        classs=CLASSES["ranger"],
        strength=6,
        dexterity=8,
        constitution=16,
        intelligence=8,
        wisdom=12,
        charisma=8
    )
    ranger.main_hand = Item('rapier')
    ranger.off_hand = Item('dagger')

    assert ranger.offhand_attack.finesse == True
    assert ranger.offhand_attack.ability_modifier == -1
    assert ranger.offhand_attack.expected_damage == 1.5



def test_convert_fighter():
    fighter = CharacterExtra(
        classs=CLASSES["fighter"],
        strength=16,
        dexterity=12,
        constitution=16,
        intelligence=8,
        wisdom=12,
        charisma=8
    )
    fighter.give_item(Item("chain-mail"))
    fighter.give_item(Item("longsword"))
    fighter.give_item(Item("longbow"))
    fighter.main_hand = 1
    fighter.off_hand = 2

    assert fighter.off_hand == "longbow"

    fighter_jaxxed = convert_to_character_array(fighter, CharacterArray)
    assert fighter_jaxxed.current_hp == 13.
    assert fighter_jaxxed.current_hp.dtype == jnp.int32
    assert fighter_jaxxed.ability_modifier.dexterity == 1
