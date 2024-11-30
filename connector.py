from dataclasses import dataclass
from enum import StrEnum
from typing import Container

import chex
from dnd_character import Character
from dnd_character.SRD import SRD
from dnd_character.equipment import _Item, Item
from jax import numpy as jnp
from constants import DamageType
import dice

map_damage_type = {}
for damage_type in DamageType:
    map_damage_type[damage_type.name.lower()] = damage_type


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


@chex.dataclass
class WeaponArray:
    ability_modifier: int
    expected_damage: float
    damage_type: int
    finesse: bool


@chex.dataclass
class CharacterArray:
    armor_class: jnp.int8
    current_hp: jnp.float16
    prof_bonus: jnp.int8
    armor_class: jnp.int8
    ability_modifier: AbilitiesArray
    main_attack: WeaponArray
    ranged_main_attack: WeaponArray


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
             dexterity_ability_bonus:int,
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
            damage_type = map_damage_type[damage['damage_type']['index']]
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
            dex_bonus =  self._armor.armor_class['dex_bonus']

        if self.off_hand is not None:
            if self._off_hand.armor_class is not None:
                shield = self._off_hand.armor_class['base']

        return base_armor + shield + self.ability_modifier.dexterity * dex_bonus

    @armor_class.setter
    def armor_class(self, ac):
        # this does nothing by design, it's just to keep the parent class happy
        pass

    @property
    def main_hand(self) -> Item:
        return self._main_hand

    @main_hand.setter
    def main_hand(self, item):
        if item is not None:
            weapon_properties = set([p['index'] for p in item.properties])
            assert item.weapon_range == "Melee", "must equip melee weapon, use ranged_main_hand to equip ranged or thrown"
            assert not weapon_properties.intersection({'two-handed'}), "two handed weapons must be equipped using two hands"
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
                assert not weapon_properties.intersection({'two-handed'}), "two handed weapons must be equipped using two hands"
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
            return Attack.make(self.two_hand, self.ability_modifier.strength, self.ability_modifier.dexterity, two_hand=True)
        else:
            return Attack.make(unarmed, self.ability_modifier.strength, self.ability_modifier.dexterity)

    @property
    def offhand_attack(self):
        if self.off_hand is not None:
            return Attack.make(self.off_hand, self.ability_modifier.strength, self.ability_modifier.dexterity, off_hand=True)

    @property
    def ranged_main_hand(self) -> Item:
        return self._ranged_main_hand

    @ranged_main_hand.setter
    def ranged_main_hand(self, item):
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
            assert not weapon_properties.intersection({'two-handed'}), "two handed weapons must be equipped using two hands"
            self._ranged_off_hand = item
            self._ranged_two_hand = None
        else:
            assert False, "only shields and light weapons can be off hand equipped in the current implementation"

    @property
    def ranged_two_hand(self):
        return self._ranged_two_hand

    @ranged_two_hand.setter
    def ranged_two_hand(self, item):
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
            return Attack.make(self.ranged_main_hand, self.ability_modifier.strength, self.ability_modifier.dexterity, thrown=thrown)
        elif self.ranged_two_hand is not None:
            thrown = set([p['index'] for p in self.ranged_two_hand.properties]).intersection({"thrown"})
            return Attack.make(self.ranged_two_hand, self.ability_modifier.strength, self.ability_modifier.dexterity, two_hand=True, thrown=thrown)
        else:
            return None

    @property
    def ranged_offhand_attack(self):
        if self.ranged_off_hand is not None:
            thrown = set([p['index'] for p in self.ranged_off_hand.properties]).intersection({"thrown"})
            return Attack.make(self.ranged_off_hand, self.ability_modifier.strength, self.ability_modifier.dexterity, off_hand=True, thrown=thrown)
        else:
            return None


def default_values(clazz: type):
    kwargs = {}
    for field_name, field_info in clazz.__dataclass_fields__.items():
        if issubclass(field_info.type, Container):
            kwargs[field_name] = default_values(field_info.type)
        else:
            kwargs[field_name] = jnp.array(jnp.zeros(1, dtype=field_info.type))


def convert(character: CharacterExtra, clazz: type) -> CharacterArray:
    kwargs = {}
    for field_name, field_info in clazz.__dataclass_fields__.items():

        if issubclass(field_info.type, Container):
            field_value = getattr(character, field_name)
            if field_value is not None:
                kwargs[field_name] = convert(field_value, field_info.type)
            else:
                kwargs[field_name] = default_values(field_info.type)
        else:
            kwargs[field_name] = jnp.array(getattr(character, field_name))
            print(f"Field Name: {field_name}, Type: {field_info.type}")

    return clazz(**kwargs)
