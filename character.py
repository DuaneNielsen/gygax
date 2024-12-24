from dataclasses import dataclass

import conditions
from conditions import ConditionState, ConditionStateArray
import constants
from actions import ActionEntry, WeaponRange, unarmed, ActionArray
from to_jax import C, convert, JaxStringArray
from tree_serialization import CumBinType, OneHotType
import chex
from dnd_character import Character
from dnd_character.equipment import Item
from jax import numpy as jnp
from constants import DamageType, Party
import dice
from functools import partial
import jax
from constants import Abilities


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
        if 'name' not in kwargs:
            kwargs['name'] = ''
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

        self.conditions = ConditionState()
        self.effects = [ActionEntry()] * constants.N_EFFECTS

        self.concentrating = [False] * constants.MAX_TARGETS
        self.concentration_ref = [[0, 0, 0]] * constants.MAX_TARGETS # player, character, effect_slot
        self.concentration_check_cum = 1.0

    def jax(self):
        return convert(self, Character)

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




from typing import Dict, List


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


def add_damage_modifiers(a, b):
    damage_mods = jnp.stack([a, b])
    return jnp.max(damage_mods, axis=0) * jnp.min(damage_mods, axis=0)

@chex.dataclass
class Character:
    name: JaxStringArray
    hp: jnp.float16
    ac: jnp.int8
    prof_bonus: jnp.int8
    ability_mods: jnp.int8
    attack_ability_mods: jnp.int8
    save_bonus: jnp.int8
    damage_type_mul: jnp.float16
    conditions: ConditionStateArray
    effects: ActionArray
    concentrating: jnp.bool
    concentration_ref: jnp.int8
    concentration_check_cum: jnp.float16

    def __add__(self, b):
        return Character(
            name=self.name,
            hp=self.hp + b.hp,
            ac=self.ac + b.ac,
            prof_bonus=self.prof_bonus + b.prof_bonus,
            ability_mods=self.ability_mods + b.ability_mods,
            save_bonus=self.save_bonus + b.save_bonus,
            damage_type_mul=add_damage_modifiers(self.damage_type_mul, b.damage_type_mul),
            conditions=self.conditions + b.conditions,
            effects=jax.vmap(lambda x, y: x + y, self, b),
        )





