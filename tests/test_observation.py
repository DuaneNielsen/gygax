import pytest
from pytest import fixture

import constants
from constants import *
from default_config import default_config
from character import stack_party, CharacterExtra, AbilitiesArray, AbilityModifierObservation, WeaponArray, WeaponObservation, \
    CharacterArray, CharacterObservation
from to_jax import convert
from dnd_character import CLASSES
from dnd_character.equipment import Item
from character import AC_LOWER, ABILITY_MODIFIER_LOWER

fizban, jimmy, goldmoon, riverwind = (0, 0), (0, 1), (0, 2), (0, 3)
raistlin, joffrey, clarion, pikachu = (1, 0), (1, 1), (1, 2), (1, 3)


@fixture
def party():
    names, party = stack_party(default_config[ConfigItems.PARTY])
    return party


from tree_serialization import convert_to_observation


def test_observe_ability_modifier():
    ability_mods = AbilitiesArray(
        strength=jnp.int8(3),
        dexterity=jnp.int8(1),
        constitution=jnp.int8(2),
        intelligence=jnp.int8(-2),
        wisdom=jnp.int8(0),
        charisma=jnp.int8(-1)
    )

    ability_mods_cumbin = convert_to_observation(ability_mods, AbilityModifierObservation)
    assert ability_mods_cumbin.strength.sum() + ABILITY_MODIFIER_LOWER - 1 == ability_mods.strength
    assert ability_mods_cumbin.charisma.sum() + ABILITY_MODIFIER_LOWER - 1 == ability_mods.charisma
    assert ability_mods_cumbin.wisdom.sum() + ABILITY_MODIFIER_LOWER - 1 == ability_mods.wisdom


def test_observe_weapon_array():
    weapon_array = WeaponArray(
        ability_modifier=jnp.int32(3),
        expected_damage=jnp.float32(4),
        damage_type=jnp.int32(3),
        finesse=jnp.bool(False)
    )

    weapon_array_obs = convert_to_observation(weapon_array, WeaponObservation)
    assert weapon_array_obs.expected_damage.sum() - 1 == 4
    assert jnp.argmax(weapon_array_obs.damage_type) == 3
    assert weapon_array_obs.finesse == False

@pytest.fixture
def fighter() -> CharacterExtra:
    fighter = CharacterExtra(
        classs=CLASSES["fighter"],
        strength=16,
        dexterity=12,
        constitution=16,
        intelligence=8,
        wisdom=12,
        charisma=8
    )
    fighter.armor = Item('chain-mail')
    fighter.main_hand = Item('longsword')
    fighter.off_hand = Item('shield')
    fighter.ranged_two_hand = Item('shortbow')
    return fighter


def test_observe_character(fighter):

    fighter_array = convert(fighter, CharacterArray)
    fighter_obs = convert_to_observation(fighter_array, CharacterObservation)
    assert jnp.argmax(fighter_obs.character_class) == constants.CharacterClass.FIGHTER
    assert fighter_obs.armor_class.sum() + AC_LOWER - 1 == fighter.armor_class
    assert fighter_obs.ranged_main_attack.ability_modifier.sum() + ABILITY_MODIFIER_LOWER -1 == fighter.ability_modifier.dexterity
    assert fighter_obs.main_attack.ability_modifier.sum() + ABILITY_MODIFIER_LOWER - 1 == fighter.ability_modifier.strength