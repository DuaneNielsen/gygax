import pytest
from dnd_character import CLASSES
from dnd_character.equipment import Item
import jax.numpy as jnp

import constants
from character import DamageType, CharacterArray, CharacterExtra
from actions import WeaponRange
from to_jax import convert


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

    fighter.ranged_two_hand = Item("longbow")
    weapon = fighter.ranged_main_attack
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


def test_equip_thrown_weapon():
    ranger = CharacterExtra(
        classs=CLASSES["ranger"],
        strength=12,
        dexterity=16,
        constitution=16,
        intelligence=8,
        wisdom=12,
        charisma=8
    )
    ranger.main_hand = Item('handaxe')
    ranger.off_hand = Item('handaxe')
    ranger.ranged_main_hand = Item('handaxe')
    ranger.ranged_off_hand = Item('handaxe')

    assert ranger.main_attack.finesse == False
    assert ranger.main_attack.ability_modifier == 1
    assert ranger.main_attack.expected_damage == 4.5

    assert ranger.offhand_attack.finesse == False
    assert ranger.offhand_attack.ability_modifier == 1
    assert ranger.offhand_attack.expected_damage == 3.5

    assert ranger.ranged_main_attack.finesse == False
    assert ranger.ranged_main_attack.ability_modifier == 1
    assert ranger.ranged_main_attack.expected_damage == 4.5

    assert ranger.ranged_offhand_attack.finesse == False
    assert ranger.ranged_offhand_attack.ability_modifier == 1
    assert ranger.ranged_offhand_attack.expected_damage == 3.5


def test_melee_assertions():
    fighter = CharacterExtra(
        classs=CLASSES["fighter"],
        strength=16,
        dexterity=12,
        constitution=16,
        intelligence=8,
        wisdom=12,
        charisma=8
    )

    fighter.main_hand = Item('longsword')
    fighter.two_hand = Item('longsword')
    fighter.off_hand = Item('dagger')
    fighter.two_hand = Item('greatsword')

    with pytest.raises(AssertionError) as exc_info:
        fighter.off_hand = Item('longsword')
    with pytest.raises(AssertionError) as exc_info:
        fighter.two_hand = Item('longbow')
    with pytest.raises(AssertionError) as exc_info:
        fighter.main_hand = Item('longbow')
    with pytest.raises(AssertionError) as exc_info:
        fighter.off_hand = Item('longbow')
    with pytest.raises(AssertionError) as exc_info:
        fighter.main_hand = Item('greatsword')
    with pytest.raises(AssertionError) as exc_info:
        fighter.off_hand = Item('greatsword')

    fighter.ranged_two_hand = Item('crossbow-light')
    fighter.ranged_off_hand = Item('crossbow-hand')
    fighter.ranged_main_hand = Item('crossbow-hand')
    fighter.ranged_main_hand = Item('dagger')
    fighter.ranged_off_hand = Item('dagger')

    with pytest.raises(AssertionError) as exc_info:
        fighter.ranged_two_hand = Item('dagger')

    with pytest.raises(AssertionError) as exc_info:
        fighter.ranged_main_hand = Item('longsword')
    with pytest.raises(AssertionError) as exc_info:
        fighter.ranged_off_hand = Item('longsword')
    with pytest.raises(AssertionError) as exc_info:
        fighter.ranged_two_hand = Item('longsword')

    with pytest.raises(AssertionError) as exc_info:
        fighter.ranged_main_hand = Item('longbow')

    with pytest.raises(AssertionError) as exc_info:
        fighter.ranged_off_hand = Item('crossbow-heavy')
    with pytest.raises(AssertionError) as exc_info:
        fighter.ranged_off_hand = Item('crossbow-light')
    with pytest.raises(AssertionError) as exc_info:
        fighter.ranged_main_hand = Item('crossbow-light')


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
    fighter.main_hand = Item("longsword")
    fighter.ranged_two_hand = Item("longbow")

    fighter_jaxxed = convert(fighter, CharacterArray)

    assert fighter_jaxxed.character_class == constants.CharacterClass.FIGHTER
    assert fighter_jaxxed.current_hp == 13.
    assert fighter_jaxxed.current_hp.dtype == jnp.float16
    assert fighter_jaxxed.ability_modifier.dexterity == 1
    assert fighter_jaxxed.main_attack.finesse == False
    assert fighter_jaxxed.main_attack.expected_damage == 7.5
    assert fighter_jaxxed.main_attack.ability_modifier == 3
    assert fighter_jaxxed.main_attack.damage_type == DamageType.SLASHING
    assert fighter_jaxxed.ranged_main_attack.finesse == False
    assert fighter_jaxxed.ranged_main_attack.ability_modifier == 1
    assert fighter_jaxxed.ranged_main_attack.expected_damage == 5.5
    assert fighter_jaxxed.ranged_main_attack.damage_type == DamageType.PIERCING


def test_equip_armor():
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
    assert fighter.armor_class == Item('chain-mail').armor_class['base']
    fighter.off_hand = Item('shield')
    assert fighter.armor_class == Item('chain-mail').armor_class['base'] + 2
    fighter.off_hand = None
    assert fighter.armor_class == Item('chain-mail').armor_class['base']
    fighter.armor = None
    assert fighter.armor_class == 10 + 1
    fighter.armor = Item('leather-armor')
    assert fighter.armor_class == 11 + 1
    fighter.off_hand = Item('shield')
    assert fighter.armor_class == 11 + 1 + 2
    fighter.off_hand = Item('dagger')
    assert fighter.armor_class == 11 + 1

    with pytest.raises(AssertionError) as exc_info:
        fighter.armor = Item('crossbow-light')
