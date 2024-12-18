import pytest
from constants import CharacterClass, DamageType, Race
from conditions import Conditions


def test_character_classes():
    assert CharacterClass['fighter'.upper()] == CharacterClass.FIGHTER


def test_race():
    assert Race['elf'.upper()] == Race.ELF


def test_conditions():
    assert Conditions['poisoned'.upper()] == Conditions.POISONED


def test_damage_types():
    assert DamageType['slashing'.upper()] == DamageType.SLASHING