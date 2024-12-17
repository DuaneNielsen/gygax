from jax import numpy as jnp

import dice
from dice import RollType, ad_rule


def test_notation():

    num_dice, num_sides, modifier = dice.parse_dice_notation('1d4')
    assert num_dice == 1
    assert num_sides == 4
    assert modifier == 0

    num_dice, num_sides, modifier = dice.parse_dice_notation('1d4+1')
    assert num_dice == 1
    assert num_sides == 4
    assert modifier == 1

    num_dice, num_sides, modifier = dice.parse_dice_notation('2d6')
    assert num_dice == 2
    assert num_sides == 6
    assert modifier == 0

    num_dice, num_sides, modifier = dice.parse_dice_notation('1d20')
    assert num_dice == 1
    assert num_sides == 20
    assert modifier == 0

def test_expected_value():

    expectation = dice.expected_roll('1d4')
    assert expectation == 2.5

    expectation = dice.expected_roll('1d4+1')
    assert expectation == 3.5

    expectation = dice.expected_roll('2d6')
    assert expectation == 7

    expectation = dice.expected_roll('1d20')
    assert expectation == 10.5


def test_ad_rule():
    advantage = jnp.array([RollType.ADVANTAGE, RollType.ADVANTAGE, RollType.ADVANTAGE], dtype=jnp.int8)
    assert ad_rule(advantage) == RollType.ADVANTAGE
    mixed = jnp.array([RollType.ADVANTAGE, RollType.DISADVANTAGE, RollType.ADVANTAGE], dtype=jnp.int8)
    assert ad_rule(mixed) == RollType.NORMAL
    disadvantage = jnp.array([RollType.DISADVANTAGE, RollType.DISADVANTAGE, RollType.DISADVANTAGE], dtype=jnp.int8)
    assert ad_rule(disadvantage) == RollType.DISADVANTAGE
    normal = jnp.array([RollType.NORMAL, RollType.NORMAL, RollType.NORMAL], dtype=jnp.int8)

    stacked = jnp.stack([advantage, disadvantage, mixed, normal])
    stacked_reduced = ad_rule(stacked, axis=-1)
    assert jnp.allclose(stacked_reduced, jnp.array([RollType.ADVANTAGE, RollType.DISADVANTAGE, RollType.NORMAL, RollType.NORMAL]))
