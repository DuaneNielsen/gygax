import re
import jax.numpy as jnp
import jax
from enum import IntEnum

def parse_dice_notation(notation):
    pattern = r'(\d+)d(\d+)(?:\s*([-+])\s*(\d+))?'
    match = re.match(pattern, notation)
    if not match:
        raise ValueError(f"Invalid dice notation: {notation}")

    num_dice = int(match.group(1))
    num_sides = int(match.group(2))
    modifier = 0
    if match.group(3) and match.group(4):
        modifier = int(match.group(4)) if match.group(3) == '+' else -int(match.group(4))

    return num_dice, num_sides, modifier


def expected_roll(notation):
    num_dice, num_sides, modifier = parse_dice_notation(notation)
    return num_dice * (num_sides + 1) / 2 + modifier


"""
Here's a probability lookup table for d20 rolls:



Roll Value | 1      | 2      | 3      | 4      | 5      | 6      | 7      | 8      | 9      | 10     | 11     | 12     | 13     | 14     | 15     | 16     | 17     | 18     | 19     | 20
-----------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------
Normal     | 0.0500 | 0.0500 | 0.0500 | 0.0500 | 0.0500 | 0.0500 | 0.0500 | 0.0500 | 0.0500 | 0.0500 | 0.0500 | 0.0500 | 0.0500 | 0.0500 | 0.0500 | 0.0500 | 0.0500 | 0.0500 | 0.0500 | 0.0500
Advantage  | 0.0025 | 0.0075 | 0.0125 | 0.0175 | 0.0225 | 0.0275 | 0.0325 | 0.0375 | 0.0425 | 0.0475 | 0.0525 | 0.0575 | 0.0625 | 0.0675 | 0.0725 | 0.0775 | 0.0825 | 0.0875 | 0.0925 | 0.0975
Disadvant  | 0.0975 | 0.0925 | 0.0875 | 0.0825 | 0.0775 | 0.0725 | 0.0675 | 0.0625 | 0.0575 | 0.0525 | 0.0475 | 0.0425 | 0.0375 | 0.0325 | 0.0275 | 0.0225 | 0.0175 | 0.0125 | 0.0075 | 0.0025

For advantage, each probability follows the formula (2k-1)/400 where k is the roll value.
For disadvantage, each probability follows the formula (41-2k)/400 where k is the roll value.
For normal rolls, each value has exactly 1/20 = 0.05 probability.
"""


"""
For advantage, each probability follows the formula (2k-1)/400 where k is the roll value.
For disadvantage, each probability follows the formula (41-2k)/400 where k is the roll value.
For normal rolls, each value has exactly 1/20 = 0.05 probability.
"""

pdf_d20_table = jnp.stack([
    jnp.ones(20)/20,
    (2 * jnp.arange(1, 21) -1)/400,
    (41 - 2 * jnp.arange(1, 21))/400
])


"""
For normal rolls:
F(k) = k/20   (for k from 1 to 20)
For advantage:

F(k) = P(max(X,Y) ≤ k) = P(both rolls ≤ k) = (k/20)²

For disadvantage:

F(k) = P(min(X,Y) ≤ k) = 1 - P(both rolls > k)
= 1 - ((20-k)/20)²
= 1 - (20-k)²/400
"""

cdf_d20_table = jnp.stack([
    jax.lax.cumsum(jnp.ones(20)/20),
    (jnp.arange(1, 21)/20) ** 2,
    1 - (20 - jnp.arange(1, 21)) ** 2 / 400
])


class RollType(IntEnum):
    NORMAL=0
    ADVANTAGE=1
    DISADVANTAGE=2


def pdf_20(x, roll_type=0):
    return pdf_d20_table[roll_type, x - 1]


def cdf_20(x, roll_type=0):
    return cdf_d20_table[roll_type, x - 1]