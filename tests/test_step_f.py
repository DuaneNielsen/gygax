import pytest
from actions import ActionsEnum
from step_f import decode_action
from step import init
from to_jax import JaxStringArray
import jax.numpy as jnp

from test_step import party, goldmoon, joffrey, make_action, riverwind, clarion, wyll


def test_decode_action(party):
    state = init(party)
    state, action, (src, tgt), (hit_prob, crit_prob) = make_action(state, party, riverwind, 'longsword', clarion)
    source, weapon, target_mask = decode_action(state, encoded_action=action)

    assert JaxStringArray.uint8_array_to_str(source.name) == 'riverwind'
    assert JaxStringArray.uint8_array_to_str(weapon.name) == 'longsword'
    assert jnp.all(target_mask == jnp.array([
        [0, 0, 0, 0],
        [0, 0, 1, 0]
    ], dtype=jnp.bool))

    state, action, (src, tgt), (hit_prob, crit_prob) = make_action(state, party, clarion, 'longsword', riverwind)
    source, weapon, target_mask = decode_action(state, encoded_action=action)

    assert JaxStringArray.uint8_array_to_str(source.name) == 'clarion'
    assert JaxStringArray.uint8_array_to_str(weapon.name) == 'longsword'
    assert jnp.all(target_mask == jnp.array([
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ], dtype=jnp.bool))

    state, action, (src, tgt), (hit_prob, crit_prob) = make_action(state, party, wyll, 'fireball', clarion)
    source, weapon, target_mask = decode_action(state, encoded_action=action)

    assert JaxStringArray.uint8_array_to_str(source.name) == 'wyll'
    assert JaxStringArray.uint8_array_to_str(weapon.name) == 'fireball'
    assert jnp.all(target_mask == jnp.array([
        [0, 0, 0, 0],
        [1, 1, 1, 1]
    ], dtype=jnp.bool))

    state, action, (src, tgt), (hit_prob, crit_prob) = make_action(state, party, clarion, 'fireball', wyll)
    source, weapon, target_mask = decode_action(state, encoded_action=action)

    assert JaxStringArray.uint8_array_to_str(source.name) == 'clarion'
    assert JaxStringArray.uint8_array_to_str(weapon.name) == 'fireball'
    assert jnp.all(target_mask == jnp.array([
        [1, 1, 1, 1],
        [0, 0, 0, 0]
    ], dtype=jnp.bool))

    state, action, (src, tgt), (hit_prob, crit_prob) = make_action(state, party, clarion, 'burning-hands', wyll)
    source, weapon, target_mask = decode_action(state, encoded_action=action)

    assert JaxStringArray.uint8_array_to_str(source.name) == 'clarion'
    assert JaxStringArray.uint8_array_to_str(weapon.name) == 'burning-hands'
    assert jnp.all(target_mask == jnp.array([
        [0, 0, 1, 1],
        [0, 0, 0, 0]
    ], dtype=jnp.bool))
