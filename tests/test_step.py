from step import init, step, State, action_lookup, Character
from dnd5e import encode_action, ActionTuple, decode_action
from character import CharacterExtra, convert
from dnd_character import CLASSES
from dnd_character.equipment import Item
import jax
import jax.numpy as jnp
import pytest
from copy import deepcopy
from constants import Party


def exp_dmg(state: State, next_state: State, action):
    action = decode_action(action, state.current_player, state.pos)
    return state.character.hp[*action.target] - next_state.character.hp[*action.target]


def test_character():
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
    fighter.off_hand = Item('shield')
    pikachu = convert(fighter, Character)
    assert pikachu.hp.dtype == jnp.float16
    assert jnp.allclose(pikachu.hp, 13)
    assert pikachu.ac == 18
    assert pikachu.prof_bonus == 2
    assert jnp.allclose(pikachu.ability_mods, jnp.array([3, 1, 3, -1, 1, -1]))
    assert jnp.allclose(pikachu.attack_ability_mods, jnp.array([-1, 3, 1, 1]))
    assert jnp.allclose(pikachu.save_bonus, jnp.array([1, 0, 1, 0, 0, 0]))
    assert jnp.allclose(pikachu.damage_type_mul, jnp.ones_like(pikachu.damage_type_mul))

@pytest.fixture
def party():
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

    cleric = CharacterExtra(
        classs=CLASSES["cleric"],
        strength=10,
        dexterity=12,
        constitution=16,
        intelligence=10,
        wisdom=15,
        charisma=8
    )
    cleric.armor = Item('chain-mail')
    cleric.main_hand = Item('mace')
    cleric.off_hand = Item('shield')
    cleric.ranged_two_hand = Item('shortbow')

    rogue = CharacterExtra(
        classs=CLASSES["rogue"],
        strength=10,
        dexterity=18,
        constitution=10,
        intelligence=14,
        wisdom=8,
        charisma=14
    )
    rogue.armor = Item('leather-armor')
    rogue.main_hand = Item('dagger')
    rogue.ranged_two_hand = Item('shortbow')

    wizard = CharacterExtra(
        classs=CLASSES["wizard"],
        strength=10,
        dexterity=8,
        constitution=16,
        intelligence=16,
        wisdom=14,
        charisma=8
    )
    wizard.armor = None
    wizard.main_hand = Item('dagger')
    wizard.ranged_two_hand = Item('shortbow')

    warlock = CharacterExtra(
        classs=CLASSES["warlock"],
        strength=10,
        dexterity=8,
        constitution=16,
        intelligence=10,
        wisdom=14,
        charisma=16
    )
    warlock.armor = Item('leather-armor')
    warlock.main_hand = Item('dagger')
    warlock.ranged_two_hand = None

    return {
        Party.PC: {'wyll': warlock, 'jimmy': rogue, 'goldmoon': cleric, 'riverwind': fighter},
        Party.NPC: {'raistlin': wizard, 'joffrey': rogue, 'clarion': cleric, 'pikachu': fighter}
    }


def test_init(party):
    state = init(party)

    assert state.character.hp.shape == (2, 4)
    assert jnp.allclose(state.character.hp, jnp.float16([
        [11, 8, 11, 13],
        [9, 8, 11, 13],
    ]))


def test_longsword(party):
    state = init(party)
    longsword = action_lookup['longsword']
    action = encode_action(longsword, 3, 1, 1)
    prev_state = deepcopy(state)
    state = step(state, action)
    assert exp_dmg(prev_state, state, action) == 4.5 * 0.5


def test_vmap(party):
    state = init(party)
    rng_init = jax.random.split(jax.random.PRNGKey(0), 2)

    @jax.vmap
    def vmap_init(rng):
        del rng
        return jax.tree.map(lambda x: x.copy(), state)

    state = vmap_init(rng_init)

    longsword = action_lookup['longsword']
    action = encode_action(longsword, 3, 1, 1)
    action = jnp.repeat(action, 2)
    prev_state = deepcopy(state)
    state = jax.vmap(step)(state, action)
    assert jax.vmap(exp_dmg)(prev_state, state, action)[0] == 4.5 * 0.5


def test_eldrich_blast(party):
    state = init(party)
    longsword = action_lookup['eldrich-blast']
    action = encode_action(longsword, 0, 1, 1)
    prev_state = deepcopy(state)
    state = step(state, action)
    assert exp_dmg(prev_state, state, action) == 5.0 * 11/20
