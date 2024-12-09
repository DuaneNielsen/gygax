import constants
from step import init, step, State, Actions, Character
from dnd5e import encode_action, ActionTuple, decode_action
from character import CharacterExtra, convert, JaxStringArray
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
        name='pikachu',
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
    assert JaxStringArray.uint8_array_to_str(pikachu.name) == 'pikachu'
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
    riverwind = CharacterExtra(
        name='riverwind',
        classs=CLASSES["fighter"],
        strength=16,
        dexterity=12,
        constitution=16,
        intelligence=8,
        wisdom=12,
        charisma=8
    )
    riverwind.armor = Item('chain-mail')
    riverwind.main_hand = Item('longsword')
    riverwind.off_hand = Item('shield')
    riverwind.ranged_two_hand = Item('shortbow')

    pikachu = CharacterExtra(
        name='pikachu',
        classs=CLASSES["fighter"],
        strength=16,
        dexterity=12,
        constitution=16,
        intelligence=8,
        wisdom=12,
        charisma=8
    )
    pikachu.armor = Item('chain-mail')
    pikachu.main_hand = Item('longsword')
    pikachu.off_hand = Item('shield')
    pikachu.ranged_two_hand = Item('shortbow')

    goldmoon = CharacterExtra(
        name='goldmoon',
        classs=CLASSES["cleric"],
        strength=10,
        dexterity=12,
        constitution=16,
        intelligence=10,
        wisdom=16,
        charisma=10
    )
    goldmoon.armor = Item('chain-mail')
    goldmoon.main_hand = Item('mace')
    goldmoon.off_hand = Item('shield')
    goldmoon.ranged_two_hand = Item('shortbow')

    clarion = CharacterExtra(
        name='clarion',
        classs=CLASSES["cleric"],
        strength=10,
        dexterity=12,
        constitution=16,
        intelligence=10,
        wisdom=15,
        charisma=8
    )
    clarion.armor = Item('chain-mail')
    clarion.main_hand = Item('mace')
    clarion.off_hand = Item('shield')
    clarion.ranged_two_hand = Item('shortbow')

    jimmy = CharacterExtra(
        name='jimmy',
        classs=CLASSES["rogue"],
        strength=10,
        dexterity=18,
        constitution=10,
        intelligence=14,
        wisdom=8,
        charisma=14
    )
    jimmy.armor = Item('leather-armor')
    jimmy.main_hand = Item('dagger')
    jimmy.ranged_two_hand = Item('shortbow')

    joffrey = CharacterExtra(
        name='joffrey',
        classs=CLASSES["rogue"],
        strength=10,
        dexterity=18,
        constitution=8,
        intelligence=14,
        wisdom=8,
        charisma=14
    )
    joffrey.armor = Item('leather-armor')
    joffrey.main_hand = Item('dagger')
    joffrey.ranged_two_hand = Item('shortbow')

    raistlin = CharacterExtra(
        name='raistlin',
        classs=CLASSES["wizard"],
        strength=10,
        dexterity=8,
        constitution=16,
        intelligence=16,
        wisdom=14,
        charisma=8
    )
    raistlin.armor = None
    raistlin.main_hand = Item('dagger')
    raistlin.ranged_two_hand = Item('shortbow')

    wyll = CharacterExtra(
        name='wyll',
        classs=CLASSES["warlock"],
        strength=10,
        dexterity=8,
        constitution=16,
        intelligence=10,
        wisdom=14,
        charisma=16
    )
    wyll.armor = Item('leather-armor')
    wyll.main_hand = Item('dagger')
    wyll.ranged_two_hand = None

    return {
        Party.PC: [wyll, jimmy, goldmoon, riverwind],
        Party.NPC: [raistlin, joffrey, clarion, pikachu]
    }


def test_init(party):
    state = init(party)

    assert state.character.hp.shape == (2, 4)
    assert jnp.allclose(state.character.hp, jnp.float16([
        [11, 8, 11, 13],
        [9, 7, 11, 13],
    ]))
    name = JaxStringArray.uint8_array_to_str(state.character.name[0, 0])
    assert name == 'wyll'


def test_longsword(party):
    state = init(party)
    longsword = Actions['longsword']
    action = encode_action(longsword, 3, 1, 1)
    prev_state = deepcopy(state)
    state = step(state, action)
    assert exp_dmg(prev_state, state, action) == (4.5 + 3) * 0.5


def test_eldrich_blast(party):
    state = init(party)
    blast = Actions['eldrich-blast']
    action = encode_action(blast, 0, 1, 1)
    prev_state = deepcopy(state)
    state = step(state, action)
    assert exp_dmg(prev_state, state, action) == 5.5 * 10/20


def test_vmap(party):
    state = init(party)
    rng_init = jax.random.split(jax.random.PRNGKey(0), 2)

    @jax.vmap
    def vmap_init(rng):
        del rng
        return jax.tree.map(lambda x: x.copy(), state)

    state = vmap_init(rng_init)

    longsword = encode_action(Actions['longsword'], 3, 1, 1)
    blast = encode_action(Actions['eldrich-blast'], 0, 1, 1)
    action = jnp.array([longsword, blast])
    prev_state = deepcopy(state)
    state = jax.vmap(step)(state, action)
    assert jax.vmap(exp_dmg)(prev_state, state, action)[0] == 4.5 * 0.5
    assert jax.vmap(exp_dmg)(prev_state, state, action)[1] == 5.5 * 10/20


def test_longbow(party):
    state = init(party)
    longsword = Actions['longbow']
    action = encode_action(longsword, 3, 1, 1)
    prev_state = deepcopy(state)
    state = step(state, action)
    assert exp_dmg(prev_state, state, action) == (4.5 + 1) * 8/20


def test_poison_spray(party):
    state = init(party)
    spray = Actions['poison-spray']
    action = encode_action(spray, 0, 1, 1)
    prev_state = deepcopy(state)
    state = step(state, action)
    assert exp_dmg(prev_state, state, action) == 6.5 * (8+2+3+1)/20

    state = init(party)
    spray = Actions['poison-spray']
    action = encode_action(spray, 0, 1, 3)
    prev_state = deepcopy(state)
    state = step(state, action)
    assert jnp.allclose(exp_dmg(prev_state, state, action), 6.5 * (8 + 2 + 3 - 3 - 2) / 20, atol=0.01)


def test_burning_hands(party):
    state = init(party)
    burning_hands = Actions['burning-hands']
    action = encode_action(burning_hands, 0, 1, 1)
    prev_state = deepcopy(state)
    state = step(state, action)
    save_fail_prob = (8 + 2 + 3 - 3 - 3) / 20
    exp_damage = (save_fail_prob + (1-save_fail_prob) * 0.5) * 3.5 * 3
    assert jnp.allclose(exp_dmg(prev_state, state, action), exp_damage, atol=0.01)

def test_hold_person(party):
    state = init(party)
    hold_person = Actions['hold-person'].value
    action = encode_action(hold_person, 2, 1, 1, n_actions=len(Actions))
    prev_state = deepcopy(state)
    state = step(state, action)
    save_fail_prob = (8 + 2 + 3 + 1) / 20
    exp_damage = 0
    assert jnp.allclose(exp_dmg(prev_state, state, action), exp_damage, atol=0.01)
    assert state.character.conditions[1, 1, constants.Conditions.PARALYZED] == save_fail_prob
