import constants
from step import init, step, State, Actions, Character
from dnd5e import encode_action, ActionTuple, decode_action
from character import CharacterExtra, convert, JaxStringArray
from constants import Abilities
from dnd_character import CLASSES
from dnd_character.equipment import Item
import jax
import jax.numpy as jnp
import pytest
from copy import deepcopy
from constants import Party


def d_hp_target(state: State, next_state: State, action):
    action = decode_action(action, state.current_player, state.pos, n_actions=len(Actions))
    return state.character.hp[*action.target] - next_state.character.hp[*action.target]

def d_hp_source(state: State, next_state: State, action):
    action = decode_action(action, state.current_player, state.pos, n_actions=len(Actions))
    return state.character.hp[*action.source] - next_state.character.hp[*action.source]


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
        wisdom=18,
        charisma=10
    )
    goldmoon.armor = Item('scale-mail')
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
    clarion.armor = Item('scale-mail')
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
        intelligence=18,
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

wyll = 0, 0
jimmy = 0, 1
goldmoon = 0, 2
riverwind = 0, 3
raistlin= 1, 0
joffrey=1, 1
clarion=1, 2
pikachu=1, 3


def test_init(party):
    state = init(party)

    assert state.character.hp.shape == (2, 4)
    assert jnp.allclose(state.character.hp, jnp.float16([
        [11, 8, 11, 13],
        [9, 7, 11, 13],
    ]))
    name = JaxStringArray.uint8_array_to_str(state.character.name[0, 0])
    assert name == 'wyll'
    assert jnp.any(state.character.effect_active) == False
    assert state.character.concentrating.shape == (constants.N_PLAYERS, constants.N_CHARACTERS, constants.MAX_TARGETS)
    assert state.character.concentration_ref.shape == (constants.N_PLAYERS, constants.N_CHARACTERS, constants.MAX_TARGETS, 3)


def make_action(state, party, source, action_str, target):
    source_char = party[source[0]][source[1]]
    target_char = party[target[0]][target[1]]
    action = Actions[action_str]
    current_party = source[0]

    if current_party == state.current_player:
        encoded_action = encode_action(action, source[1], target[0], target[1], n_actions=len(Actions))
    else:
        state = state.replace(current_player=(state.current_player + 1) % 2)
        target_party = (target[0] + state.current_player) % 2
        encoded_action = encode_action(action, source[1], target_party, target[1], n_actions=len(Actions))
    return state, encoded_action,  (source_char, target_char)


def test_longsword(party):
    state = init(party)
    state, action, (source, target) = make_action(state, party, riverwind,  'longsword', clarion)
    prev_state = deepcopy(state)
    state = step(state, action)
    hitroll = (20 - target.ac + source.prof_bonus + source.ability_mods[Abilities.STR]) / 20
    assert d_hp_target(prev_state, state, action) == (4.5 + source.ability_mods[Abilities.STR]) * hitroll


def test_eldrich_blast(party):
    state = init(party)
    state, action, (source, target) = make_action(state, party, wyll,  'eldrich-blast', clarion)
    prev_state = deepcopy(state)
    state = step(state, action)
    hitroll = (20 - target.ac + source.prof_bonus + source.ability_mods[Abilities.CHA]) / 20
    exp_damage = 5.5 * hitroll
    assert jnp.allclose(d_hp_target(prev_state, state, action), exp_damage, atol=0.01)


def test_vmap(party):
    state = init(party)
    rng_init = jax.random.split(jax.random.PRNGKey(0), 2)

    @jax.vmap
    def vmap_init(rng):
        del rng
        return jax.tree.map(lambda x: x.copy(), state)

    state = vmap_init(rng_init)

    def select(state, index):
        return jax.tree.map(lambda s: s[index], state)

    state0, longsword, (source, target) = make_action(select(state, 0), party, riverwind,  'longsword', clarion)
    hitroll_sword = (20 - target.ac + source.prof_bonus + source.ability_mods[Abilities.STR]) / 20

    state1, blast, (source, target) = make_action(select(state, 1), party, riverwind,  'eldrich-blast', clarion)
    hitroll_blast = (20 - target.ac + source.prof_bonus + source.ability_mods[Abilities.CHA]) / 20

    action = jnp.array([longsword, blast])
    state = jax.tree.map(lambda *x: jnp.stack(x), state0, state1)
    prev_state = deepcopy(state)
    state = jax.vmap(step)(state, action)
    dmg = jax.vmap(d_hp_target)(prev_state, state, action)

    assert jnp.allclose(dmg[0], (4.5 + source.ability_mods[Abilities.STR]) * hitroll_sword, atol=0.01)
    assert jnp.allclose(dmg[1],  5.5 * hitroll_blast, atol=0.01)


def test_longbow(party):
    state = init(party)
    state, action, (source, target) = make_action(state, party, riverwind,  'longbow', clarion)
    prev_state = deepcopy(state)
    hitroll = (20 - target.ac + source.prof_bonus + source.ability_mods[Abilities.DEX]) / 20
    state = step(state, action)
    dmg = d_hp_target(prev_state, state, action)
    exp_dmg = hitroll * (4.5 + source.ability_mods[Abilities.DEX])
    assert jnp.allclose(dmg, exp_dmg, atol=0.01)


def test_poison_spray(party):
    state = init(party)
    spray = Actions['poison-spray']
    action = encode_action(spray, 0, 1, 1, n_actions=len(Actions))
    prev_state = deepcopy(state)
    state = step(state, action)
    assert d_hp_target(prev_state, state, action) == 6.5 * (8 + 2 + 3 + 1) / 20

    state = init(party)
    spray = Actions['poison-spray']
    action = encode_action(spray, 0, 1, 3)
    prev_state = deepcopy(state)
    state = step(state, action)
    assert jnp.allclose(d_hp_target(prev_state, state, action), 6.5 * (8 + 2 + 3 - 3 - 2) / 20, atol=0.01)


def test_burning_hands(party):
    state = init(party)
    burning_hands = Actions['burning-hands']
    action = encode_action(burning_hands, 0, 1, 1, n_actions=len(Actions))
    prev_state = deepcopy(state)
    state = step(state, action)
    save_fail_prob = (8 + 2 + 3 - 3 - 3) / 20
    exp_damage = (save_fail_prob + (1-save_fail_prob) * 0.5) * 3.5 * 3
    assert jnp.allclose(d_hp_target(prev_state, state, action), exp_damage, atol=0.01)


def test_acid_arrow(party):

    state = init(party)
    state, action, (source, target) = make_action(state, party, raistlin, 'acid-arrow', goldmoon)
    prev_state = deepcopy(state)
    state = step(state, action)
    hit_roll = 1 - ((target.ac - 4 - 2) / 20)
    exp_damage = hit_roll * 4 * 2.5
    assert jnp.allclose(d_hp_target(prev_state, state, action), exp_damage, atol=0.01)
    effect_name = JaxStringArray.uint8_array_to_str(state.character.effects.name[*goldmoon, 0])
    assert effect_name == 'acid-arrow'
    assert state.character.effect_active[*goldmoon, 0]
    exp_dmg = 2.5 * 2 * hit_roll
    assert jnp.allclose(state.character.effects.recurring_damage[*goldmoon, 0], exp_dmg, atol=0.01)

    state, action, (source, target) = make_action(state, party, goldmoon, 'end-turn', goldmoon)
    prev_state = deepcopy(state)
    state = step(state, action)
    assert jnp.allclose(d_hp_source(prev_state, state, action), exp_dmg, atol=0.01)
    assert state.character.effects.duration[*goldmoon, 0] == 0
    assert not state.character.effect_active[*goldmoon, 0]

    state, action, (source, target) = make_action(state, party, goldmoon, 'end-turn', goldmoon)
    prev_state = deepcopy(state)
    state = step(state, action)
    assert jnp.allclose(d_hp_source(prev_state, state, action), 0., atol=0.01)
    assert state.character.effects.duration[*goldmoon, 0] < 1
    assert not state.character.effect_active[*goldmoon, 0]


def save(source, target, hitroll_type, ability):
    dc = 8 + source.prof_bonus + source.attack_ability_mods[hitroll_type]
    save_prob = (dc - target.ability_mods[ability] - target.save_bonus[ability] * target.prof_bonus) / 20
    return save_prob, dc


def test_hold_person(party):
    state = init(party)
    state, action, (source, target) = make_action(state, party, goldmoon, 'hold-person', joffrey)
    prev_state = deepcopy(state)
    state = step(state, action)
    save_fail_prob, dc = save(source, target, constants.HitrollType.SPELL, Abilities.WIS)

    # expectation is joffrey will fail his saving throw 0.75
    assert state.character.conditions.shape == (2, 4, len(constants.Conditions))
    assert jnp.allclose(d_hp_target(prev_state, state, action), 0, atol=0.01)
    assert state.character.conditions[*joffrey, constants.Conditions.PARALYZED]
    assert JaxStringArray.uint8_array_to_str(state.character.effects.name[1, 1, 0]) == 'hold-person'
    assert state.character.effect_active[1, 1, 0]
    assert state.character.effects.cum_save[1, 1, 0] == save_fail_prob
    assert state.character.effects.save_dc[1, 1, 0] == dc

    # second attempt at save will also fail in the expectation (0.567 chance of fail)
    state, action, (source, target) = make_action(state, party, joffrey, 'end-turn', joffrey)
    prev_state = deepcopy(state)
    state = step(state, action)
    assert jnp.allclose(d_hp_target(prev_state, state, action), 0, atol=0.01)
    assert state.character.conditions[1, 1, constants.Conditions.PARALYZED]
    assert JaxStringArray.uint8_array_to_str(state.character.effects.name[1, 1, 0]) == 'hold-person'
    assert state.character.effect_active[1, 1, 0]
    assert state.character.effects.cum_save[1, 1, 0] == save_fail_prob ** 2

    # third will succeed (0.42 chance of fail)
    state, action, (source, target) = make_action(state, party, joffrey, 'end-turn', joffrey)
    prev_state = deepcopy(state)
    state = step(state, action)
    assert jnp.allclose(d_hp_target(prev_state, state, action), 0, atol=0.01)
    assert not state.character.conditions[1, 1, constants.Conditions.PARALYZED]
    assert JaxStringArray.uint8_array_to_str(state.character.effects.name[1, 1, 0]) == 'hold-person'
    assert not state.character.effect_active[1, 1, 0]
    assert state.character.effects.cum_save[1, 1, 0] == 0.


def test_concentration(party):
    state = init(party)
    state, action, (source, target) = make_action(state, party, goldmoon, 'hold-person', joffrey)
    prev_state = deepcopy(state)
    state = step(state, action)
    save_fail_prob, dc = save(source, target, constants.HitrollType.SPELL, Abilities.WIS)
    assert state.character.conditions[*joffrey, constants.Conditions.PARALYZED]

    state, action, (source, target) = make_action(state, party, goldmoon, 'hold-person', pikachu)
    prev_state = deepcopy(state)
    state = step(state, action)
    save_fail_prob, dc = save(source, target, constants.HitrollType.SPELL, Abilities.WIS)
    assert not state.character.conditions[*joffrey, constants.Conditions.PARALYZED]
    assert state.character.conditions[*pikachu, constants.Conditions.PARALYZED]
