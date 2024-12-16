import constants
from step import hitroll
from step import init, step, State, Actions, Character, action_table
from dice import RollType
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
# jax.config.update('jax_platform_name', 'cpu')


def d_hp_target(state: State, next_state: State, action):
    action = decode_action(action, state.current_player, state.pos, n_actions=len(Actions))
    return state.character.hp[*action.target] - next_state.character.hp[*action.target]

def d_hp_source(state: State, next_state: State, action):
    action = decode_action(action, state.current_player, state.pos, n_actions=len(Actions))
    return state.character.hp[*action.source] - next_state.character.hp[*action.source]


def save_throw_fail(source, target, hitroll_type, ability, dc=None):
    source_dc = 8 + source.prof_bonus + source.attack_ability_mods[hitroll_type]
    dc = source_dc if dc is None else dc
    save_prob = (dc - target.ability_mods[ability] - target.save_bonus[ability] * target.prof_bonus) / 20
    return save_prob, dc


def save_dc_fail(target, ability, dc):
    return (dc - target.ability_mods[ability] - target.save_bonus[ability] * target.prof_bonus) / 20


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


def test_hitroll(party):
    """
    Here's a probability lookup table for d20 rolls:



    Roll Value | 1      | 2      | 3      | 4      | 5      | 6      | 7      | 8      | 9      | 10     | 11     | 12     | 13     | 14     | 15     | 16     | 17     | 18     | 19     | 20
    -----------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------
    Normal     | 0.0500 | 0.0500 | 0.0500 | 0.0500 | 0.0500 | 0.0500 | 0.0500 | 0.0500 | 0.0500 | 0.0500 | 0.0500 | 0.0500 | 0.0500 | 0.0500 | 0.0500 | 0.0500 | 0.0500 | 0.0500 | 0.0500 | 0.0500
    Advantage  | 0.0025 | 0.0075 | 0.0125 | 0.0175 | 0.0225 | 0.0275 | 0.0325 | 0.0375 | 0.0425 | 0.0475 | 0.0525 | 0.0575 | 0.0625 | 0.0675 | 0.0725 | 0.0775 | 0.0825 | 0.0875 | 0.0925 | 0.0975
    Disadvant  | 0.0975 | 0.0925 | 0.0875 | 0.0825 | 0.0775 | 0.0725 | 0.0675 | 0.0625 | 0.0575 | 0.0525 | 0.0475 | 0.0425 | 0.0375 | 0.0325 | 0.0275 | 0.0225 | 0.0175 | 0.0125 | 0.0075 | 0.0025

    """

    state = init(party)
    state, action, (src, tgt), (hit_prob, crit_prob) = make_action(state, party, riverwind, 'longsword', clarion)
    action = decode_action(action, state.current_player, state.pos, n_actions=len(Actions))
    source = jax.tree.map(lambda x: x[*action.source], state.character)
    weapon = jax.tree.map(lambda action_items: action_items[action.action], action_table)
    target: Character = jax.tree.map(lambda x: x[*action.target], state.character)

    hit_chance, _, (hit_prob, crit_prob) = hitroll(source, target, weapon, RollType.NORMAL)
    assert tgt.ac == 17
    assert crit_prob.dtype == jnp.float16
    assert jnp.allclose(crit_prob, 0.05, atol=0.01)
    assert jnp.allclose(hit_prob, 0.4, atol=0.01)

    hit_chance, _, (hit_prob, crit_prob) = hitroll(source, target, weapon, RollType.ADVANTAGE)
    assert crit_prob.dtype == jnp.float16
    assert jnp.allclose(crit_prob, 0.0975, atol=0.01)
    assert jnp.allclose(hit_prob, 0.0575 + 0.0625 + 0.0675 + 0.0725 + 0.0775 + 0.0825 + 0.0875 + 0.0925, atol=0.01)

    hit_chance, _, (hit_prob, crit_prob) = hitroll(source, target, weapon, RollType.DISADVANTAGE)
    assert crit_prob.dtype == jnp.float16
    assert jnp.allclose(crit_prob, 0.0025, atol=0.01)
    assert jnp.allclose(hit_prob, 0.0425 + 0.0375 + 0.0325 + 0.0275 + 0.0225 + 0.0175 + 0.0125 + 0.0075, atol=0.01)



def make_action(state, party, source, action_str, target, roll_type: RollType = RollType.NORMAL):
    source_char = party[source[0]][source[1]]
    target_char = party[target[0]][target[1]]
    source_char_idx = source[1]
    state = state.replace(current_player=source[0])
    target_char_party = 1 if source[0] != target[0] else 0
    target_char_idx = target[1]
    action = Actions[action_str]
    encoded_action = encode_action(action, source_char_idx, target_char_party, target_char_idx, n_actions=len(Actions))
    action = decode_action(encoded_action, state.current_player, state.pos, n_actions=len(Actions))
    source = jax.tree.map(lambda x: x[*source], state.character)
    weapon = jax.tree.map(lambda action_items: action_items[action.action], action_table)
    target: Character = jax.tree.map(lambda x: x[target_char_party, target_char_idx], state.character)
    hit_chance, hit_dmg, _ = hitroll(source, target, weapon, roll_type)
    return state, encoded_action,  (source_char, target_char), (hit_chance, hit_dmg)


def test_longsword(party):
    state = init(party)
    state, action, (source, target), (hit_chance, hit_dmg) = make_action(state, party, riverwind,  'longsword', clarion)
    prev_state = deepcopy(state)
    state = step(state, action)
    exp_damage = (4.5 + source.ability_mods[Abilities.STR]) * hit_dmg
    assert jnp.allclose(d_hp_target(prev_state, state, action), exp_damage, atol=0.01)

    state, action, (source, target), (hit_chance, hit_dmg) = make_action(state, party, riverwind, 'longsword-two-hand', clarion)
    prev_state = deepcopy(state)
    state = step(state, action)
    exp_damage = (5.5 + source.ability_mods[Abilities.STR]) * hit_dmg
    assert jnp.allclose(d_hp_target(prev_state, state, action), exp_damage, atol=0.01)


def test_rapier(party):
    state = init(party)
    state, action, (source, target), (hit_chance, hit_dmg) = make_action(state, party, jimmy,  'rapier', clarion)
    prev_state = deepcopy(state)
    state = step(state, action)
    exp_damage = (4.5 + source.ability_mods[Abilities.STR]) * hit_dmg
    assert jnp.allclose(d_hp_target(prev_state, state, action), exp_damage, atol=0.01)

    state, action, (source, target), (hit_chance, hit_dmg) = make_action(state, party, jimmy,  'rapier-finesse', clarion)
    prev_state = deepcopy(state)
    state = step(state, action)
    exp_damage = (4.5 + source.ability_mods[Abilities.DEX]) * hit_dmg
    assert jnp.allclose(d_hp_target(prev_state, state, action), exp_damage, atol=0.01)


def test_eldrich_blast(party):
    state = init(party)
    state, action, (source, target), (hit_chance, hit_dmg) = make_action(state, party, wyll,  'eldrich-blast', clarion)
    prev_state = deepcopy(state)
    state = step(state, action)
    exp_damage = 5.5 * hit_dmg
    assert jnp.allclose(d_hp_target(prev_state, state, action), exp_damage, atol=0.01)

    state, action, (source, target), (hit_chance, hit_dmg) = make_action(state, party, wyll,  'agonizing-blast', clarion)
    prev_state = deepcopy(state)
    state = step(state, action)
    exp_damage = (5.5 + source.ability_mods[Abilities.CHA]) * hit_dmg
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

    state0, longsword, (source, target), (hit_chance, hitroll_sword) = make_action(select(state, 0), party, riverwind,  'longsword', clarion)

    state1, blast, (source, target), (hit_chance, hitroll_blast) = make_action(select(state, 1), party, riverwind,  'eldrich-blast', clarion)

    action = jnp.array([longsword, blast])
    state = jax.tree.map(lambda *x: jnp.stack(x), state0, state1)
    prev_state = deepcopy(state)
    state = jax.vmap(step)(state, action)
    dmg = jax.vmap(d_hp_target)(prev_state, state, action)

    assert jnp.allclose(dmg[0], (4.5 + source.ability_mods[Abilities.STR]) * hitroll_sword, atol=0.01)
    assert jnp.allclose(dmg[1],  5.5 * hitroll_blast, atol=0.01)


def test_longbow(party):
    state = init(party)
    state, action, (source, target), (hit_chance, hit_dmg) = make_action(state, party, riverwind,  'longbow', clarion)
    prev_state = deepcopy(state)
    state = step(state, action)
    dmg = d_hp_target(prev_state, state, action)
    exp_dmg = hit_dmg * (4.5 + source.ability_mods[Abilities.DEX])
    assert jnp.allclose(dmg, exp_dmg, atol=0.01)


def test_poison_spray(party):
    state = init(party)
    state, action, (source, target), (hit_chance, hit_dmg) = make_action(state, party, wyll, 'poison-spray', joffrey)
    prev_state = deepcopy(state)
    state = step(state, action)
    assert d_hp_target(prev_state, state, action) == 6.5 * (8 + 2 + 3 + 1) / 20

    state = init(party)
    state, action, (source, target), (hit_chance, hit_dmg) = make_action(state, party, wyll, 'poison-spray', pikachu)
    prev_state = deepcopy(state)
    state = step(state, action)
    assert jnp.allclose(d_hp_target(prev_state, state, action), 6.5 * (8 + 2 + 3 - 3 - 2) / 20, atol=0.01)


def test_burning_hands(party):
    state = init(party)
    burning_hands = Actions['burning-hands']
    state, action, (source, target), (hit_chance, hit_dmg) = make_action(state, party, wyll, 'burning-hands', joffrey)
    prev_state = deepcopy(state)
    state = step(state, action)
    save_fail_prob = (8 + 2 + 3 - 3 - 3) / 20
    exp_damage = (save_fail_prob + (1-save_fail_prob) * 0.5) * 3.5 * 3
    assert jnp.allclose(d_hp_target(prev_state, state, action), exp_damage, atol=0.01)


def test_acid_arrow(party):

    state = init(party)
    state, action, (source, target), (hit_chance, hit_dmg) = make_action(state, party, raistlin, 'acid-arrow', goldmoon)
    prev_state = deepcopy(state)
    state = step(state, action)
    exp_damage = hit_dmg * 4 * 2.5
    assert jnp.allclose(d_hp_target(prev_state, state, action), exp_damage, atol=0.01)
    effect_name = JaxStringArray.uint8_array_to_str(state.character.effects.name[*goldmoon, 0])
    assert effect_name == 'acid-arrow'
    assert state.character.effect_active[*goldmoon, 0]
    exp_dmg = 2.5 * 2 * hit_dmg
    assert jnp.allclose(state.character.effects.recurring_damage[*goldmoon, 0], exp_dmg, atol=0.01)

    state, action, (source, target), (hit_chance, hit_dmg) = make_action(state, party, goldmoon, 'end-turn', goldmoon)
    prev_state = deepcopy(state)
    state = step(state, action)
    assert jnp.allclose(d_hp_source(prev_state, state, action), exp_dmg, atol=0.01)
    assert state.character.effects.duration[*goldmoon, 0] == 0
    assert not state.character.effect_active[*goldmoon, 0]

    state, action, (source, target), (hit_chance, hit_dmg) = make_action(state, party, goldmoon, 'end-turn', goldmoon)
    prev_state = deepcopy(state)
    state = step(state, action)
    assert jnp.allclose(d_hp_source(prev_state, state, action), 0., atol=0.01)
    assert state.character.effects.duration[*goldmoon, 0] < 1
    assert not state.character.effect_active[*goldmoon, 0]


def test_hold_person(party):
    state = init(party)
    state, action, (source, target), (hit_chance, hit_dmg) = make_action(state, party, goldmoon, 'hold-person', joffrey)
    prev_state = deepcopy(state)
    state = step(state, action)
    save_fail_prob, dc = save_throw_fail(source, target, constants.HitrollType.SPELL, Abilities.WIS)

    # expectation is joffrey will fail his saving throw 0.75
    assert state.character.conditions.shape == (2, 4, len(constants.Conditions))
    assert jnp.allclose(d_hp_target(prev_state, state, action), 0, atol=0.01)
    assert state.character.conditions[*joffrey, constants.Conditions.PARALYZED]
    assert JaxStringArray.uint8_array_to_str(state.character.effects.name[1, 1, 0]) == 'hold-person'
    assert state.character.effect_active[1, 1, 0]
    assert state.character.effects.cum_save[1, 1, 0] == save_fail_prob
    assert state.character.effects.save_dc[1, 1, 0] == dc

    # second attempt at save will also fail in the expectation (0.567 chance of fail)
    state, action, (source, target), (hit_chance, hit_dmg) = make_action(state, party, joffrey, 'end-turn', joffrey)
    prev_state = deepcopy(state)
    state = step(state, action)
    assert jnp.allclose(d_hp_target(prev_state, state, action), 0, atol=0.01)
    assert state.character.conditions[1, 1, constants.Conditions.PARALYZED]
    assert JaxStringArray.uint8_array_to_str(state.character.effects.name[1, 1, 0]) == 'hold-person'
    assert state.character.effect_active[1, 1, 0]
    assert state.character.effects.cum_save[1, 1, 0] == save_fail_prob ** 2

    # third will succeed (0.42 chance of fail)
    state, action, (source, target), (hit_chance, hit_dmg) = make_action(state, party, joffrey, 'end-turn', joffrey)
    prev_state = deepcopy(state)
    state = step(state, action)
    assert jnp.allclose(d_hp_target(prev_state, state, action), 0, atol=0.01)
    assert not state.character.conditions[1, 1, constants.Conditions.PARALYZED]
    assert JaxStringArray.uint8_array_to_str(state.character.effects.name[1, 1, 0]) == 'hold-person'
    assert not state.character.effect_active[1, 1, 0]
    assert state.character.effects.cum_save[1, 1, 0] == 0.


def test_concentration(party):

    # cast hold person on joffrey
    state = init(party)
    state, action, (source, target), (hit_chance, hit_dmg) = make_action(state, party, goldmoon, 'hold-person', joffrey)
    state = step(state, action)
    assert state.character.conditions[*joffrey, constants.Conditions.PARALYZED]
    assert state.character.concentrating[*goldmoon].any()

    # cast on pikachu, hold on joffrey should end
    state, action, (source, target), (hit_chance, hit_dmg) = make_action(state, party, goldmoon, 'hold-person', pikachu)
    state = step(state, action)
    assert not state.character.conditions[*joffrey, constants.Conditions.PARALYZED]
    assert state.character.conditions[*pikachu, constants.Conditions.PARALYZED]
    assert state.character.concentrating[*goldmoon].any()

    # using an action that does not require concentration should not break concentration
    state, action, (source, target), (hit_chance, hit_dmg) = make_action(state, party, goldmoon, 'longsword', pikachu)
    state = step(state, action)
    assert not state.character.conditions[*joffrey, constants.Conditions.PARALYZED]
    assert state.character.conditions[*pikachu, constants.Conditions.PARALYZED]
    assert state.character.concentrating[*goldmoon].any()

    for t in range(1, 5):
        # taking damage should force a concentration check, first succeeds
        state, action, (source, target), (hit_chance, hit_dmg) = make_action(state, party, joffrey, 'shortbow', goldmoon)
        state = step(state, action)
        concentration_succ_prob = 1 - hit_chance * save_dc_fail(target, Abilities.CON, 10)
        assert not state.character.conditions[*joffrey, constants.Conditions.PARALYZED]
        assert state.character.conditions[*pikachu, constants.Conditions.PARALYZED] == (concentration_succ_prob ** t > 0.5)
        assert state.character.concentrating[*goldmoon].any() == (concentration_succ_prob ** t > 0.5)
        assert jnp.allclose(state.character.concentration_check_cum[*goldmoon],  concentration_succ_prob ** t, atol=0.01)


def test_concentration_from_spell_effects(party):

    state = init(party)
    state, action, (source, target), (hit_chance, hit_dmg) = make_action(state, party, goldmoon, 'hold-person', pikachu)
    state = step(state, action)
    assert state.character.conditions[*pikachu, constants.Conditions.PARALYZED]
    assert state.character.concentrating[*goldmoon].any()

    state, action, (source, target), (hit_chance, hit_dmg) = make_action(state, party, raistlin, 'acid-arrow', goldmoon)
    state = step(state, action)
    concentration_succ_prob = 1 - hit_chance * save_dc_fail(target, Abilities.CON, 10)
    assert state.character.conditions[*pikachu, constants.Conditions.PARALYZED] == (concentration_succ_prob ** 1 > 0.5)
    assert state.character.concentrating[*goldmoon].any() == (concentration_succ_prob ** 1 > 0.5)
    assert jnp.allclose(state.character.concentration_check_cum[*goldmoon], concentration_succ_prob ** 1, atol=0.01)

    state, action, (source, target), (hit_chance, hit_dmg) = make_action(state, party, raistlin, 'acid-arrow', goldmoon)
    state = step(state, action)
    concentration_succ_prob = 1 - hit_chance * save_dc_fail(target, Abilities.CON, 10)
    assert state.character.conditions[*pikachu, constants.Conditions.PARALYZED] == (concentration_succ_prob ** 2 > 0.5)
    assert state.character.concentrating[*goldmoon].any() == (concentration_succ_prob ** 2 > 0.5)
    assert jnp.allclose(state.character.concentration_check_cum[*goldmoon], concentration_succ_prob ** 2, atol=0.01)

    state, action, (source, target), (hit_chance, hit_dmg) = make_action(state, party, raistlin, 'acid-arrow', goldmoon)
    state = step(state, action)
    concentration_succ_prob = 1 - hit_chance * save_dc_fail(target, Abilities.CON, 10)
    assert state.character.conditions[*pikachu, constants.Conditions.PARALYZED] == (concentration_succ_prob ** 3 > 0.5)
    assert state.character.concentrating[*goldmoon].any() == (concentration_succ_prob ** 3 > 0.5)
    assert jnp.allclose(state.character.concentration_check_cum[*goldmoon], concentration_succ_prob ** 3, atol=0.01)

    # three acid arrow spell effects should hit at once, breaking concentration
    state, action, (source, target), (hit_chance, hit_dmg) = make_action(state, party, goldmoon, 'end-turn', goldmoon)
    state = step(state, action)
    assert state.character.conditions[*pikachu, constants.Conditions.PARALYZED] == (concentration_succ_prob ** 6 > 0.5)
    assert state.character.concentrating[*goldmoon].any() == (concentration_succ_prob ** 6 > 0.5)
    assert jnp.allclose(state.character.concentration_check_cum[*goldmoon], concentration_succ_prob ** 6, atol=0.01)

    # recast the spell, concentration check should reset
    state = init(party)
    state, action, (source, target), (hit_chance, hit_dmg) = make_action(state, party, goldmoon, 'hold-person', pikachu)
    state = step(state, action)
    assert state.character.conditions[*pikachu, constants.Conditions.PARALYZED]
    assert state.character.concentrating[*goldmoon].any()
    assert jnp.allclose(state.character.concentration_check_cum[*goldmoon], 1., atol=0.01)
