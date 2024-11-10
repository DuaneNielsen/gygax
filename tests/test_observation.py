from dnd5e import _observe_party, configure_party, convert_equipment, _observe_actions
from equipment.weapons import weapons
from default_config import default_config
from pytest import fixture
from constants import *
import jax

fizban, jimmy, goldmoon, riverwind = (0, 0), (0, 1), (0, 2), (0, 3)
raistlin, joffrey, clarion, pikachu = (1, 0), (1, 1), (1, 2), (1, 3)

@fixture
def party():
    return configure_party(default_config)


def test_observe_party(party):
    observed_party = _observe_party(party)
    assert observed_party.hitpoints[*fizban].sum(-1) == 6. - HP_LOWER
    assert observed_party.actions.damage[*fizban, Actions.ATTACK_RANGED_WEAPON].sum(-1) == 3.

    assert observed_party.hitpoints[*pikachu].sum(-1) == 13. - HP_LOWER
    assert observed_party.actions.damage[*pikachu, Actions.ATTACK_RANGED_WEAPON].sum(-1) == 4


def test_observe_action():
    action = convert_equipment(weapons['longsword'])
    observed_action = _observe_actions(action)
    assert observed_action.damage.sum(-1) == 4
    assert observed_action.damage_type[DamageType.SLASHING] == 1.
    assert observed_action.legal_target_pos.dtype == jnp.float32
