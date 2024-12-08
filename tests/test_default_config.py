from character import stack_party
from constants import ConfigItems


fizban, jimmy, goldmoon, riverwind = (0, 0), (0, 1), (0, 2), (0, 3)
raistlin, joffrey, clarion, pikachu = (1, 0), (1, 1), (1, 2), (1, 3)

def test_default_config():
    from default_config import default_config
    names, party = stack_party(default_config[ConfigItems.PARTY])
    assert names[*fizban] == 'fizban'
    assert names[*riverwind] == 'riverwind'
    assert names[*raistlin] == 'raistlin'
    assert names[*pikachu] == 'pikachu'
    assert party.current_hp[0, 0] == 6. + party.ability_modifier.constitution[0, 0]
    assert not party.dead[0, 0]

