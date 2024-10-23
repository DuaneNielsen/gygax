import constants
from constants import *

fighter = {
    CharacterStats.ABILITIES: {
        Abilities.STR: 16,
        Abilities.DEX: 12,
        Abilities.CON: 16,
        Abilities.INT: 8,
        Abilities.WIS: 12,
        Abilities.CHA: 8
    },
    CharacterStats.HITPOINTS: 13
}

cleric = {
    CharacterStats.ABILITIES: {
        Abilities.STR: 10,
        Abilities.DEX: 12,
        Abilities.CON: 16,
        Abilities.INT: 10,
        Abilities.WIS: 16,
        Abilities.CHA: 8
    },
    CharacterStats.HITPOINTS: 11
}

rogue = {
    CharacterStats.ABILITIES: {
        Abilities.STR: 10,
        Abilities.DEX: 16,
        Abilities.CON: 10,
        Abilities.INT: 16,
        Abilities.WIS: 8,
        Abilities.CHA: 14
    },
    CharacterStats.HITPOINTS: 8
}

wizard = {
    CharacterStats.ABILITIES: {
        Abilities.STR: 10,
        Abilities.DEX: 8,
        Abilities.CON: 16,
        Abilities.INT: 10,
        Abilities.WIS: 16,
        Abilities.CHA: 8
    },
    CharacterStats.HITPOINTS: 6
}

default_config = {
    ConfigItems.PARTY: {
        constants.Party.PC: {'riverwind': fighter, 'goldmoon': cleric, 'johnny': rogue, 'fizban': wizard},
        constants.Party.NPC: {'pikachu': fighter, 'clarion': cleric, 'joffrey': rogue, 'raistlin': wizard}
    }
}
