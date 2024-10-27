import constants
from constants import *
from equipment.armor import armor
from equipment.weapons import weapons
from equipment.equipment import empty


fighter = {
    CharacterSheet.ABILITIES: {
        Abilities.STR: 16,
        Abilities.DEX: 12,
        Abilities.CON: 16,
        Abilities.INT: 8,
        Abilities.WIS: 12,
        Abilities.CHA: 8
    },
    CharacterSheet.HITPOINTS: 13,
    CharacterSheet.ARMOR: armor['chain-mail'],
    CharacterSheet.MAIN_HAND: weapons['longsword'],
    CharacterSheet.OFF_HAND: armor['shield'],
    CharacterSheet.RANGED_WEAPON: weapons['crossbow-light']
}

cleric = {
    CharacterSheet.ABILITIES: {
        Abilities.STR: 10,
        Abilities.DEX: 12,
        Abilities.CON: 16,
        Abilities.INT: 10,
        Abilities.WIS: 16,
        Abilities.CHA: 8
    },
    CharacterSheet.HITPOINTS: 11,
    CharacterSheet.ARMOR: armor['chain-mail'],
    CharacterSheet.OFF_HAND: armor['shield'],
    CharacterSheet.MAIN_HAND: weapons['mace'],
    CharacterSheet.RANGED_WEAPON: weapons['crossbow-light']
}

rogue = {
    CharacterSheet.ABILITIES: {
        Abilities.STR: 10,
        Abilities.DEX: 16,
        Abilities.CON: 10,
        Abilities.INT: 16,
        Abilities.WIS: 8,
        Abilities.CHA: 14
    },
    CharacterSheet.HITPOINTS: 8,
    CharacterSheet.ARMOR: armor['leather'],
    CharacterSheet.OFF_HAND: empty,
    CharacterSheet.MAIN_HAND: weapons['rapier'],
    CharacterSheet.RANGED_WEAPON: weapons['shortbow']
}

wizard = {
    CharacterSheet.ABILITIES: {
        Abilities.STR: 10,
        Abilities.DEX: 8,
        Abilities.CON: 16,
        Abilities.INT: 10,
        Abilities.WIS: 16,
        Abilities.CHA: 8
    },
    CharacterSheet.HITPOINTS: 6,
    CharacterSheet.ARMOR: armor['cloth'],
    CharacterSheet.OFF_HAND: empty,
    CharacterSheet.MAIN_HAND: weapons['rapier'],
    CharacterSheet.RANGED_WEAPON: weapons['shortbow']
}

default_config = {
    ConfigItems.PARTY: {
        constants.Party.PC: {'riverwind': fighter, 'goldmoon': cleric, 'jimmy': rogue, 'fizban': wizard},
        constants.Party.NPC: {'pikachu': fighter, 'clarion': cleric, 'joffrey': rogue, 'raistlin': wizard}
    }
}
