from character import DamageType, CharacterArray, CharacterExtra
from actions import WeaponRange
from to_jax import convert
from dnd_character import CLASSES
from dnd_character.equipment import Item
from constants import ConfigItems, Party

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
    dexterity=16,
    constitution=10,
    intelligence=16,
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


default_config = {
    ConfigItems.PARTY: {
        Party.PC: {'fizban': wizard, 'jimmy': rogue, 'goldmoon': cleric, 'riverwind': fighter},
        Party.NPC: {'raistlin': wizard, 'joffrey': rogue, 'clarion': cleric, 'pikachu': fighter}
    }
}
