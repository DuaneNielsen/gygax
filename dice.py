import re


def parse_dice_notation(notation):
    pattern = r'(\d+)d(\d+)(?:\s*([-+])\s*(\d+))?'
    match = re.match(pattern, notation)
    if not match:
        raise ValueError(f"Invalid dice notation: {notation}")

    num_dice = int(match.group(1))
    num_sides = int(match.group(2))
    modifier = 0
    if match.group(3) and match.group(4):
        modifier = int(match.group(4)) if match.group(3) == '+' else -int(match.group(4))

    return num_dice, num_sides, modifier


def expected_roll(notation):
    num_dice, num_sides, modifier = parse_dice_notation(notation)
    return num_dice * (num_sides + 1) / 2 + modifier