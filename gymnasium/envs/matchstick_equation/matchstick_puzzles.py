import os
import json
import re
import itertools
from PIL import Image, ImageDraw
import math
# =============================================================================
# 1. Universal Symbol Representation
# =============================================================================
# We represent digits and operators as sets of “matchstick” segments.
# Digits (using a 7-seg style) use segments 0–6.
# Operators use additional segments.
SYMBOL_TO_SEGMENTS = {
    # Digits:
    '0': {0, 1, 2, 3, 4, 5},
    '1': {1, 2},
    '2': {0, 1, 6, 4, 3},
    '3': {0, 1, 6, 2, 3},
    '4': {5, 6, 1, 2},
    '5': {0, 5, 6, 2, 3},
    '6': {0, 5, 6, 4, 2, 3},
    '7': {0, 1, 2},
    '8': {0, 1, 2, 3, 4, 5, 6},
    '9': {0, 1, 2, 3, 5, 6},
    # Operators:
    '-': {6},         # minus: one horizontal match (middle)
    '+': {6, 7},      # plus: horizontal (6) plus extra vertical (7)
    '*': {8, 9},      # multiply: two diagonal matches
    '/': {9},        # divide: one diagonal match
    '=': {11, 12},    # equal: two horizontal matches
}

ALLOWED_SYMBOLS = set('0123456789+-*/=')

# Reverse lookup: map frozenset of segments to a symbol.
SEGMENTS_TO_SYMBOL = {frozenset(v): k for k, v in SYMBOL_TO_SEGMENTS.items()}

# Extra mappings for creative transformations:
# Example: if "*" loses one match so that only {8} or {9} remains, interpret that as "/".
SEGMENTS_TO_SYMBOL[frozenset({8})] = '/'
SEGMENTS_TO_SYMBOL[frozenset({9})] = '/'

# If "-" (which is {6}) gains a match (segment 0) so that it becomes {0,6}, interpret that as "+".
SEGMENTS_TO_SYMBOL[frozenset({0,6})] = '+'
TOTAL = {0,1,2,3,4,5,6,7,8,9,10,11,12}
# =============================================================================
# 2. Universal Segment Coordinates (40x60 box)
# =============================================================================
SEGMENT_COORDS = {
    0: ((10, 5), (30, 5)),    # top horizontal
    1: ((30, 5), (30, 25)),   # upper-right vertical
    2: ((30, 25), (30, 45)),  # lower-right vertical
    3: ((10, 45), (30, 45)),  # bottom horizontal
    4: ((10, 25), (10, 45)),  # lower-left vertical
    5: ((10, 5), (10, 25)),   # upper-left vertical
    6: ((10, 25), (30, 25)),  # middle horizontal
    # Extra segments for operators:
    7: ((20, 5), (20, 45)),   # vertical for plus sign
    8: ((10, 10), (30, 40)),  # diagonal for multiply (first stroke)
    9: ((30, 10), (10, 40)),  # diagonal for multiply (second stroke)
    10: ((10, 40), (30, 10)), # diagonal for divide
    11: ((10, 15), (30, 15)), # top horizontal for equal
    12: ((10, 35), (30, 35)), # bottom horizontal for equal
}

# =============================================================================
# 3. Move Functions with Optional Conversion
# =============================================================================
def possible_match_moves(from_sym, to_sym, k=1, conversion=None):
    """
    Yield all ways to move k matchsticks from from_sym to to_sym.
    Only consider sticks not already present in the target.
    If a conversion function is provided (for k==1), try converting the moved segment.
    """
    s1 = set(SYMBOL_TO_SEGMENTS[from_sym])
    s2 = set(SYMBOL_TO_SEGMENTS[to_sym])
    available = [seg for seg in s1]
    available_for_s2 = [seg for seg in TOTAL if seg not in s2]
    for segs in itertools.combinations(available, k):
        for segs2 in itertools.combinations(available_for_s2, k):
            new_s1 = s1 - set(segs)
            new_s2 = s2 | set(segs2)
            new_from = SEGMENTS_TO_SYMBOL.get(frozenset(new_s1))
            new_to = SEGMENTS_TO_SYMBOL.get(frozenset(new_s2))
            if new_from is not None and new_to is not None:
                yield (new_from, new_to)


# =============================================================================
# 4. Recursive Generation of Arbitrary Moves
# =============================================================================
def generate_moves_recursive(eq, moves_left, move_history=None):
    """
    Recursively generate all final equations and move histories by applying moves.
    Each move is a tuple (i, j, k, (old_src, old_tgt, new_src, new_tgt)).
    This function yields (new_eq, move_history) pairs.
    """
    if move_history is None:
        move_history = []
    if moves_left == 0:
        yield eq, move_history
    else:
        for i in range(len(eq)):
            if eq[i] not in SYMBOL_TO_SEGMENTS:
                continue
            for j in range(len(eq)):
                if i == j:
                    continue
                if eq[j] not in SYMBOL_TO_SEGMENTS:
                    continue
                for k in range(1, moves_left + 1):
                    for (new_src, new_tgt) in possible_match_moves(eq[i], eq[j], k):
                        new_eq_list = list(eq)
                        new_eq_list[i] = new_src
                        new_eq_list[j] = new_tgt
                        new_eq = "".join(new_eq_list)
                        new_history = move_history + [(i, j, k, (eq[i], eq[j], new_src, new_tgt))]
                        yield from generate_moves_recursive(new_eq, moves_left - k, new_history)

# =============================================================================
# 5. New Drawing Functions with Endpoint Markers
# =============================================================================
def draw_symbol_realistic(
    draw,
    symbol,
    offset=(0, 0),
    thickness=4,
    head_radius=4
):
    """
    Draw a symbol (digit or operator) in a 60×80 box, using draw_realistic_match
    for each segment.
    """
    if symbol not in SYMBOL_TO_SEGMENTS:
        return
    segments = SYMBOL_TO_SEGMENTS[symbol]
    ox, oy = offset
    for seg_id in segments:
        (sx, sy), (ex, ey) = SEGMENT_COORDS[seg_id]
        start = (ox + sx, oy + sy)
        end   = (ox + ex, oy + ey)
        draw_realistic_match(
            draw,
            start,
            end,
            thickness=thickness,
            head_radius=head_radius
        )

def draw_equation_universal(eq_str, filename, symbol_w=60, symbol_h=80):
    """
    Draw the entire equation eq_str in a minimal bounding box (width = #symbols * symbol_w,
    height = symbol_h). Centered? It's effectively "centered" with no extra margin.
    
    We use a white background. If you want a transparent background,
    switch to mode="RGBA" and fill=(255,255,255,0).
    """
    # Count how many symbols (excluding spaces)
    symbols = [ch for ch in eq_str if ch != ' ']
    n = len(symbols)

    # The total image size
    img_w = n * symbol_w
    img_h = symbol_h

    # Create a new white image
    img = Image.new("RGB", (img_w, img_h), (255, 255, 255))
    draw_obj = ImageDraw.Draw(img)

    x_cursor = 0
    for ch in eq_str:
        if ch == ' ':
            # Optionally handle spaces
            x_cursor += symbol_w // 2
            continue
        # Draw the symbol in the box [x_cursor, 0, x_cursor+symbol_w, symbol_h]
        draw_symbol_realistic(draw_obj, ch, offset=(x_cursor, 0))
        x_cursor += symbol_w

    img.save(filename)

def draw_realistic_match(draw, start, end,
                         match_color=(235,210,150),
                         head_color=(200,0,0),
                         thickness=4,
                         head_radius=4):
    """Draw a single matchstick from start to end with a small red head at start."""
    x1, y1 = start
    x2, y2 = end
    length = math.hypot(x2 - x1, y2 - y1)
    angle = math.atan2(y2 - y1, x2 - x1)

    half_t = thickness/2.0
    dx = math.cos(angle)
    dy = math.sin(angle)
    px = -dy
    py = dx

    # corners of the rotated rectangle
    def corner(x, y, px, py):
        return (x + px, y + py)

    top_left     = corner(x1 + px*half_t, y1 + py*half_t, 0, 0)
    top_right    = corner(x2 + px*half_t, y2 + py*half_t, 0, 0)
    bottom_right = corner(x2 - px*half_t, y2 - py*half_t, 0, 0)
    bottom_left  = corner(x1 - px*half_t, y1 - py*half_t, 0, 0)

    # wooden body
    draw.polygon([top_left, top_right, bottom_right, bottom_left], fill=match_color)

    # red head at start
    r = head_radius
    draw.ellipse((x1-r, y1-r, x1+r, y1+r), fill=head_color)

# A brownish background color (simulate wood).
BACKGROUND_COLOR = (70, 45, 20)  # Or load an actual wood texture if you want


# =============================================================================
# 7. Utility Functions: Equation Evaluation & Movable Positions
# =============================================================================
def is_equation_correct(eq_str):
    parts = eq_str.split('=')
    if len(parts) != 2:
        return False
    try:
        return abs(eval(parts[0]) - eval(parts[1])) < 1e-6
    except Exception:
        return False

def extract_movable_positions(eq_str):
    return [i for i, ch in enumerate(eq_str) if ch in SYMBOL_TO_SEGMENTS]

# =============================================================================
# 8. Folder Helper & Move Classification
# =============================================================================
def ensure_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def classify_move(src, tgt):
    def sym_type(s):
        return "digit" if s in "0123456789" else "operator"
    return sym_type(src) + "_to_" + sym_type(tgt)
