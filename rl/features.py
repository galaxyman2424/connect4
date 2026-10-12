"""Board features for the tunable heuristic (rl/evolve.py).

The hand-crafted heuristic (engine/heuristic.py) is a weighted sum of five
counts with weights chosen by hand: centre stones 3, own open three 5, own
open two 2, opponent open three -4, opponent open two -2.  Here the same
counts plus five more become a feature vector, and evolution chooses the
weights.  With the hand weights and zeros for the new features the score is
identical to engine.heuristic.evaluate_bits (tests/test_rl.py checks this).

Features (side-to-move "me" vs "opp"; everything bit-parallel):
  0 centre      my stones - opponent stones in the centre column
  1 three       my open threes  (4-cell window: 3 mine + 1 empty)
  2 two         my open twos    (2 mine + 2 empty)
  3 opp_three   opponent open threes
  4 opp_two     opponent open twos
  5 near_centre my - opponent stones in columns 2 and 4 (next to the centre)
  6 threats     empty cells that would complete MY four (anywhere on the board)
  7 opp_threats empty cells that would complete the OPPONENT's four
  8 good_threats      my threat cells on "my" rows: odd rows (1, 3, 5 from
                      the bottom) if I moved first, even rows otherwise - the
                      classic Connect 4 zugzwang rule (Allis 1988)
  9 opp_good_threats  the same for the opponent
"""

from engine.board import BOARD_MASK, COLUMN_MASK, COLS, H1, winning_cells_mask
from engine.heuristic import CENTER, OPP_THREE, OPP_TWO, THREE, TWO, _counts

NAMES = ("centre", "three", "two", "opp_three", "opp_two", "near_centre",
         "threats", "opp_threats", "good_threats", "opp_good_threats")
HAND_WEIGHTS = (CENTER, THREE, TWO, -OPP_THREE, -OPP_TWO, 0, 0, 0, 0, 0)
N_FEATURES = len(NAMES)
WEIGHT_LIMIT = 60.0       # keeps scores far below the search's win scores (~1e6)

_CENTER = COLUMN_MASK[3]
_NEAR = COLUMN_MASK[2] | COLUMN_MASK[4]
_ODD = 0       # heights 0, 2, 4 = rows 1, 3, 5 counted from the bottom
_EVEN = 0
for _c in range(COLS):
    for _h in range(6):
        if _h % 2 == 0:
            _ODD |= 1 << (_c * H1 + _h)
        else:
            _EVEN |= 1 << (_c * H1 + _h)


def features(me, opp):
    mask = me | opp
    t, w = _counts(me, BOARD_MASK & ~opp)
    ot, ow = _counts(opp, BOARD_MASK & ~me)
    mt = winning_cells_mask(me, mask)
    oth = winning_cells_mask(opp, mask)
    me_first = me.bit_count() == opp.bit_count()      # equal counts: side to move started
    good_me, good_opp = (_ODD, _EVEN) if me_first else (_EVEN, _ODD)
    return ((me & _CENTER).bit_count() - (opp & _CENTER).bit_count(), t, w, ot, ow,
            (me & _NEAR).bit_count() - (opp & _NEAR).bit_count(),
            mt.bit_count(), oth.bit_count(), (mt & good_me).bit_count(), (oth & good_opp).bit_count())


class WeightedEvaluator:
    """Leaf evaluation f(me, opp) = weights . features(me, opp) for engine.minimax.search.
    Picklable, so it works inside multiprocessing workers."""

    def __init__(self, weights):
        w = [max(-WEIGHT_LIMIT, min(WEIGHT_LIMIT, float(x))) for x in weights]
        if len(w) != N_FEATURES:
            raise ValueError("need %d weights" % N_FEATURES)
        self.w = tuple(w)
        self.extended = any(abs(x) > 1e-12 for x in w[5:])

    def __call__(self, me, opp):
        w = self.w
        if not self.extended:            # fast path: only the five classic features
            t, tw = _counts(me, BOARD_MASK & ~opp)
            ot, ow = _counts(opp, BOARD_MASK & ~me)
            return (w[0] * ((me & _CENTER).bit_count() - (opp & _CENTER).bit_count())
                    + w[1] * t + w[2] * tw + w[3] * ot + w[4] * ow)
        f = features(me, opp)
        return sum(a * b for a, b in zip(w, f))
