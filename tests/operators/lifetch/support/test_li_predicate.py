"""CPU contract checks for LI resident predicate publication.

These models check unsigned-score comparison, causal visibility, and the
compact bitmap consumed by FastService. They do not compile or execute CANN
MicroAPI instructions and are not a substitute for device/simulator validation.
"""

import random
import unittest


LANES = 128
VICTIMS = 2048
RESIDENT = 16384


def pack_predicate(predicate):
    """Model the consumer's low bit in each two-bit token field."""
    words = [0] * (len(predicate) // 16)
    for lane, selected in enumerate(predicate):
        if selected:
            words[lane // 16] |= 1 << (2 * (lane % 16))
    return b"".join(word.to_bytes(4, "little") for word in words)


def reload_predicate(stored):
    words = [int.from_bytes(stored[i : i + 4], "little") for i in range(0, len(stored), 4)]
    return [bool(word & (1 << (2 * lane))) for word in words for lane in range(16)]


def base_bitmap(scores, victims, threshold, end):
    """Model ClassifyScore -> UB store/reload -> visibility AND."""
    result = bytearray()
    for begin in range(0, VICTIMS, LANES):
        threshold_mask = [score > threshold for score in scores[begin : begin + LANES]]
        reloaded = reload_predicate(pack_predicate(threshold_mask))
        result.extend(
            pack_predicate(
                [selected and token < end for selected, token in zip(reloaded, victims[begin : begin + LANES])]
            )
        )
    return bytes(result)


def register_bitmap(scores, victims, threshold, end):
    """Model register threshold mask AND register visibility mask, one store."""
    result = bytearray()
    for begin in range(0, VICTIMS, LANES):
        score_words = scores[begin : begin + LANES]
        token_words = victims[begin : begin + LANES]
        threshold_mask = sum(1 << i for i, score in enumerate(score_words) if score > threshold)
        visible_mask = sum(1 << i for i, token in enumerate(token_words) if token < end)
        selected = threshold_mask & visible_mask
        result.extend(pack_predicate([bool(selected & (1 << i)) for i in range(LANES)]))
    return bytes(result)


class ResidentPredicateContractTest(unittest.TestCase):
    def check_case(self, scores, victims, threshold, end):
        expected_flags = [score > threshold and token < end for score, token in zip(scores, victims)]
        expected = pack_predicate(expected_flags)
        self.assertEqual(len(expected), 512)
        self.assertEqual(base_bitmap(scores, victims, threshold, end), expected)
        self.assertEqual(register_bitmap(scores, victims, threshold, end), expected)
        self.assertEqual(reload_predicate(expected), expected_flags)
        # Odd bits are not candidates in FastService's 0x55555555 mask.
        for offset in range(0, len(expected), 4):
            self.assertEqual(int.from_bytes(expected[offset : offset + 4], "little") & 0xAAAAAAAA, 0)

    def test_equal_scores_keep_strict_gt(self):
        victims = list(range(VICTIMS))
        for threshold in (0, 1, 0x3F80, 0x7FFF, 0xFFFF):
            with self.subTest(threshold=threshold):
                self.check_case([threshold] * VICTIMS, victims, threshold, RESIDENT)
                self.assertFalse(any(register_bitmap([threshold] * VICTIMS, victims, threshold, RESIDENT)))

    def test_causal_boundary_and_all_mask_extremes(self):
        # Deliberately include token=end-1, end, end+1 across vector boundaries.
        victims = [(i * 97) % RESIDENT for i in range(VICTIMS)]
        for end in (0, 1, 127, 128, 129, 2048, RESIDENT - 1, RESIDENT):
            for threshold in (0, 0x7FFF, 0xFFFF):
                with self.subTest(end=end, threshold=threshold):
                    scores = [0xFFFF] * VICTIMS
                    victims[127:130] = [max(0, end - 1), min(RESIDENT - 1, end), min(RESIDENT - 1, end + 1)]
                    self.check_case(scores, victims, threshold, end)

    def test_random_unsigned_scores_and_sparse_victims(self):
        rng = random.Random(20261105)
        for case in range(48):
            threshold = rng.randrange(0x10000)
            scores = [rng.randrange(0x10000) for _ in range(VICTIMS)]
            # BF16 ties must remain excluded, including each chunk boundary.
            for lane in range(0, VICTIMS, LANES):
                scores[lane] = threshold
                scores[lane + LANES - 1] = threshold
            victims = rng.sample(range(RESIDENT), VICTIMS)
            end = rng.randrange(RESIDENT + 1)
            with self.subTest(case=case, threshold=threshold, end=end):
                self.check_case(scores, victims, threshold, end)


if __name__ == "__main__":
    unittest.main()
