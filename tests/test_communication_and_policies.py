import unittest

from physioworld.communication import (
    ErasureChannel,
    PacketLoss,
    RLNCDecoder,
    RLNCEncoder,
    packetize,
    reconstruct,
)
from physioworld.policies import DynamicProgrammingOracle, FiniteMDP


class CommunicationTests(unittest.TestCase):
    def test_rlnc_reconstructs_symbols_after_reordering(self):
        source = (b"aa", b"bb", b"cc")
        words = RLNCEncoder(source, seed=11).encode(5)
        decoder = RLNCDecoder(3, 2)
        for word in reversed(words[:3]):
            decoder.add(word)
        self.assertEqual(decoder.decode(), source)

    def test_rlnc_randomized_round_trips(self):
        for seed in range(5):
            source = tuple(bytes([seed, index, index + 1]) for index in range(4))
            words = RLNCEncoder(source, seed=seed).encode(8)
            decoder = RLNCDecoder(4, 3)
            for word in reversed(words):
                decoder.add(word)
            self.assertEqual(decoder.decode(), source)

    def test_erasure_is_not_zero_recovery(self):
        packets = packetize(b"abcdef", 2)
        result = ErasureChannel(PacketLoss(1.0, seed=1)).transmit(packets)
        self.assertEqual(result.received, ())
        with self.assertRaises(ValueError):
            reconstruct(result.received, expected_count=len(packets))


class PolicyTests(unittest.TestCase):
    def test_exact_oracle(self):
        mdp = FiniteMDP(
            states=(0, 1),
            actions={0: ("stay", "go"), 1: ("stay", "go")},
            transition=lambda state, action: {1: 1.0} if action == "go" else {state: 1.0},
            reward=lambda state, action: 1.0 if action == "go" else 0.0,
            horizon=2,
        )
        oracle = DynamicProgrammingOracle(mdp)
        self.assertEqual(oracle.action_at(0, 0), "go")
        self.assertEqual(oracle.value(0, 0), 2.0)


if __name__ == "__main__":
    unittest.main()
