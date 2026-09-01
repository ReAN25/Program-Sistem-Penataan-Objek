import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
FULL_GUI = ROOT / "full_gui"
sys.path.insert(0, str(FULL_GUI))

spec = importlib.util.spec_from_file_location(
    "full_gui_stream_transport", FULL_GUI / "stream_transport.py"
)
transport = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(transport)

from pulse_planner import PulseBlock


class StreamTransportTest(unittest.TestCase):
    def test_crc_nak_retries_same_block_until_ack(self):
        streamer = transport.PulseStreamer("TEST")
        block = PulseBlock(41, ((1, 0, 0), (1, 0, 1)))
        writes = []
        replies = iter(
            [
                transport.PulseStreamError("NAK:CRC:41:EXPECTED:0000:ACTUAL:1111:LEN:11"),
                transport.PulseStreamError("NAK:FORMAT"),
                "BLOCK_ACK:41:0",
            ]
        )

        streamer._write = writes.append

        def fake_wait(_predicate, _timeout_s):
            reply = next(replies)
            if isinstance(reply, Exception):
                raise reply
            return reply

        streamer._wait_for = fake_wait
        streamer._send_block(block)

        self.assertEqual(writes, [block.command, block.command, block.command])

    def test_stop_interrupts_retry_without_ping(self):
        streamer = transport.PulseStreamer("TEST")
        block = PulseBlock(9, ((0, 1, 0),))
        writes = []
        streamer._write = writes.append

        def fake_wait(_predicate, _timeout_s):
            raise transport.PulseStreamError("Streaming dihentikan pengguna.")

        streamer._wait_for = fake_wait
        with self.assertRaisesRegex(transport.PulseStreamError, "dihentikan pengguna"):
            streamer._send_block(block)

        self.assertEqual(writes, [block.command])

    def test_missing_ack_uses_state_without_duplicate_block(self):
        streamer = transport.PulseStreamer("TEST")
        block = PulseBlock(12, ((2, 1, 15),))
        writes = []
        replies = iter([
            transport.PulseStreamError("Timeout menunggu jawaban Arduino."),
            "STATE:RUN=1:XYRUN=1:XYWAIT=0:ACTIVE=13:READY=20:X=10:Y=5",
        ])
        streamer._write = writes.append

        def fake_wait(_predicate, _timeout_s):
            reply = next(replies)
            if isinstance(reply, Exception):
                raise reply
            return reply

        streamer._wait_for = fake_wait
        streamer._send_block(block)
        self.assertEqual(writes, [block.command, "STATE?\n"])

    def test_state_timeout_retries_payload_without_stopping(self):
        streamer = transport.PulseStreamer("TEST")
        block = PulseBlock(27, ((1, 0, 0),))
        writes = []
        replies = iter([
            transport.PulseStreamError("Timeout menunggu jawaban Arduino."),
            "BLOCK_ACK:27:SLOT:0",
        ])
        state_replies = iter([
            transport.PulseStreamError("Timeout menunggu jawaban Arduino."),
            {"ACTIVE": "0", "XYRUN": "0", "XYWAIT": "0", "CFG": "1", "TOTAL": "100"},
        ])
        streamer._write = writes.append

        def fake_wait(_predicate, _timeout_s):
            reply = next(replies)
            if isinstance(reply, Exception):
                raise reply
            return reply

        streamer._wait_for = fake_wait
        def fake_state(_timeout_s=0.5):
            state = next(state_replies)
            if isinstance(state, Exception):
                raise state
            return state

        streamer._query_state = fake_state
        streamer._send_block(block)
        self.assertEqual(writes, [block.command, block.command])

    def test_count_error_is_retried_without_stopping(self):
        streamer = transport.PulseStreamer("TEST")
        block = PulseBlock(70000, ((1, 0, 0),))
        writes = []
        replies = iter([
            transport.PulseStreamError("NAK:COUNT:70000"),
            "BLOCK_ACK:70000:0",
        ])
        streamer._write = writes.append

        def fake_wait(_predicate, _timeout_s):
            reply = next(replies)
            if isinstance(reply, Exception):
                raise reply
            return reply

        streamer._wait_for = fake_wait
        streamer._send_block(block)
        self.assertEqual(writes, [block.command, block.command])

    def test_long_stream_uses_32_bit_sequence_and_releases_done_blocks(self):
        total = 65540

        class LongPlan:
            block_count = total
            blocks = [None] * total
            block_ms = 20
            slice_us = 10000

            def get_block(self, sequence):
                block = self.blocks[sequence]
                if block is None:
                    block = PulseBlock(sequence, ((1, 0, 0),))
                    self.blocks[sequence] = block
                return block

            def release_block(self, sequence):
                self.blocks[sequence] = None

        plan = LongPlan()
        streamer = transport.PulseStreamer("TEST")
        writes = []
        streamer.connect = lambda: None
        streamer._ping_until_response = lambda: None
        streamer._write = writes.append

        def fake_wait(predicate, _timeout_s):
            for value in (f"STREAM_ACCEPTED:{total}:10000", "STREAM_RUNNING"):
                if predicate(value):
                    return value
            raise AssertionError("handshake predicate tidak sesuai simulasi")

        streamer._wait_for = fake_wait
        streamer._send_block = lambda block: writes.append(block.sequence)
        next_done = 0

        def fake_readline(_timeout_s):
            nonlocal next_done
            if next_done < total:
                value = f"BLOCK_DONE:{next_done}"
                next_done += 1
                return value
            return "STREAM_DONE"

        streamer._readline = fake_readline
        streamer.stream(plan)

        self.assertEqual(writes[-1], total - 1)
        self.assertEqual(sum(isinstance(value, int) for value in writes), total)
        self.assertIsNone(plan.blocks[0])
        self.assertIsNone(plan.blocks[65535])
        self.assertIsNone(plan.blocks[-1])
        self.assertEqual(next_done, total)


if __name__ == "__main__":
    unittest.main()
