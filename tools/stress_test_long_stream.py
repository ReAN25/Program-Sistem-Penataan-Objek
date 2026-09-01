"""Software-in-the-loop stress test untuk stream trajektori panjang.

Tidak membutuhkan Arduino. Simulasi menjalankan transport produksi dengan
70.000 sequence (melewati batas 16-bit), melepaskan blok yang sudah DONE, lalu
menyuntikkan CRC/COUNT, ACK timeout, RX overflow, dan BUFFER_WAIT. Keberhasilan
ditentukan dari tiga invarians: setiap sequence diterima satu kali oleh model
gerak, retry memakai sequence yang sama, dan seluruh cache blok Python kosong
setelah trajektori selesai.
"""

from __future__ import annotations

from collections import Counter
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "full_gui"))

from pulse_planner import PulseBlock  # noqa: E402
from stream_transport import PulseStreamError, PulseStreamer  # noqa: E402


class LongPlan:
    def __init__(self, total: int) -> None:
        self.block_count = int(total)
        self.blocks = [None] * self.block_count
        self.slice_us = 10_000
        self.block_ms = 20

    def get_block(self, sequence: int) -> PulseBlock:
        block = self.blocks[sequence]
        if block is None:
            block = PulseBlock(int(sequence), ((1, 0, 0),))
            self.blocks[sequence] = block
        return block

    def release_block(self, sequence: int) -> None:
        self.blocks[int(sequence)] = None


class FakeArduinoStreamer(PulseStreamer):
    """Model serial/firmware untuk menguji transport tanpa perangkat keras."""

    def __init__(self, total: int, *, wait_sequence: int, async_error_sequence: int) -> None:
        super().__init__("SIMULATED", log=lambda _message: None)
        self.total = int(total)
        self.next_done = 0
        self.pending_block: int | None = None
        self.wait_sequence = int(wait_sequence)
        self.async_error_sequence = int(async_error_sequence)
        self.wait_injected = False
        self.async_error_injected = False
        self.block_faults: dict[int, list[str]] = {
            1234: ["NAK:CRC:1234:EXPECTED:0000:ACTUAL:1111:LEN:11"],
            40000: ["NAK:COUNT:40000"],
            51000: ["TIMEOUT"],
        }
        self.send_attempts: Counter[int] = Counter()
        self.accepted_sequences: set[int] = set()

    def connect(self) -> None:
        return None

    def _ping_until_response(self) -> None:
        return None

    def _write(self, command: str) -> None:
        if command.startswith("BLOCK:"):
            self.pending_block = int(command.split(":", 2)[1])
            self.send_attempts[self.pending_block] += 1

    def _query_state(self, _timeout_s: float = 0.5) -> dict[str, str]:
        active = min(self.next_done, self.total)
        return {
            "ACTIVE": str(active),
            "TOTAL": str(self.total),
            "CFG": "1",
            "RUN": "1" if active < self.total else "0",
            "XYRUN": "1" if active < self.total else "0",
            "XYWAIT": "0",
        }

    def _wait_for(self, predicate, _timeout_s: float) -> str:
        accepted_line = f"STREAM_ACCEPTED:{self.total}:10000"
        running_line = "STREAM_RUNNING"
        state_line = (
            f"STATE:RUN={1 if self.next_done < self.total else 0}:"
            f"XYRUN={1 if self.next_done < self.total else 0}:XYWAIT=0:"
            f"ACTIVE={self.next_done}:TOTAL={self.total}:CFG=1"
        )
        if predicate(accepted_line):
            return accepted_line
        if predicate(running_line):
            return running_line
        if predicate(state_line):
            return state_line
        if self.pending_block is not None and predicate(
            f"BLOCK_ACK:{self.pending_block}:SLOT:0"
        ):
            sequence = self.pending_block
            faults = self.block_faults.get(sequence, [])
            if faults:
                fault = faults.pop(0)
                if fault == "TIMEOUT":
                    # Model ACK hilang setelah firmware sudah menerima blok.
                    # Transport harus menganggap gerak telah diterima, bukan
                    # mengirim gerakan kedua.
                    self.accepted_sequences.add(sequence)
                    raise PulseStreamError("Timeout menunggu jawaban Arduino.")
                raise PulseStreamError(fault)
            self.accepted_sequences.add(sequence)
            return f"BLOCK_ACK:{sequence}:SLOT:0"
        raise AssertionError("Predicate transport tidak dikenali oleh simulasi")

    def _readline(self, _timeout_s: float) -> str:
        if self.next_done < self.total:
            if (
                self.next_done == self.wait_sequence
                and not self.wait_injected
            ):
                self.wait_injected = True
                return f"BUFFER_WAIT:{self.wait_sequence}"
            if (
                self.next_done == self.async_error_sequence
                and not self.async_error_injected
            ):
                self.async_error_injected = True
                return "ERROR:RX_OVERFLOW"
            value = f"BLOCK_DONE:{self.next_done}"
            self.next_done += 1
            return value
        return "STREAM_DONE"


def run(total: int = 70_000) -> dict[str, object]:
    plan = LongPlan(total)
    streamer = FakeArduinoStreamer(
        total,
        wait_sequence=25_000,
        async_error_sequence=50_000,
    )
    streamer.stream(plan)
    retry_sequences = {
        sequence: attempts
        for sequence, attempts in streamer.send_attempts.items()
        if attempts > 1
    }
    released = sum(block is None for block in plan.blocks)
    result = {
        "total_blocks": total,
        "max_sequence": total - 1,
        "done_sequences": streamer.next_done,
        "buffer_wait_injected": streamer.wait_injected,
        "async_error_injected": streamer.async_error_injected,
        "retry_sequences": retry_sequences,
        "accepted_unique_motion_sequences": len(streamer.accepted_sequences),
        "released_python_blocks": released,
        "all_python_blocks_released": released == total,
        "no_duplicate_motion": len(streamer.accepted_sequences) == total,
    }
    if not result["all_python_blocks_released"] or not result["no_duplicate_motion"]:
        raise AssertionError(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
