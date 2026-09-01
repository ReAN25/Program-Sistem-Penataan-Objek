"""Transport serial dengan sliding window 32 blok dan retry CRC/timeout."""

from __future__ import annotations

import queue
import threading
import time
from typing import Callable

import serial

from pulse_planner import PulsePlan


WINDOW_SLOTS = 32
RETRY_DELAY_S = 0.12


class PulseStreamError(RuntimeError):
    pass


class PulseStreamer:
    def __init__(
        self,
        port: str,
        baudrate: int = 250000,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.log = log or (lambda _message: None)
        self.serial: serial.Serial | None = None
        self.stop_event = threading.Event()
        self._deferred: queue.SimpleQueue[str] = queue.SimpleQueue()

    @staticmethod
    def _compact_sequence_log(line: str) -> bool:
        if not line.startswith(("BLOCK_ACK:", "BLOCK_DONE:")):
            return True
        try:
            sequence = int(line.split(":", 2)[1])
        except (ValueError, IndexError):
            return True
        return sequence < 3 or sequence % 25 == 0

    def connect(self) -> None:
        if self.serial and self.serial.is_open:
            return
        self.serial = serial.Serial(self.port, self.baudrate, timeout=0.1)
        time.sleep(1.8)
        self.serial.reset_input_buffer()
        self._write("HELLO\n")
        line = self._wait_for(lambda value: value.startswith("PULSE_BLOCK_READY"), 3.0)
        self.log(f"RX <- {line}")

    def close(self) -> None:
        if self.serial and self.serial.is_open:
            self.serial.close()

    def stop(self) -> None:
        self.stop_event.set()
        if self.serial and self.serial.is_open:
            self._write("STOP\n")

    def _write(self, command: str) -> None:
        if not self.serial or not self.serial.is_open:
            raise PulseStreamError("Port serial belum terhubung.")
        self.serial.write(command.encode("ascii"))
        self.serial.flush()

    def _readline(self, timeout_s: float) -> str:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.stop_event.is_set():
                raise PulseStreamError("Streaming dihentikan pengguna.")
            try:
                return self._deferred.get_nowait()
            except queue.Empty:
                pass
            assert self.serial is not None
            raw = self.serial.readline()
            if raw:
                line = raw.decode("ascii", errors="replace").strip()
                if line:
                    if self._compact_sequence_log(line):
                        self.log(f"RX <- {line}")
                    return line
        raise PulseStreamError("Timeout menunggu jawaban Arduino.")

    def _wait_for(self, predicate, timeout_s: float) -> str:
        deadline = time.monotonic() + timeout_s
        postponed: list[str] = []
        try:
            while time.monotonic() < deadline:
                line = self._readline(max(0.05, deadline - time.monotonic()))
                if predicate(line):
                    return line
                if line.startswith(("ERROR:", "NAK:", "STREAM_ABORTED", "BUFFER_UNDERRUN")):
                    raise PulseStreamError(line)
                postponed.append(line)
        finally:
            for line in postponed:
                self._deferred.put(line)
        raise PulseStreamError("Timeout handshake Arduino.")

    @staticmethod
    def _parse_state(line: str) -> dict[str, str]:
        values: dict[str, str] = {}
        for token in line.split(":")[1:]:
            if "=" in token:
                key, value = token.split("=", 1)
                values[key] = value
        return values

    def _discard_pending_input(self) -> None:
        """Buang ACK/progres dari stream lama setelah STOP terkonfirmasi."""
        while True:
            try:
                self._deferred.get_nowait()
            except queue.Empty:
                break
        if self.serial and self.serial.is_open:
            self.serial.reset_input_buffer()

    @staticmethod
    def _plan_block(plan: PulsePlan, sequence: int):
        getter = getattr(plan, "get_block", None)
        if getter is not None:
            return getter(sequence)
        return plan.blocks[sequence]

    @staticmethod
    def _release_plan_block(plan: PulsePlan, sequence: int) -> None:
        releaser = getattr(plan, "release_block", None)
        if releaser is not None:
            releaser(sequence)

    def _query_state(self, timeout_s: float = 0.5) -> dict[str, str]:
        attempt = 0
        while not self.stop_event.is_set():
            attempt += 1
            try:
                self._write("STATE?\n")
                line = self._wait_for(
                    lambda value: value.startswith(("STATE:", "STATUS:")), timeout_s
                )
                self.log(f"Konfirmasi keselamatan <- {line}")
                return self._parse_state(line)
            except PulseStreamError as exc:
                reason = str(exc)
                if reason == "Streaming dihentikan pengguna.":
                    raise
                if reason.startswith("Port serial belum terhubung"):
                    raise
                if reason.startswith((
                    "ERROR:NOT_HOMED", "ERROR:Z_ENCODER",
                    "ERROR:Z_ENCODER_READ", "ERROR:Z_STALL",
                    "ERROR:Z_WRONG_DIRECTION",
                )):
                    raise
                if attempt == 1 or attempt % 5 == 0:
                    self.log(f"STATE belum dijawab ({attempt}); kirim ulang: {exc}")
                time.sleep(RETRY_DELAY_S)
        raise PulseStreamError("Pemeriksaan STATE dihentikan pengguna.")

    def _ping_until_response(self) -> None:
        attempt = 0
        while not self.stop_event.is_set():
            attempt += 1
            try:
                self._write("PING\n")
                self._wait_for(lambda line: line == "PONG", 0.5)
                return
            except PulseStreamError as exc:
                if str(exc) == "Streaming dihentikan pengguna.":
                    raise
                if str(exc).startswith("Port serial belum terhubung"):
                    raise
                if attempt == 1 or attempt % 5 == 0:
                    self.log(f"PING belum dijawab ({attempt}); kirim ulang: {exc}")
                time.sleep(RETRY_DELAY_S)
        raise PulseStreamError("PING dihentikan pengguna.")

    def abort_until_idle(self) -> bool:
        """Hentikan stream lama dan tunggu firmware benar-benar idle.

        Firmware tetap dapat menerima STOP ketika blok sebelumnya gagal.
        Pengiriman diulang karena satu frame STOP dapat hilang, lalu STATE
        diverifikasi sebelum caller mengirim STREAM_BEGIN baru.
        """
        attempt = 0
        while not self.stop_event.is_set():
            attempt += 1
            if self.stop_event.is_set():
                raise PulseStreamError("Streaming dihentikan pengguna.")
            try:
                self._write("STOP\n")
                self.log(f"TX -> STOP (pemulihan stream, percobaan {attempt})")
                self._wait_for(lambda line: line == "STREAM_ABORTED", 1.0)
                self._write("STATE?\n")
                state_line = self._wait_for(
                    lambda line: line.startswith(("STATE:", "STATUS:")), 1.0
                )
                state = self._parse_state(state_line)
                if not any(
                    int(state.get(key, "0"))
                    for key in ("RUN", "XYRUN", "XYWAIT")
                ):
                    self._discard_pending_input()
                    self.log("STATE memastikan Arduino idle setelah STOP.")
                    return True
            except PulseStreamError as exc:
                if str(exc) == "Streaming dihentikan pengguna.":
                    raise
                if str(exc).startswith("Port serial belum terhubung"):
                    raise
                if attempt == 1 or attempt % 5 == 0:
                    self.log(f"STOP pemulihan belum terkonfirmasi ({attempt}): {exc}")
                time.sleep(RETRY_DELAY_S)
        raise PulseStreamError("STOP pemulihan dihentikan pengguna.")

    def _send_block(self, block) -> None:
        attempt = 0
        while True:
            attempt += 1
            if (
                attempt == 1
                and (block.sequence < 3 or block.sequence % 25 == 0)
            ) or attempt in (2, 3) or attempt % 10 == 0:
                self.log(
                    f"TX -> BLOCK {block.sequence} ({len(block.slices)} slice, "
                    f"CRC={block.crc16:04X}, {len(block.command.encode('ascii'))} byte, "
                    f"percobaan={attempt})"
                )
            self._write(block.command)
            try:
                self._wait_for(
                    lambda line: line.startswith(f"BLOCK_ACK:{block.sequence}:"),
                    0.12,
                )
                return
            except PulseStreamError as exc:
                reason = str(exc)
                if reason == "Streaming dihentikan pengguna.":
                    raise
                if reason.startswith("Port serial belum terhubung"):
                    raise
                recoverable_nak = reason.startswith(
                    (
                        f"NAK:CRC:{block.sequence}", "NAK:FORMAT",
                        f"NAK:PAYLOAD:{block.sequence}", f"NAK:SLOT:{block.sequence}",
                        f"NAK:COUNT:{block.sequence}", f"NAK:Z_TARGET:{block.sequence}",
                        "ERROR:UNKNOWN_COMMAND",
                        "ERROR:RX_OVERFLOW",
                    )
                )
                if recoverable_nak:
                    # Selama Arduino membalas CRC/FORMAT, link masih hidup.
                    # Ulangi blok yang sama tanpa batas; BUFFER_WAIT firmware
                    # akan melanjutkan otomatis setelah satu salinan valid.
                    if attempt <= 3 or attempt % 10 == 0:
                        self.log(
                            f"Retry aman BLOCK {block.sequence} percobaan {attempt}: {reason}"
                        )
                    time.sleep(RETRY_DELAY_S)
                    continue
                if reason.startswith("NAK:STATE"):
                    # Bedakan frame yang terlambat dari stream yang benar-benar
                    # hilang. Firmware baru mengirim CFG/TOTAL agar sequence
                    # tidak ditebak dari READY saja.
                    try:
                        state = self._query_state(0.5)
                        cfg_text = state.get("CFG")
                        total_text = state.get("TOTAL")
                        if cfg_text is None or total_text is None:
                            raise PulseStreamError("STATE lama tidak memiliki CFG/TOTAL")
                        configured = int(cfg_text)
                        total = int(total_text)
                        active = int(state.get("ACTIVE", "-1"))
                        running = int(state.get("XYRUN", state.get("RUN", "0")))
                        waiting = int(state.get("XYWAIT", "0"))
                        if not configured or block.sequence >= total:
                            raise PulseStreamError(
                                "STREAM_RESET:STATE_LOST"
                            )
                        if active > block.sequence or (
                            active == block.sequence and running and not waiting
                        ):
                            return
                        self.log(
                            f"NAK:STATE BLOCK {block.sequence}; stream masih valid "
                            f"(TOTAL={total}, ACTIVE={active}), kirim ulang blok yang sama."
                        )
                        time.sleep(RETRY_DELAY_S)
                        continue
                    except PulseStreamError as state_exc:
                        state_reason = str(state_exc)
                        if state_reason == "Streaming dihentikan pengguna.":
                            raise
                        if state_reason.startswith("Port serial belum terhubung"):
                            raise
                        if state_reason.startswith("STREAM_RESET:"):
                            raise
                        # STATE dapat hilang bersamaan dengan ACK ketika USB
                        # sedang penuh. Jangan menghentikan lintasan hanya
                        # karena satu query gagal; sequence dan payload yang
                        # sama tetap aman untuk dikirim ulang.
                        if attempt <= 3 or attempt % 10 == 0:
                            self.log(
                                f"STATE BLOCK {block.sequence} belum dijawab; "
                                f"retry payload yang sama ({attempt}): {state_reason}"
                            )
                        time.sleep(RETRY_DELAY_S)
                        continue
                if reason.startswith((
                    "ERROR:NOT_HOMED", "ERROR:Z_ENCODER",
                    "ERROR:Z_ENCODER_READ", "ERROR:Z_STALL",
                    "ERROR:Z_WRONG_DIRECTION",
                )):
                    raise PulseStreamError(
                        f"BLOCK {block.sequence} tidak dapat dilanjutkan: {reason}"
                    ) from exc
                if attempt <= 3 or attempt % 10 == 0:
                    self.log(f"Periksa STATE setelah BLOCK {block.sequence}: {reason}")
                try:
                    state = self._query_state(0.5)
                    active = int(state.get("ACTIVE", "-1"))
                    running = int(state.get("XYRUN", state.get("RUN", "0")))
                    waiting = int(state.get("XYWAIT", "0"))
                    if active > block.sequence or (
                        active == block.sequence and running and not waiting
                    ):
                        self.log(
                            f"STATE memastikan BLOCK {block.sequence} sudah aktif/selesai; "
                            "ACK yang hilang tidak menyebabkan pengiriman ganda."
                        )
                        return
                except PulseStreamError as state_exc:
                    state_reason = str(state_exc)
                    if state_reason == "Streaming dihentikan pengguna.":
                        raise
                    if state_reason.startswith("Port serial belum terhubung"):
                        raise
                    if state_reason.startswith("STREAM_RESET:"):
                        raise
                    if attempt <= 3 or attempt % 10 == 0:
                        self.log(
                            f"STATE BLOCK {block.sequence} belum dijawab; "
                            f"payload yang sama akan dikirim ulang ({attempt}): "
                            f"{state_reason}"
                        )
                    time.sleep(RETRY_DELAY_S)
                    continue
                time.sleep(RETRY_DELAY_S)

    def stream(self, plan: PulsePlan, progress=None) -> None:
        block_count = getattr(plan, "block_count", None)
        if block_count is None:
            block_count = len(getattr(plan, "blocks", ()))
        total_blocks = int(block_count)
        if total_blocks <= 0:
            raise PulseStreamError("Plan tidak memiliki blok gerak.")
        self.stop_event.clear()
        self.connect()
        self._ping_until_response()

        begin_command = f"STREAM_BEGIN:{total_blocks}:{int(plan.slice_us)}\n"
        self.log(f"TX -> {begin_command.strip()}")
        begin_attempt = 0
        while not self.stop_event.is_set():
            begin_attempt += 1
            self._write(begin_command)
            try:
                self._wait_for(lambda line: line.startswith("STREAM_ACCEPTED:"), 0.8)
                break
            except PulseStreamError as exc:
                reason = str(exc)
                # BEGIN_FORMAT dapat terjadi bila satu frame serial terbaca
                # tidak lengkap. Kirim ulang frame yang sama; BEGIN_TIMING
                # tetap fatal karena berarti parameter plan memang salah.
                if reason.startswith((
                    "ERROR:NOT_HOMED", "ERROR:Z_ENCODER", "ERROR:BEGIN_TIMING",
                )):
                    raise
                if reason.startswith("ERROR:ALREADY_RUNNING"):
                    self.log(
                        "STREAM_BEGIN ditolak karena stream lama masih aktif; "
                        "STOP sebelum mengulang blok baru."
                    )
                    if not self.abort_until_idle():
                        raise PulseStreamError(
                            "Arduino tetap aktif setelah ERROR:ALREADY_RUNNING."
                        ) from exc
                    # Laporkan reset ke caller agar posisi hardware dibaca
                    # ulang sebelum plan dibentuk kembali.
                    raise PulseStreamError(
                        "STREAM_RESET:ALREADY_RUNNING"
                    ) from exc
                if begin_attempt == 1 or begin_attempt % 5 == 0:
                    self.log(f"STREAM_BEGIN belum dijawab ({begin_attempt}); kirim ulang: {reason}")
                time.sleep(RETRY_DELAY_S)
        if self.stop_event.is_set():
            raise PulseStreamError("Streaming dihentikan pengguna.")

        next_sequence = 0
        while next_sequence < min(WINDOW_SLOTS, total_blocks):
            self._send_block(self._plan_block(plan, next_sequence))
            next_sequence += 1

        self.log("TX -> STREAM_RUN")
        run_attempt = 0
        while not self.stop_event.is_set():
            run_attempt += 1
            self._write("STREAM_RUN\n")
            try:
                self._wait_for(lambda line: line == "STREAM_RUNNING", 0.8)
                break
            except PulseStreamError as exc:
                reason = str(exc)
                if reason.startswith(("STREAM_ABORTED", "ERROR:NO_INITIAL_BLOCK")):
                    raise PulseStreamError("STREAM_RESET:RUN_STATE_LOST") from exc
                if reason.startswith("ERROR:ALREADY_RUNNING"):
                    # Jangan menganggap stream lama sebagai STREAM_RUN baru.
                    # Jika BEGIN sebelumnya gagal, slot/total dapat berasal
                    # dari trajektori berbeda. Hentikan lalu ulangi handshake.
                    self.log("STREAM_RUN ditolak karena stream lama masih aktif; STOP sebelum retry.")
                    if not self.abort_until_idle():
                        raise PulseStreamError(
                            "Arduino tetap aktif setelah ERROR:ALREADY_RUNNING."
                        ) from exc
                    raise PulseStreamError(
                        "STREAM_RESET:ALREADY_RUNNING"
                    ) from exc
                if run_attempt == 1 or run_attempt % 5 == 0:
                    self.log(f"STREAM_RUN belum dijawab ({run_attempt}); kirim ulang: {reason}")
                time.sleep(RETRY_DELAY_S)
        if self.stop_event.is_set():
            raise PulseStreamError("Streaming dihentikan pengguna.")

        completed = 0
        last_done_sequence = -1
        stream_done = False
        while completed < total_blocks or not stream_done:
            timeout_s = 90.0 if completed >= total_blocks and not stream_done else max(
                3.0, plan.block_ms / 1000.0 * 4.0
            )
            try:
                line = self._readline(timeout_s)
            except PulseStreamError as exc:
                if str(exc) == "Streaming dihentikan pengguna.":
                    raise
                state = self._query_state(0.8)
                active = int(state.get("ACTIVE", "-1"))
                if int(state.get("RUN", "0")) or int(state.get("XYWAIT", "0")):
                    self.log("Jawaban progres hilang tetapi STATE masih aktif; RUN tetap ditunggu.")
                    continue
                if active >= total_blocks:
                    self.log("STATE memastikan seluruh blok selesai meskipun STREAM_DONE hilang.")
                    completed = total_blocks
                    stream_done = True
                    continue
                self.log(
                    f"Jawaban sementara hilang (ACTIVE={active}); STATE belum selesai, "
                    "stream tetap dipantau dan sequence aktif akan dikirim ulang bila diminta."
                )
                if 0 <= active < total_blocks:
                    self._send_block(self._plan_block(plan, active))
                    next_sequence = max(next_sequence, active + 1)
                time.sleep(RETRY_DELAY_S)
                continue
            if line.startswith("BLOCK_DONE:"):
                sequence = int(line.split(":", 1)[1])
                if 0 <= sequence < total_blocks and sequence > last_done_sequence:
                    for released_sequence in range(last_done_sequence + 1, sequence + 1):
                        self._release_plan_block(plan, released_sequence)
                    last_done_sequence = sequence
                    completed = max(completed, sequence + 1)
                    if progress:
                        progress(completed, total_blocks)
                if next_sequence < total_blocks:
                    self._send_block(self._plan_block(plan, next_sequence))
                    next_sequence += 1
            elif line.startswith("BUFFER_WAIT:"):
                # BUFFER_WAIT membawa sequence yang sedang ditunggu Arduino.
                # Jangan menunggu BLOCK_DONE yang mungkin hilang; isi langsung
                # sequence tersebut (dan gap sebelumnya) dari host.
                try:
                    waiting_sequence = int(line.split(":", 1)[1])
                except (ValueError, IndexError) as exc:
                    raise PulseStreamError(f"Format BUFFER_WAIT tidak valid: {line}") from exc
                if 0 <= waiting_sequence < total_blocks:
                    self.log(
                        f"BUFFER_WAIT:{waiting_sequence}; mengisi ulang blok yang ditunggu."
                    )
                    if waiting_sequence < next_sequence:
                        # Host sudah pernah mengirim sequence ini, tetapi
                        # Arduino menyatakan slotnya kosong; kirim ulang.
                        self._send_block(self._plan_block(plan, waiting_sequence))
                    else:
                        while next_sequence <= waiting_sequence and next_sequence < total_blocks:
                            self._send_block(self._plan_block(plan, next_sequence))
                            next_sequence += 1
            elif line.startswith("BUFFER_RESUMED:"):
                self.log(f"RX <- {line}; stream Arduino dilanjutkan.")
            elif line == "STREAM_DONE":
                stream_done = True
            elif line.startswith(("ERROR:", "NAK:", "BUFFER_UNDERRUN", "STREAM_ABORTED")):
                if line == "STREAM_ABORTED":
                    raise PulseStreamError(line)
                if line.startswith((
                    "ERROR:NOT_HOMED", "ERROR:Z_ENCODER",
                    "ERROR:Z_ENCODER_READ", "ERROR:Z_STALL",
                    "ERROR:Z_WRONG_DIRECTION",
                )):
                    raise PulseStreamError(line)
                if line.startswith("ERROR:ALREADY_RUNNING"):
                    if self.abort_until_idle():
                        raise PulseStreamError("STREAM_RESET:ALREADY_RUNNING")
                    raise PulseStreamError(line)
                # Error asynchronous dari frame lama tidak boleh membuat
                # trajektori panjang berhenti. STATE menentukan sequence yang
                # masih aktif; blok tersebut dikirim ulang dengan CRC/sequence
                # yang sama sehingga tidak menggandakan gerakan.
                self.log(f"RX error sementara: {line}; verifikasi STATE dan lanjutkan.")
                state = self._query_state(0.8)
                active = int(state.get("ACTIVE", "-1"))
                if active >= total_blocks:
                    completed = total_blocks
                    stream_done = True
                elif 0 <= active < total_blocks:
                    self._send_block(self._plan_block(plan, active))
                    next_sequence = max(next_sequence, active + 1)
                time.sleep(RETRY_DELAY_S)

        self.log("Trajektori selesai tanpa underrun buffer.")
