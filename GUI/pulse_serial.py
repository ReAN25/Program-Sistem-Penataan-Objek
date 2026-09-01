"""Komunikasi GUI dengan firmware Pulse Block XYZ-angle dan servo ber-CRC."""

from __future__ import annotations

import binascii
import json
import threading
import time
from datetime import datetime

import numpy as np
import serial
import serial.tools.list_ports

from pulse_planner import plan_polyline
from stream_transport import PulseStreamError, PulseStreamer


class PulseSerialMixin:
    BAUDRATE = 250000
    # Durasi aksi GRIP pada firmware. Selama interval ini firmware mengirim
    # pulsa PWM 50 Hz, lalu mengirim GOK; RUN baru boleh meneruskan segmen
    # setelah ACK tersebut diterima.
    SERVO_GRIP_ACTIVE_S = 1.0
    MOTOR_STEPS_PER_REV = 6400.0
    Z_MOTOR_TO_OUTPUT_REDUCTION = 20.0
    Z_MOTOR_STEPS_PER_OUTPUT_DEG = (
        MOTOR_STEPS_PER_REV * Z_MOTOR_TO_OUTPUT_REDUCTION / 360.0
    )

    def is_arduino_connected(self):
        return self.ser is not None and self.ser.is_open

    def _load_position_memory(self):
        path = getattr(self, "position_state_path", None)
        if path is None or not path.exists():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            position = np.asarray(payload["position_mm"], dtype=float)
            if position.shape != (3,) or not np.all(np.isfinite(position)):
                raise ValueError("format posisi tidak valid")
            position = np.clip(position, [0.0, 0.0, 0.0], [
                self.workspace_x_max, self.workspace_y_max, self.workspace_z_max
            ])
            self._hardware_pos = position.copy()
            self.current_pos = position.copy()
            self.log_msg(
                f"Posisi terakhir dimuat -> X:{position[0]:.3f} "
                f"Y:{position[1]:.3f} Z:{position[2]:.3f} mm"
            )
            return True
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.log_msg(f"Memori posisi tidak dapat dibaca: {exc}")
            return False

    def _save_position_memory(self):
        path = getattr(self, "position_state_path", None)
        if path is None:
            return
        position = np.asarray(self._hardware_pos, dtype=float)
        payload = {
            "position_mm": [float(value) for value in position],
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "reference": "xy_open_loop_counter_z_as5600_multiturn",
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            temporary.replace(path)
        except OSError as exc:
            self.log_msg(f"Memori posisi gagal disimpan: {exc}")

    def _steps_per_cm(self):
        px, py = float(self.pulse_x_entry.get()), float(self.pulse_y_entry.get())
        if px <= 0 or py <= 0:
            raise ValueError("Kalibrasi pulsa/cm X dan Y harus lebih besar dari nol.")
        return px, py

    def _z_deg_per_cm(self):
        value = float(self.z_deg_per_cm_entry.get())
        if value <= 0:
            raise ValueError("Kalibrasi Z derajat/cm harus lebih besar dari nol.")
        return value

    def _grip_release_z_offset(self):
        """Baca offset aksi terbaru dari GUI sebelum setiap RUN/konfigurasi.

        Offset hanya dipakai Python untuk membuat gerak turun-naik gripper,
        sehingga tidak dikirim sebagai konfigurasi Arduino. Sebelumnya nilai
        baru hanya tersalin ketika tombol *Set Config* ditekan; akibatnya RUN
        dapat memakai nilai lama dan gerakan offset terlihat tidak terjadi.
        """
        entry = getattr(self, "grip_release_z_offset_entry", None)
        value = (
            float(entry.get())
            if entry is not None
            else float(getattr(self, "grip_release_z_offset", 1.0))
        )
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("Offset gripper Z harus berupa angka >= 0 mm.")
        workspace_max = float(getattr(self, "workspace_z_max", value))
        value = min(value, max(0.0, workspace_max))
        self.grip_release_z_offset = value
        return value

    def _speed_mm_s(self):
        speed = np.asarray(
            [
                float(self.speed_x_entry.get()),
                float(self.speed_y_entry.get()),
                float(self.speed_z_entry.get()),
            ],
            dtype=float,
        )
        if np.any(speed <= 0):
            raise ValueError("Kecepatan X, Y, dan Z harus lebih besar dari nol.")
        return speed

    def _timer_speed_capacity_mm_s(self):
        pulse_capacity_hz = 1_000_000.0 / 20.0
        xy_steps_per_mm = np.asarray(self._steps_per_cm()) / 10.0
        # Encoder membaca poros output; rasio gearbox hanya dipakai untuk
        # menghitung pulsa motor, bukan untuk mengubah sudut encoder.
        z_steps_per_mm = (self._z_deg_per_cm() / 10.0) * self.Z_MOTOR_STEPS_PER_OUTPUT_DEG
        return pulse_capacity_hz / np.asarray([
            xy_steps_per_mm[0], xy_steps_per_mm[1], z_steps_per_mm * 2.0
        ])

    def _send_z_config(self):
        if not self.is_arduino_connected():
            return True
        with self.serial_command_lock:
            deg_per_cm = self._z_deg_per_cm()
            speed_z = float(self._speed_mm_s()[2])
            command = f"ZCFG:{deg_per_cm:.4f}:{speed_z:.4f}\n"
            attempt = 0
            while self.is_arduino_connected() and not getattr(self, "stop_requested", False):
                attempt += 1
                try:
                    self.log_msg(f"TX -> {command.strip()} (percobaan {attempt})")
                    self.ser.write(command.encode("ascii"))
                    self.ser.flush()
                    deadline = time.monotonic() + 2.0
                    while time.monotonic() < deadline:
                        line = self.ser.readline().decode("ascii", errors="replace").strip()
                        if not line:
                            continue
                        self.log_msg(f"RX <- {line}")
                        if line.startswith("ZCFG_OK:"):
                            return True
                        if line.startswith(("ERROR:", "NAK:")):
                            raise RuntimeError(line)
                    raise TimeoutError("Arduino tidak mengonfirmasi konfigurasi Z.")
                except (OSError, serial.SerialException, TimeoutError) as exc:
                    if attempt == 1 or attempt % 5 == 0:
                        self.log_msg(f"Konfigurasi Z belum dijawab; kirim ulang: {exc}")
                    time.sleep(0.12)
            raise RuntimeError("Pengiriman konfigurasi Z dihentikan sebelum Arduino merespons.")

    def _sync_remembered_position_to_arduino(self):
        units = np.rint(
            np.asarray(self._hardware_pos[:2]) * (np.asarray(self._steps_per_cm()) / 10.0)
        ).astype(np.int64)
        command = f"SET_POSITION:{units[0]}:{units[1]}\n"
        attempt = 0
        while self.is_arduino_connected() and not getattr(self, "stop_requested", False):
            attempt += 1
            try:
                self.ser.write(command.encode("ascii"))
                self.ser.flush()
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline:
                    line = self.ser.readline().decode("ascii", errors="replace").strip()
                    if not line:
                        continue
                    self.log_msg(f"RX <- {line}")
                    if line.startswith("POSITION_SET:"):
                        return True
                    if line.startswith("ERROR:"):
                        raise RuntimeError(line)
                raise TimeoutError("Arduino tidak mengonfirmasi SET_POSITION XY.")
            except (OSError, serial.SerialException, TimeoutError) as exc:
                if attempt == 1 or attempt % 5 == 0:
                    self.log_msg(f"SET_POSITION belum dijawab; kirim ulang ({attempt}): {exc}")
                time.sleep(0.12)
        raise RuntimeError("Pengiriman SET_POSITION dihentikan sebelum Arduino merespons.")

    def _sync_servo_sequence_from_arduino(self):
        """Lanjutkan sequence servo setelah GUI restart tanpa mereset Arduino."""
        attempt = 0
        while self.is_arduino_connected() and not getattr(self, "stop_requested", False):
            attempt += 1
            try:
                self.ser.write(b"STATE?\n")
                self.ser.flush()
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline:
                    line = self.ser.readline().decode("ascii", errors="replace").strip()
                    if not line:
                        continue
                    self.log_msg(f"RX <- {line}")
                    if line.startswith(("STATE:", "STATUS:")):
                        values = self._parse_status_line(line)
                        last_sequence = int(values.get("SSEQ", "-1"))
                        self.servo_action_sequence = max(0, last_sequence)
                        if "ZDECI" in values:
                            self._hardware_pos[2] = float(values["ZDECI"]) / self._z_deg_per_cm()
                            self.current_pos[2] = self._hardware_pos[2]
                        return
                raise TimeoutError("Arduino tidak menjawab sinkronisasi sequence servo.")
            except (OSError, serial.SerialException, TimeoutError) as exc:
                if attempt == 1 or attempt % 5 == 0:
                    self.log_msg(f"STATE sequence belum dijawab; kirim ulang ({attempt}): {exc}")
                time.sleep(0.12)
        raise RuntimeError("Sinkronisasi sequence dihentikan sebelum Arduino merespons.")

    def update_status(self, is_connected, message):
        if hasattr(self, "status_bar"):
            self.status_bar.config(
                text=message,
                bg="#ccffcc" if is_connected else "#ffcccc",
                fg="#006600" if is_connected else "#cc0000",
            )

    def update_connect_button(self):
        if hasattr(self, "btn_connect"):
            text = "Disconnect Arduino" if self.is_arduino_connected() else "Connect Arduino"
            self.btn_connect.config(text=text, bg="SystemButtonFace", fg="black")

    def toggle_connection(self):
        if self.is_running:
            self.log_msg("Tidak bisa mengubah koneksi saat RUN aktif.")
            return
        self.disconnect_arduino() if self.is_arduino_connected() else self.connect_arduino()

    def _candidate_ports(self):
        ports = list(serial.tools.list_ports.comports())
        preferred = [item for item in ports if any(
            name in item.description.upper() for name in ("ARDUINO", "CH340", "USB")
        )]
        return preferred + [item for item in ports if item not in preferred]

    def connect_arduino(self):
        if self.is_running:
            return False
        for port in self._candidate_ports():
            candidate = None
            try:
                candidate = serial.Serial(
                    port.device, self.BAUDRATE, timeout=0.02, write_timeout=1
                )
                deadline, next_hello, received = time.monotonic() + 5.0, time.monotonic() + 0.8, []
                while time.monotonic() < deadline:
                    line = candidate.readline().decode("ascii", errors="replace").strip()
                    if line:
                        received.append(line)
                    if line.startswith("PULSE_BLOCK_READY"):
                        required = (
                            ":SEQ_BITS:32:", ":BLOCK_SLICES:2:", ":SLOTS:32:", ":TICK_US:20:", ":AXES:XYZ_ANGLE:",
                            ":Y_DIR_INVERTED:1:",
                            ":Z_HIGH_US:20:", ":Z_AS5600:1:", ":Z_ENCODER_OUTPUT:1:",
                            ":Z_SAMPLE_MS:20:",
                            ":REDUCTION:20:",
                            ":SERVO_FIXED:1:", ":SERVO_GRIP_MS:1000:", ":STATE_RECOVERY:1",
                        )
                        if not line.startswith("PULSE_BLOCK_READY:32:") or not all(
                            token in line for token in required
                        ):
                            received.append("firmware tidak cocok: perlu XYZ-angle + AS5600 + servo fixed")
                            break
                        self.ser = candidate
                        self._sync_remembered_position_to_arduino()
                        self._sync_servo_sequence_from_arduino()
                        self._send_z_config()
                        self.connection_enabled = True
                        self.update_connect_button()
                        self.update_status(True, f"Pulse Block XYZ terhubung: {port.device}")
                        self.log_msg(
                            f"Arduino Pulse Block XYZ-angle + Servo aktif di {port.device} "
                            f"@ {self.BAUDRATE} baud | encoder Z = poros output, "
                            "target sudut tidak dikali 20 | arah Y dibalik"
                        )
                        return True
                    if time.monotonic() >= next_hello:
                        candidate.write(b"HELLO\n")
                        candidate.flush()
                        next_hello = time.monotonic() + 0.5
                candidate.close()
                detail = ", ".join(received[-3:]) if received else "tanpa respons"
                self.log_msg(f"Handshake {port.device} gagal: {detail}")
            except (OSError, serial.SerialException, RuntimeError, ValueError) as exc:
                if candidate is not None:
                    try:
                        candidate.close()
                    except Exception:
                        pass
                self.ser = None
                self.log_msg(f"Port {port.device} tidak dapat dibuka: {exc}")
        self.connection_enabled = False
        self.update_connect_button()
        self.update_status(False, "Mode Offline - firmware Pulse Block XYZ tidak ditemukan")
        return False

    def disconnect_arduino(self, reason="Mode Offline - Arduino disconnected"):
        self.connection_enabled = False
        if self.ser:
            try:
                self.ser.close()
            except (OSError, serial.SerialException):
                pass
        self.ser = None
        self.manual_waiting_ack = False
        self.homing_waiting_ack = False
        self.homing_active_seen = False
        self.update_connect_button()
        self.update_status(False, reason)
        self.log_msg("Arduino disconnected. GUI kembali ke mode offline.")

    def _retry_home_if_start_not_confirmed(self):
        if not self.is_arduino_connected() or not self.homing_waiting_ack:
            return
        if self.homing_active_seen:
            return
        if time.monotonic() - float(getattr(self, "homing_last_tx", 0.0)) < 1.0:
            return
        try:
            self.ser.write(b"HOME\n")
            self.ser.flush()
            self.homing_last_tx = time.monotonic()
            self.homing_retry_count = int(getattr(self, "homing_retry_count", 0)) + 1
            self.log_msg(
                f"TX -> HOME ulang sampai Arduino merespons "
                f"(percobaan {self.homing_retry_count}, microdelay 30 us)"
            )
        except (OSError, serial.SerialException) as exc:
            self.log_msg(f"HOME belum terkirim; mencoba lagi: {exc}")

    def connection_monitor(self):
        if self.is_arduino_connected() and not self.is_running and not self.manual_waiting_ack:
            try:
                while self.ser.in_waiting:
                    line = self.ser.readline().decode("ascii", errors="replace").strip()
                    if not line:
                        continue
                    self.log_msg(f"RX <- {line}")
                    if line in ("BUTTON:HOME", "HOMING_START"):
                        self.homing_waiting_ack = True
                        self.homing_active_seen = True
                        self.log_msg("Homing Arduino sedang berlangsung; kontrol gerak dikunci.")
                    elif line.startswith("PULSE_BLOCK_READY"):
                        # READY yang muncul lagi setelah koneksi terbentuk berarti
                        # Arduino baru reset dan seluruh referensi posisi hilang.
                        self.homing_waiting_ack = False
                        self._hardware_pos[:] = 0.0
                        self.current_pos[:] = 0.0
                        self.log_msg(
                            "RESET ARDUINO TERDETEKSI: posisi dibatalkan. Lakukan HOME ulang."
                        )
                    elif line == "HOMING_END":
                        self._hardware_pos[:] = 0.0
                        self.current_pos[:] = 0.0
                        self._save_position_memory()
                        self.homing_waiting_ack = False
                        self.homing_active_seen = False
                        self.homing_retry_count = 0
                        self.update_plot()
                    elif line.startswith(("HOMING_FAILED", "HOMING_ABORTED")):
                        self.homing_waiting_ack = False
                        self.homing_active_seen = False
                    elif line.startswith(("STATE:", "STATUS:")) and self.homing_waiting_ack:
                        values = self._parse_status_line(line)
                        if int(values.get("HOMING", "0")):
                            self.homing_active_seen = True
                            self.log_msg("STATE mengonfirmasi Arduino masih homing; HOME tidak digandakan.")
                    elif line == "BUTTON:RUN":
                        self.root.after(0, self.run_sequence)
                self._retry_home_if_start_not_confirmed()
            except (OSError, serial.SerialException):
                self.disconnect_arduino("Koneksi Pulse Block XYZ terputus")
        self.root.after(80, self.connection_monitor)

    def update_config(self):
        try:
            steps, speed = self._steps_per_cm(), self._speed_mm_s()
            z_deg_per_cm = self._z_deg_per_cm()
            grip_offset = self._grip_release_z_offset()
            self.log_msg(
                f"Config diterapkan | X/Y pulsa/cm={steps}, Z={z_deg_per_cm:g} deg/cm | "
                f"kecepatan X={speed[0]:g}, Y={speed[1]:g}, Z={speed[2]:g} mm/s | "
                f"offset aksi gripper Z={grip_offset:g} mm | "
                "servo Grip=30°, Release=95° | Z closed-loop AS5600 output "
                "(rasio gearbox hanya untuk pulsa motor)"
            )
            if self.is_arduino_connected() and (self.is_running or self.manual_waiting_ack):
                self.log_msg(
                    "Nilai konfigurasi tersimpan di GUI dan akan dikirim sebelum gerakan berikutnya."
                )
            elif self.is_arduino_connected():
                self._send_z_config()
        except (ValueError, RuntimeError, serial.SerialException) as exc:
            self.log_msg(f"Config tidak valid: {exc}")

    def _send_current_config_to_arduino(self):
        try:
            self._steps_per_cm()
            self._z_deg_per_cm()
            self._speed_mm_s()
            # Offset adalah parameter gerak Python, tetapi harus selalu dibaca
            # dari entry terbaru sebelum RUN/homing agar titik aksi tidak
            # menggunakan nilai lama dari konfigurasi sebelumnya.
            self._grip_release_z_offset()
            if self.is_arduino_connected():
                self._send_z_config()
            return True
        except (ValueError, RuntimeError, serial.SerialException) as exc:
            self.log_msg(f"Config tidak valid: {exc}")
            return False

    def homing(self):
        if self.is_running:
            self.log_msg("Tidak bisa Homing saat RUN aktif.")
            return
        if not self.is_arduino_connected():
            self.current_pos[:] = self._hardware_pos[:] = 0.0
            self.update_plot()
            self.log_msg("Homing offline: posisi XY grafik kembali ke nol.")
            return
        try:
            with self.serial_command_lock:
                self.stop_requested = False
                if not self._send_current_config_to_arduino():
                    return
                self.homing_waiting_ack = True
                self.homing_active_seen = False
                self.homing_retry_count = 0
                self.homing_last_tx = time.monotonic()
                self.ser.write(b"HOME\n")
                self.ser.flush()
                self.log_msg("TX -> HOME (limit X, Y, dan Z; microdelay 30 us)")
        except (OSError, serial.SerialException):
            self.disconnect_arduino("Koneksi terputus saat homing")

    def _make_streamer(self):
        streamer = PulseStreamer("existing", baudrate=self.BAUDRATE, log=self.log_msg)
        streamer.serial = self.ser
        return streamer

    @staticmethod
    def _parse_status_line(status):
        values = {}
        for token in status.split(":")[1:]:
            if "=" in token:
                key, value = token.split("=", 1)
                values[key] = value
        if not {"X", "Y", "ZDECI"}.issubset(values):
            raise RuntimeError(f"Format STATE/STATUS Arduino tidak lengkap: {status}")
        return values

    def _read_hardware_position(self):
        with self.serial_command_lock:
            attempt = 0
            while (
                (attempt == 0 or self.is_arduino_connected())
                and not getattr(self, "stop_requested", False)
            ):
                attempt += 1
                try:
                    streamer = self._make_streamer()
                    streamer._write("STATE?\n")
                    state = streamer._wait_for(
                        lambda value: value.startswith(("STATE:", "STATUS:", "PULSE_BLOCK_READY")),
                        2.0,
                    )
                    if state.startswith("PULSE_BLOCK_READY"):
                        self._hardware_pos[:] = 0.0
                        self.current_pos[:] = 0.0
                        raise RuntimeError(
                            "Arduino baru mengalami reset; referensi posisi hilang. Lakukan HOME ulang."
                        )
                    values = self._parse_status_line(state)
                    units = np.asarray([int(values["X"]), int(values["Y"])], dtype=np.int64)
                    self._hardware_pos[:2] = units.astype(float) / (np.asarray(self._steps_per_cm()) / 10.0)
                    self._hardware_pos[2] = float(values["ZDECI"]) / self._z_deg_per_cm()
                    self._save_position_memory()
                    return units, self._hardware_pos.copy(), values
                except (OSError, serial.SerialException, PulseStreamError) as exc:
                    if str(exc) == "Streaming dihentikan pengguna.":
                        raise
                    if attempt == 1 or attempt % 5 == 0:
                        self.log_msg(f"STATE belum dijawab; kirim ulang ({attempt}): {exc}")
                    time.sleep(0.12)
            raise RuntimeError("STATE dihentikan sebelum Arduino merespons.")

    def _ensure_arduino_idle_for_manual(self):
        units, position, values = self._read_hardware_position()
        busy = any(int(values.get(key, "0")) for key in ("RUN", "XYRUN", "XYWAIT"))
        if busy:
            self.log_msg("Gerakan lama masih aktif; STOP dikirim sebelum kontrol manual.")
            streamer = self._make_streamer()
            attempt = 0
            while (
                (attempt == 0 or self.is_arduino_connected())
                and not getattr(self, "stop_requested", False)
            ):
                attempt += 1
                try:
                    streamer._write("STOP\n")
                    streamer._wait_for(lambda value: value == "STREAM_ABORTED", 1.0)
                    break
                except (OSError, serial.SerialException, PulseStreamError) as exc:
                    if attempt == 1 or attempt % 5 == 0:
                        self.log_msg(f"STOP belum dijawab; kirim ulang ({attempt}): {exc}")
                    time.sleep(0.12)
            units, position, values = self._read_hardware_position()
        if int(values.get("HOME", "0")) != 1:
            raise RuntimeError(
                "Arduino belum homing (HOME=0). Tekan tombol HOME dan tunggu HOMING_END "
                "sebelum menjalankan gerak manual."
            )
        if int(values.get("ZENC", "0")) != 1:
            raise RuntimeError("Encoder Z tidak siap (ZENC=0); gerakan dibatalkan.")
        return units, position

    def _recover_manual_stream_failure(self):
        if not self.is_arduino_connected():
            return
        try:
            _units, position, values = self._read_hardware_position()
            if any(int(values.get(key, "0")) for key in ("RUN", "XYRUN", "XYWAIT")):
                self.log_msg("Arduino masih aktif saat pemulihan manual; STOP dikirim.")
                if not self._stop_stream_and_wait("pemulihan manual"):
                    return
                _units, position, values = self._read_hardware_position()
            self.current_pos = position.copy()
            self.root.after(0, self.update_plot)
        except Exception as recovery_error:
            self.log_msg(f"Pemeriksaan STATE pemulihan gagal: {recovery_error}")

    def _stop_stream_and_wait(self, context=""):
        """Hentikan stream tersisa dan pastikan firmware sudah idle."""
        if not self.is_arduino_connected():
            return False
        try:
            with self.serial_command_lock:
                streamer = self._make_streamer()
                if streamer.abort_until_idle():
                    self.log_msg(
                        "Arduino idle setelah STOP pemulihan"
                        + (f" ({context})." if context else ".")
                    )
                    return True
            self.log_msg("STOP pemulihan belum terkonfirmasi; Arduino jangan digerakkan.")
        except (OSError, serial.SerialException, PulseStreamError, RuntimeError) as exc:
            self.log_msg(f"STOP pemulihan gagal{f' ({context})' if context else ''}: {exc}")
        return False

    def _stream_points_blocking(self, points, progress=None):
        points = np.asarray(points, dtype=float)
        start = np.asarray(getattr(self, "_hardware_pos", np.zeros(3)), dtype=float)
        if np.linalg.norm(points[0] - start) > 1e-6:
            points = np.vstack((start, points))
        requested_speed = self._speed_mm_s()
        timer_capacity = self._timer_speed_capacity_mm_s()
        over_capacity = requested_speed > timer_capacity
        if np.any(over_capacity):
            self.log_msg(
                f"Input X/Y={requested_speed.tolist()} mm/s diterima tanpa batas GUI; "
                f"kapasitas timer X/Y sekitar {np.round(timer_capacity, 2).tolist()} mm/s. "
                "Timeline disesuaikan otomatis agar tidak terjadi NAK:RATE."
            )
        retry_count = 0
        while True:
            xyz_length = np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
            if len(points) < 2 or xyz_length <= 1e-9:
                self._hardware_pos = np.asarray(points[-1], dtype=float).copy()
                self._save_position_memory()
                return None
            plan = plan_polyline(
                points, max_speed_mm_s=requested_speed,
                steps_per_cm=self._steps_per_cm(), z_deg_per_cm=self._z_deg_per_cm(),
                slice_ms=self.pulse_slice_ms,
                block_ms=self.pulse_block_ms,
            )
            try:
                if self.is_arduino_connected():
                    streamer = self._make_streamer()
                    self._active_streamer = streamer
                    try:
                        streamer.stream(plan, progress=progress)
                    finally:
                        if getattr(self, "_active_streamer", None) is streamer:
                            self._active_streamer = None
                    actual_units, actual_position, _values = self._read_hardware_position()
                    expected_units = np.rint(points[-1, :2] * plan.steps_per_mm).astype(np.int64)
                    difference = actual_units - expected_units
                    expected_z_deci = int(round(points[-1, 2] * self._z_deg_per_cm()))
                    actual_z_deci = int(round(actual_position[2] * self._z_deg_per_cm()))
                    self.log_msg(
                        f"VERIFY step XY | expected={expected_units.tolist()} "
                        f"actual={actual_units.tolist()} selisih={difference.tolist()} | "
                        f"Z target={expected_z_deci / 10.0:.1f}°, aktual={actual_z_deci / 10.0:.1f}°"
                    )
                    if np.any(difference != 0):
                        raise RuntimeError("Counter pulsa XY tidak sesuai target.")
                    if abs(actual_z_deci - expected_z_deci) > 5:
                        raise RuntimeError("Sudut Z di luar toleransi 0,5°.")
                    self._hardware_pos[:2] = actual_position[:2]
                self._hardware_pos = np.asarray(points[-1], dtype=float).copy()
                self._save_position_memory()
                return plan
            except (PulseStreamError, OSError, serial.SerialException, RuntimeError) as exc:
                if (
                    isinstance(exc, RuntimeError)
                    and not isinstance(exc, PulseStreamError)
                    and not str(exc).startswith((
                        "Counter pulsa XY", "Sudut Z"
                    ))
                ):
                    raise
                reason = str(exc)
                z_fault = reason.startswith((
                    "ERROR:Z_STALL", "ERROR:Z_WRONG_DIRECTION",
                    "ERROR:Z_ENCODER_READ",
                ))
                # Kesalahan konfigurasi port/timing memang tidak dapat
                # diperbaiki dengan mengulang paket yang sama. Semua error
                # transport, CRC, slot, state, dan kehilangan ACK diproses di
                # bawah dengan retry tanpa batas berbasis posisi aktual.
                if reason.startswith((
                    "Port serial belum terhubung",
                    "ERROR:BEGIN_TIMING",
                )):
                    raise
                if not self.is_arduino_connected() or getattr(self, "stop_requested", False):
                    raise
                retry_count += 1
                if z_fault:
                    self.log_msg(
                        "Pemulihan Z tanpa batas aktif; blok akan dikirim ulang "
                        "sampai Arduino mengonfirmasi posisi."
                    )
                self.log_msg(
                    f"Blok gagal ({exc}); blok yang sudah selesai tidak dikirim ulang. "
                    f"Membaca STATE dan mengirim ulang sisa blok dari posisi aktual "
                    f"(percobaan {retry_count})."
                )
                _units, actual_position, values = self._read_hardware_position()
                if int(values.get("HOME", "0")) != 1:
                    self.log_msg(
                        "Arduino belum HOME; gerakan tidak dipaksakan. "
                        "Retry menunggu HOMING_END lalu melanjutkan dari posisi aktual."
                    )
                    time.sleep(0.5)
                    continue
                if int(values.get("ZENC", "0")) != 1:
                    self.log_msg("Encoder Z belum sehat; retry menunggu respons encoder.")
                    time.sleep(0.2)
                    continue
                if any(int(values.get(key, "0")) for key in ("RUN", "XYRUN", "XYWAIT")):
                    self.log_msg("Arduino masih menjalankan blok lama; retry menunggu tanpa duplikasi.")
                    time.sleep(0.2)
                    continue
                if np.max(np.abs(actual_position - points[-1])) <= 0.05:
                    self._hardware_pos = actual_position.copy()
                    self._save_position_memory()
                    return plan
                # Cari proyeksi posisi aktual pada polyline, bukan sekadar
                # vertex terdekat. Ini mencegah satu blok yang sudah terlewati
                # dikirim ulang dan menjaga titik awal retry tepat pada posisi
                # aktual Arduino.
                if len(points) >= 2:
                    segment = np.diff(points, axis=0)
                    length_sq = np.einsum("ij,ij->i", segment, segment)
                    valid = length_sq > 1e-12
                    projection = np.repeat(actual_position[None, :], len(segment), axis=0)
                    if np.any(valid):
                        relative = actual_position[None, :] - points[:-1]
                        fraction = np.zeros(len(segment), dtype=float)
                        fraction[valid] = np.einsum(
                            "ij,ij->i", relative[valid], segment[valid]
                        ) / length_sq[valid]
                        fraction = np.clip(fraction, 0.0, 1.0)
                        projection = points[:-1] + fraction[:, None] * segment
                        distances_to_path = np.linalg.norm(
                            projection - actual_position[None, :], axis=1
                        )
                        nearest_segment = int(np.argmin(distances_to_path))
                        nearest_distance = float(distances_to_path[nearest_segment])
                        nearest = nearest_segment + (
                            1 if fraction[nearest_segment] > 1e-6 else 0
                        )
                    else:
                        nearest = 0
                        nearest_distance = float(np.linalg.norm(points[0] - actual_position))
                else:
                    nearest = 0
                    nearest_distance = float(np.linalg.norm(points[0] - actual_position))
                if len(points) > 2 and nearest_distance > 10.0:
                    raise RuntimeError(
                        f"Posisi aktual terlalu jauh dari jalur ({nearest_distance:.1f} mm); "
                        "retry otomatis dihentikan demi keselamatan."
                    )
                remaining = points[nearest:]
                if not len(remaining):
                    remaining = points[-1:]
                if nearest_distance > 1e-6 or not np.allclose(
                    actual_position, remaining[0], atol=1e-6
                ):
                    remaining = np.vstack((actual_position, remaining))
                self.log_msg(
                    f"Retry dimulai pada titik jalur {nearest}/{len(points) - 1}; "
                    f"jarak posisi aktual ke jalur {nearest_distance:.2f} mm."
                )
                points = remaining

    def send_target(self, x, y, z, quiet=False, **_kwargs):
        target = np.array([x, y, z], dtype=float)
        if not self.is_arduino_connected():
            self.current_pos = target
            self.update_plot()
            if not quiet:
                self.log_msg(f"OFFLINE -> X:{x:.1f} Y:{y:.1f} Z:{z:.1f}")
            return True
        if self.homing_waiting_ack:
            self.log_msg("Tunggu HOMING_END sebelum menjalankan gerak manual.")
            return False
        if self.manual_waiting_ack:
            return False
        self.stop_requested = False
        self.manual_waiting_ack = True

        def worker():
            try:
                with self.serial_command_lock:
                    actual_units, actual_position = self._ensure_arduino_idle_for_manual()
                    # Terapkan nilai Z terbaru sesaat sebelum membentuk stream.
                    # RLock membuat transaksi bersarang ini tetap aman.
                    self._send_z_config()
                    target_units = np.rint(
                        target[:2] * (np.asarray(self._steps_per_cm()) / 10.0)
                    ).astype(np.int64)
                    self._hardware_pos = actual_position.copy()
                    z_reached = abs(target[2] - actual_position[2]) <= (
                        0.5 / (self._z_deg_per_cm() / 10.0)
                    )
                    if np.array_equal(target_units, actual_units) and z_reached:
                        self.current_pos = self._hardware_pos = target.copy()
                        self._save_position_memory()
                        self.root.after(0, self.update_plot)
                        self.log_msg("Target XYZ sudah tercapai; tidak ada pulsa dikirim.")
                        return
                    self._stream_points_blocking([actual_position, target])
                    self.current_pos = target
                    self.root.after(0, self.update_plot)
                    self.log_msg(f"Manual XYZ selesai -> X:{x:.1f} Y:{y:.1f} Z:{z:.1f}")
            except Exception as exc:
                self.log_msg(f"Gerak manual gagal: {exc}")
                self._recover_manual_stream_failure()
            finally:
                self.manual_waiting_ack = False

        threading.Thread(target=worker, daemon=True).start()
        return True

    def _send_grip_command(self, command):
        command = command.upper()
        if command not in ("GRIP", "RELEASE"):
            return False
        angle = 30 if command == "GRIP" else 95
        if not self.is_arduino_connected():
            self.log_msg(f"OFFLINE -> {command} servo {angle}°")
            return True

        self.servo_action_sequence = (int(getattr(self, "servo_action_sequence", 0)) + 1) & 0xFFFF
        sequence = self.servo_action_sequence
        payload = command
        crc = binascii.crc_hqx(payload.encode("ascii"), 0xFFFF)
        framed = f"SERVO:{sequence}:{crc:04X}:{payload}\n".encode("ascii")
        expected_ack = "GOK" if command == "GRIP" else "ROK"
        desired_grip = 1 if command == "GRIP" else 0
        attempt = 0

        while not bool(getattr(self, "stop_requested", False)):
            attempt += 1
            try:
                self.ser.write(framed)
                self.ser.flush()
                self.log_msg(f"TX -> SERVO {command} ({angle}° tetap) seq={sequence} percobaan={attempt}")
                deadline = time.monotonic() + 2.5
                while time.monotonic() < deadline:
                    line = self.ser.readline().decode("ascii", errors="replace").strip()
                    if not line:
                        continue
                    self.log_msg(f"RX <- {line}")
                    if line == expected_ack:
                        if command == "GRIP":
                            self.servo_grip_angle_deg = angle
                            self.log_msg(
                                "GRIP selesai: servo aktif selama "
                                f"{self.SERVO_GRIP_ACTIVE_S:.1f} detik; trajektori dilanjutkan."
                            )
                        else:
                            self.servo_release_angle_deg = angle
                        return True
                    if line.startswith((f"NAK:SERVO:{sequence}", "ERROR:SERVO")):
                        break

                self.ser.write(b"STATE?\n")
                self.ser.flush()
                state_deadline = time.monotonic() + 1.0
                state_received = False
                while time.monotonic() < state_deadline:
                    state = self.ser.readline().decode("ascii", errors="replace").strip()
                    if not state:
                        continue
                    self.log_msg(f"RX <- {state}")
                    if not state.startswith(("STATE:", "STATUS:")):
                        continue
                    state_received = True
                    values = self._parse_status_line(state)
                    if int(values.get("GRIP", "-1")) == desired_grip:
                        if int(values.get("SERVO", "0")):
                            self.log_msg("STATE: servo masih bergerak; menunggu tanpa mengulang aksi.")
                            time.sleep(0.15)
                            break
                        self.log_msg("STATE memastikan posisi servo sudah sesuai meskipun ACK hilang.")
                        return True
                    if int(values.get("RUN", "0")):
                        self.log_msg("STATE: XY masih bergerak; servo belum diulang.")
                        time.sleep(0.15)
                    else:
                        self.log_msg("STATE idle dan servo belum sesuai; paket sama dikirim ulang.")
                    break
                if not state_received:
                    self.log_msg(
                        f"STATE servo belum dijawab; paket SERVO {command} yang sama akan dikirim ulang."
                    )
                    time.sleep(0.12)
            except (OSError, serial.SerialException, PulseStreamError) as exc:
                if not self.is_arduino_connected():
                    self.disconnect_arduino(f"Koneksi terputus saat {command}: {exc}")
                    return False
                self.log_msg(f"SERVO {command} belum dijawab; paket yang sama dikirim ulang: {exc}")
                time.sleep(0.12)
        return False

    def stop_arduino_motion(self):
        self.stop_requested = True
        self.is_running = False
        active_streamer = getattr(self, "_active_streamer", None)
        if active_streamer is not None:
            active_streamer.stop_event.set()
        if self.is_arduino_connected():
            try:
                self.ser.write(b"STOP\n")
                self.ser.flush()
                self.log_msg("TX -> STOP")
            except serial.SerialException:
                self.disconnect_arduino("Koneksi terputus saat STOP")
