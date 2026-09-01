from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from serial.tools import list_ports

from pulse_planner import load_trajectory_csv, plan_polyline, plan_sinusoidal
from stream_transport import PulseStreamer


class PulseBlockApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Pulse Block Streaming - XY + Servo")
        self.root.geometry("1220x780")
        self.plan = None
        self.streamer = None
        self.loaded_points = None
        self._build_ui()
        self.generate_sine()

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)
        controls = ttk.LabelFrame(outer, text="Konfigurasi", padding=10)
        controls.pack(side="left", fill="y", padx=(0, 10))
        plot_frame = ttk.Frame(outer)
        plot_frame.pack(side="left", fill="both", expand=True)

        self.vars = {
            "port": tk.StringVar(), "axis": tk.StringVar(value="X"),
            "amplitude": tk.StringVar(value="20"), "period": tk.StringVar(value="4"),
            "cycles": tk.StringVar(value="2"),
            "speed_x": tk.StringVar(value="80"), "speed_y": tk.StringVar(value="80"),
            "speed_z": tk.StringVar(value="20"), "z_deg_cm": tk.StringVar(value="15"),
            "ppc_x": tk.StringVar(value="2800"), "ppc_y": tk.StringVar(value="1400"),
            "slice": tk.StringVar(value="10"),
            "block": tk.StringVar(value="20"), "status": tk.StringVar(value="Siap"),
        }

        row = 0
        ttk.Label(controls, text="Port Arduino").grid(row=row, column=0, sticky="w")
        self.port_box = ttk.Combobox(controls, textvariable=self.vars["port"], width=18)
        self.port_box.grid(row=row, column=1, sticky="ew")
        ttk.Button(controls, text="Perbarui", command=self.refresh_ports).grid(row=row, column=2, padx=(5, 0))
        row += 1
        ttk.Separator(controls).grid(row=row, column=0, columnspan=3, sticky="ew", pady=9)
        row += 1
        fields = [
            ("Sumbu sinus", "axis"), ("Amplitudo (mm)", "amplitude"),
            ("Periode (s)", "period"), ("Jumlah siklus", "cycles"),
            ("Kecepatan X (mm/s)", "speed_x"), ("Kecepatan Y (mm/s)", "speed_y"),
            ("Kecepatan Z (mm/s)", "speed_z"), ("Kalibrasi Z (deg/cm)", "z_deg_cm"),
            ("Pulsa/cm X", "ppc_x"),
            ("Pulsa/cm Y", "ppc_y"),
            ("Slice (ms)", "slice"), ("Blok (ms)", "block"),
        ]
        for label, key in fields:
            ttk.Label(controls, text=label).grid(row=row, column=0, sticky="w", pady=2)
            if key == "axis":
                widget = ttk.Combobox(controls, textvariable=self.vars[key], values=("X", "Y"), state="readonly", width=10)
            else:
                widget = ttk.Entry(controls, textvariable=self.vars[key], width=14)
            widget.grid(row=row, column=1, columnspan=2, sticky="ew", pady=2)
            row += 1

        ttk.Button(controls, text="Generate Sinus", command=self.generate_sine).grid(row=row, column=0, columnspan=3, sticky="ew", pady=(10, 3)); row += 1
        ttk.Button(controls, text="Load Trajektori CSV", command=self.load_csv).grid(row=row, column=0, columnspan=3, sticky="ew", pady=3); row += 1
        ttk.Button(controls, text="Kirim ke Arduino", command=self.start_stream).grid(row=row, column=0, columnspan=3, sticky="ew", pady=(10, 3)); row += 1
        ttk.Button(controls, text="STOP", command=self.stop_stream).grid(row=row, column=0, columnspan=3, sticky="ew", pady=3); row += 1
        self.progress = ttk.Progressbar(controls, maximum=100)
        self.progress.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(10, 2)); row += 1
        ttk.Label(controls, textvariable=self.vars["status"], wraplength=285).grid(row=row, column=0, columnspan=3, sticky="w"); row += 1
        self.log_text = tk.Text(controls, width=42, height=12, state="disabled")
        self.log_text.grid(row=row, column=0, columnspan=3, sticky="nsew", pady=(8, 0))
        controls.rowconfigure(row, weight=1)
        controls.columnconfigure(1, weight=1)

        self.figure = Figure(figsize=(8.5, 7), dpi=100)
        self.axes = self.figure.subplots(3, 1, sharex=True)
        self.figure.subplots_adjust(left=0.10, right=0.98, top=0.95, bottom=0.08, hspace=0.24)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.refresh_ports()

    def parameters(self):
        return {
            "steps_per_cm": tuple(float(self.vars[key].get()) for key in ("ppc_x", "ppc_y")),
            "z_deg_per_cm": float(self.vars["z_deg_cm"].get()),
            "slice_ms": int(self.vars["slice"].get()),
            "block_ms": int(self.vars["block"].get()),
        }

    def generate_sine(self) -> None:
        try:
            self.loaded_points = None
            self.plan = plan_sinusoidal(
                axis=self.vars["axis"].get(), amplitude_mm=float(self.vars["amplitude"].get()),
                period_s=float(self.vars["period"].get()), cycles=float(self.vars["cycles"].get()),
                **self.parameters(),
            )
            self._show_plan("Sinusoidal")
        except Exception as exc:
            messagebox.showerror("Parameter tidak valid", str(exc))

    def load_csv(self) -> None:
        filename = filedialog.askopenfilename(title="Pilih CSV trajektori", filetypes=[("CSV", "*.csv")])
        if not filename:
            return
        try:
            self.loaded_points = load_trajectory_csv(filename)
            self.plan = plan_polyline(
                self.loaded_points,
                max_speed_mm_s=(
                    float(self.vars["speed_x"].get()),
                    float(self.vars["speed_y"].get()),
                    float(self.vars["speed_z"].get()),
                ),
                **self.parameters(),
            )
            self._show_plan(Path(filename).name)
        except Exception as exc:
            messagebox.showerror("CSV gagal dimuat", str(exc))

    def _show_plan(self, source: str) -> None:
        plan = self.plan
        for axis in self.axes:
            axis.clear(); axis.grid(True, alpha=0.25)
        labels = ("X", "Y", "Z target")
        for idx, label in enumerate(labels):
            self.axes[0].plot(plan.times_s, plan.desired_mm[:, idx], label=f"Target {label}")
            self.axes[0].plot(plan.times_s, plan.planned_mm[:, idx], "--", linewidth=0.9, label=f"Pulsa {label}")
        rate_t = plan.times_s[1:]
        for idx, label in enumerate(("X", "Y")):
            self.axes[1].plot(rate_t, plan.pulse_rate[:, idx], label=label)
        error_um = (plan.planned_mm - plan.desired_mm) * 1000.0
        for idx, label in enumerate(labels):
            self.axes[2].plot(plan.times_s, error_um[:, idx], label=label)
        for boundary in range(0, len(plan.blocks) + 1):
            t = boundary * plan.block_ms / 1000.0
            if t <= plan.duration_s:
                for axis in self.axes: axis.axvline(t, color="0.8", linewidth=0.5)
        self.axes[0].set_ylabel("Posisi (mm)"); self.axes[0].legend(ncol=3, fontsize=8)
        self.axes[1].set_ylabel("Pulsa/s"); self.axes[1].legend(ncol=3, fontsize=8)
        self.axes[2].set_ylabel("Galat (µm)"); self.axes[2].set_xlabel("Waktu (s)")
        self.figure.suptitle(f"{source} — {len(plan.blocks)} blok, durasi {plan.duration_s:.2f} s")
        self.canvas.draw_idle()
        self.vars["status"].set(f"Plan siap: {len(plan.blocks)} blok / {len(plan.delta_steps)} slice")

    def start_stream(self) -> None:
        if self.plan is None: return
        port = self.vars["port"].get().strip()
        if not port:
            messagebox.showwarning("Port belum dipilih", "Pilih port Arduino terlebih dahulu."); return
        self.progress["value"] = 0
        threading.Thread(target=self._stream_worker, args=(port,), daemon=True).start()

    def _stream_worker(self, port: str) -> None:
        try:
            self.streamer = PulseStreamer(port, log=self.log)
            self.streamer.stream(self.plan, progress=self._progress)
            self.root.after(0, self.vars["status"].set, "Trajektori selesai")
        except Exception as exc:
            self.log(f"ERROR: {exc}")
            self.root.after(0, self.vars["status"].set, f"Gagal: {exc}")
        finally:
            if self.streamer: self.streamer.close()

    def stop_stream(self) -> None:
        if self.streamer: self.streamer.stop()
        self.vars["status"].set("STOP dikirim")

    def _progress(self, done: int, total: int) -> None:
        self.root.after(0, self.progress.configure, {"value": done * 100.0 / total})
        self.root.after(0, self.vars["status"].set, f"Blok selesai {done}/{total}")

    def log(self, message: str) -> None:
        def append():
            self.log_text.configure(state="normal")
            self.log_text.insert("end", message + "\n")
            self.log_text.see("end")
            self.log_text.configure(state="disabled")
        self.root.after(0, append)

    def refresh_ports(self) -> None:
        ports = [item.device for item in list_ports.comports()]
        self.port_box["values"] = ports
        if ports and not self.vars["port"].get(): self.vars["port"].set(ports[0])


if __name__ == "__main__":
    root = tk.Tk()
    PulseBlockApp(root)
    root.mainloop()
