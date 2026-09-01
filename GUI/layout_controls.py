import threading
import tkinter as tk


class LayoutControlsMixin:
    def button_style(self, font=("Arial", 9, "bold"), height=1):
        return {
            "bg": "SystemButtonFace",
            "fg": "black",
            "activebackground": "SystemButtonFace",
            "activeforeground": "black",
            "disabledforeground": "SystemDisabledText",
            "font": font,
            "height": height,
            "relief": tk.RAISED,
            "bd": 2,
        }

    def create_layout(self):
        self.status_bar = tk.Label(self.root, text="🔴 Menunggu Koneksi Arduino...", bd=1, relief=tk.SUNKEN, anchor=tk.W, bg="#ffcccc", font=("Arial", 9, "bold"))
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.main_frame = tk.Frame(self.root, bg="white")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.left_frame = tk.Frame(self.main_frame, bg="white", highlightbackground="black", highlightthickness=2)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        self.plot_frame = tk.Frame(self.left_frame, bg="white")
        self.plot_frame.pack(fill=tk.BOTH, expand=True)

        self.right_frame = tk.Frame(self.main_frame, bg="white")
        self.right_frame.pack(side=tk.RIGHT, fill=tk.Y)

        self.create_controls()

    def create_controls(self):
        controls_columns = tk.Frame(self.right_frame, bg="white")
        controls_columns.pack(fill=tk.BOTH, expand=True)
        controls_columns.grid_columnconfigure(0, weight=1, uniform="right_controls")
        controls_columns.grid_columnconfigure(1, weight=1, uniform="right_controls")
        controls_columns.grid_rowconfigure(0, weight=1)

        left_controls = tk.Frame(controls_columns, bg="white")
        left_controls.grid(row=0, column=0, sticky="nsew", padx=(0, 4))

        self.jog_column = tk.Frame(controls_columns, bg="white")
        self.jog_column.grid(row=0, column=1, sticky="nsew", padx=(4, 0))

        manual_frame = tk.Frame(left_controls, bg="white")
        manual_frame.pack(fill=tk.X, pady=(0, 3))
        tk.Label(manual_frame, text="Manual Control", font=("Arial", 9, "bold"), bg="white").pack(fill=tk.X, pady=(0, 2))

        manual_outer_frame = tk.Frame(manual_frame, bg="#A0A0A0", pady=5, padx=6)
        manual_outer_frame.pack(fill=tk.X)

        tk.Label(manual_outer_frame, text="Jog Control", font=("Arial", 9, "bold"), bg="#A0A0A0", fg="black").pack(fill=tk.X, pady=(0, 2))

        pos_outer_frame = tk.Frame(manual_outer_frame, bg="#A0A0A0")
        pos_outer_frame.pack(fill=tk.X)

        pos_frame = tk.Frame(pos_outer_frame, bg="#A0A0A0")
        pos_frame.pack(fill=tk.X, padx=6) 

        pos_frame.grid_columnconfigure(0, weight=1)
        pos_frame.grid_columnconfigure(1, weight=1)
        pos_frame.grid_columnconfigure(2, weight=1)
        pos_frame.grid_rowconfigure(0, weight=1, minsize=38)
        pos_frame.grid_rowconfigure(1, weight=1, minsize=38)

        btn_move_style = self.button_style(font=("Arial", 10, "bold"), height=2)

        moves = [
            ("X+", 0, 0, 1, 0, 0), ("Y+", 0, 1, 0, 1, 0), ("Z+", 0, 2, 0, 0, 1),
            ("X-", 1, 0, -1, 0, 0), ("Y-", 1, 1, 0, -1, 0), ("Z-", 1, 2, 0, 0, -1)
        ]

        for txt, r, c, dx, dy, dz in moves:
            cmd = lambda dx=dx, dy=dy, dz=dz: self.move_pos(
                dx * self.get_step_size(),
                dy * self.get_step_size(),
                dz * self.get_step_size()
            )
            tk.Button(pos_frame, text=txt, command=cmd, **btn_move_style).grid(
                row=r, column=c, padx=2, pady=1, sticky="nsew"
            )

        tk.Label(
            manual_outer_frame,
            text="Position Gripper",
            bg="#A0A0A0",
            font=("Arial", 9, "bold"),
            fg="black",
        ).pack(fill=tk.X, pady=(4, 1))

        gripper_pos_outer = tk.Frame(manual_outer_frame, bg="#A0A0A0", pady=2)
        gripper_pos_outer.pack(fill=tk.X, pady=(0, 2))

        pos_boxes_frame = tk.Frame(gripper_pos_outer, bg="#A0A0A0")
        pos_boxes_frame.pack(fill=tk.X)

        pos_boxes_frame.grid_columnconfigure(0, weight=1)
        pos_boxes_frame.grid_columnconfigure(1, weight=1)
        pos_boxes_frame.grid_columnconfigure(2, weight=1)

        fx = tk.Frame(pos_boxes_frame, bg="#A0A0A0")
        fx.grid(row=0, column=0, sticky="ew", padx=(0, 2))
        tk.Label(fx, text="X", bg="#A0A0A0", font=("Arial", 9, "bold"), fg="black").pack(side=tk.LEFT)
        self.pos_x_entry = tk.Entry(fx, width=5, justify="center", font=("Arial", 9), fg="black")
        self.pos_x_entry.insert(0, "0.0")
        self.pos_x_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))
        self.pos_x_entry.bind('<Return>', self.move_to_typed_pos)

        fy = tk.Frame(pos_boxes_frame, bg="#A0A0A0")
        fy.grid(row=0, column=1, sticky="ew", padx=(2, 2))
        tk.Label(fy, text="Y", bg="#A0A0A0", font=("Arial", 9, "bold"), fg="black").pack(side=tk.LEFT)
        self.pos_y_entry = tk.Entry(fy, width=5, justify="center", font=("Arial", 9), fg="black")
        self.pos_y_entry.insert(0, "0.0")
        self.pos_y_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))
        self.pos_y_entry.bind('<Return>', self.move_to_typed_pos)

        fz = tk.Frame(pos_boxes_frame, bg="#A0A0A0")
        fz.grid(row=0, column=2, sticky="ew", padx=(2, 0))
        tk.Label(fz, text="Z", bg="#A0A0A0", font=("Arial", 9, "bold"), fg="black").pack(side=tk.LEFT)
        self.pos_z_entry = tk.Entry(fz, width=5, justify="center", font=("Arial", 9), fg="black")
        self.pos_z_entry.insert(0, "0.0")
        self.pos_z_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))
        self.pos_z_entry.bind('<Return>', self.move_to_typed_pos)

        tk.Button(
            manual_outer_frame,
            text="Homing",
            command=self.homing,
            **self.button_style()
        ).pack(fill=tk.X, pady=(3, 0))

        servo_manual_box = tk.Frame(manual_outer_frame, bg="#A0A0A0", pady=4)
        servo_manual_box.pack(fill=tk.X, pady=(3, 0))
        tk.Label(
            servo_manual_box,
            text="Manual Gripper",
            bg="#A0A0A0",
            font=("Arial", 8, "bold"),
            fg="black",
        ).pack(fill=tk.X)

        servo_btn_frame = tk.Frame(servo_manual_box, bg="#A0A0A0")
        servo_btn_frame.pack(fill=tk.X, pady=(2, 0))
        servo_btn_frame.grid_columnconfigure(0, weight=1)
        servo_btn_frame.grid_columnconfigure(1, weight=1)

        tk.Button(
            servo_btn_frame,
            text="Grip",
            command=self.manual_servo_grip,
            **self.button_style(font=("Arial", 8, "bold"))
        ).grid(row=0, column=0, sticky="ew", padx=(0, 2))
        tk.Button(
            servo_btn_frame,
            text="Release",
            command=self.manual_servo_release,
            **self.button_style(font=("Arial", 8, "bold"))
        ).grid(row=0, column=1, sticky="ew", padx=(2, 0))

        tk.Label(
            servo_manual_box,
            text="Grip 30°  |  Release 95°",
            bg="#A0A0A0",
            font=("Arial", 8),
            fg="black",
        ).pack(fill=tk.X, pady=(3, 0))

        tk.Label(self.jog_column, text="Configuration", font=("Arial", 9, "bold"), bg="white").pack(fill=tk.X, pady=(1, 1))

        config_outer_frame = tk.Frame(self.jog_column, bg="#A0A0A0", pady=4, padx=6)
        config_outer_frame.pack(fill=tk.X, pady=(0, 2))

        step_frame = tk.Frame(config_outer_frame, bg="#A0A0A0")
        step_frame.pack(pady=(0, 2), fill=tk.X)

        tk.Label(step_frame, text="Step Size:", font=("Arial", 9), bg="#A0A0A0", fg= "black").pack(side=tk.LEFT)
        self.step_entry = tk.Entry(step_frame, width=10, font=("Arial", 9), justify="center")
        self.step_entry.pack(side=tk.LEFT, padx=2)
        self.step_entry.insert(0, "2")
        tk.Label(step_frame, text="cm", font=("Arial", 9), fg= "black", bg="#A0A0A0").pack(side=tk.LEFT, padx=(0, 2))

        config_frame = tk.Frame(config_outer_frame, bg="#A0A0A0")
        config_frame.pack(fill=tk.X)
        config_frame.grid_columnconfigure(0, weight=1)

        tk.Label(config_frame, text="Calibration", bg="#A0A0A0", font=("Arial", 9, "bold"), fg= "black").grid(row=0, column=0, pady=(0, 2))
        
        pulse_inputs = tk.Frame(config_frame, bg="#A0A0A0")
        pulse_inputs.grid(row=1, column=0, sticky="ew")
        
        pulse_inputs.grid_columnconfigure(0, weight=1)
        pulse_inputs.grid_columnconfigure(1, weight=1)

        frame_x = tk.Frame(pulse_inputs, bg="#A0A0A0")
        frame_x.grid(row=0, column=0, sticky="ew", padx=(0, 2)) 
        tk.Label(frame_x, text="X p/cm", bg="#A0A0A0", font=("Arial", 9, "bold"), fg= "black").pack(side=tk.LEFT)
        self.pulse_x_entry = tk.Entry(frame_x, width=5, justify="center", font=("Arial", 9))
        self.pulse_x_entry.insert(0, "2800",) 
        self.pulse_x_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))
        
        frame_y = tk.Frame(pulse_inputs, bg="#A0A0A0")
        frame_y.grid(row=0, column=1, sticky="ew", padx=(2, 0))
        tk.Label(frame_y, text="Y p/cm", bg="#A0A0A0", font=("Arial", 9, "bold"), fg= "black").pack(side=tk.LEFT)
        self.pulse_y_entry = tk.Entry(frame_y, width=5, justify="center", font=("Arial", 9))
        self.pulse_y_entry.insert(0, "1400") 
        self.pulse_y_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

        frame_z = tk.Frame(pulse_inputs, bg="#A0A0A0")
        frame_z.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(2, 0))
        tk.Label(frame_z, text="Z deg/cm", bg="#A0A0A0", font=("Arial", 9, "bold"), fg="black").pack(side=tk.LEFT)
        self.z_deg_per_cm_entry = tk.Entry(frame_z, width=6, justify="center", font=("Arial", 9))
        self.z_deg_per_cm_entry.insert(0, "15")
        self.z_deg_per_cm_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))
        
        # Kecepatan tiap sumbu dalam mm/s; Z dikonversi menjadi target sudut.
        speed_frame = tk.Frame(config_frame, bg="#A0A0A0")
        speed_frame.grid(row=2, column=0, sticky="ew", pady=(4, 2))
        speed_frame.grid_columnconfigure(0, weight=1)
        tk.Label(speed_frame, text="Kecepatan X/Y/Z (mm/s)", bg="#A0A0A0", font=("Arial", 9, "bold"), fg="black").grid(row=0, column=0, columnspan=3, pady=(0, 2))
        speed_frame.grid_columnconfigure(0, weight=1)
        speed_frame.grid_columnconfigure(1, weight=1)
        speed_frame.grid_columnconfigure(2, weight=1)
        speed_x_box = tk.Frame(speed_frame, bg="#A0A0A0")
        speed_x_box.grid(row=1, column=0, sticky="ew", padx=(0, 2))
        tk.Label(speed_x_box, text="X", bg="#A0A0A0", font=("Arial", 9, "bold"), fg="black").pack(side=tk.LEFT)
        self.speed_x_entry = tk.Entry(speed_x_box, width=6, justify="center", font=("Arial", 9))
        self.speed_x_entry.insert(0, "80")
        self.speed_x_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0))
        speed_y_box = tk.Frame(speed_frame, bg="#A0A0A0")
        speed_y_box.grid(row=1, column=1, sticky="ew", padx=(2, 0))
        tk.Label(speed_y_box, text="Y", bg="#A0A0A0", font=("Arial", 9, "bold"), fg="black").pack(side=tk.LEFT)
        self.speed_y_entry = tk.Entry(speed_y_box, width=6, justify="center", font=("Arial", 9))
        self.speed_y_entry.insert(0, "80")
        self.speed_y_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0))
        speed_z_box = tk.Frame(speed_frame, bg="#A0A0A0")
        speed_z_box.grid(row=1, column=2, sticky="ew", padx=(2, 0))
        tk.Label(speed_z_box, text="Z", bg="#A0A0A0", font=("Arial", 9, "bold"), fg="black").pack(side=tk.LEFT)
        self.speed_z_entry = tk.Entry(speed_z_box, width=6, justify="center", font=("Arial", 9))
        self.speed_z_entry.insert(0, "20")
        self.speed_z_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0))

        # Offset gerak aksi gripper pada sumbu Z.
        z_pid_frame = tk.Frame(config_frame, bg="#A0A0A0")
        z_pid_frame.grid(row=3, column=0, sticky="ew", pady=(4, 2))
        z_pid_frame.grid_columnconfigure(0, weight=1)

        tk.Label(z_pid_frame, text="Offset Gripper Z Axis", bg="#A0A0A0", font=("Arial", 9, "bold"), fg="black").grid(row=0, column=0, pady=(0, 2))

        z_offset_frame = tk.Frame(z_pid_frame, bg="#A0A0A0")
        z_offset_frame.grid(row=1, column=0, sticky="ew", pady=(2, 0))
        tk.Label(
            z_offset_frame,
            text="Offset",
            bg="#A0A0A0",
            font=("Arial", 9, "bold"),
            fg="black",
        ).pack(side=tk.LEFT)
        self.grip_release_z_offset_entry = tk.Entry(
            z_offset_frame,
            width=6,
            justify="center",
            font=("Arial", 9),
        )
        self.grip_release_z_offset_entry.insert(0, f"{self.grip_release_z_offset:.1f}")
        self.grip_release_z_offset_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))
        tk.Label(z_offset_frame, text="mm", bg="#A0A0A0", font=("Arial", 9), fg="black").pack(side=tk.LEFT, padx=(2, 0))
        # -------------------------------------

        self.btn_set_config = tk.Button(
            config_frame, text="Set Config", command=self.update_config, 
            **self.button_style()
        )
        self.btn_set_config.grid(row=4, column=0, pady=(4, 0), sticky="ew")

        tk.Label(
            self.jog_column,
            text="Obstacle",
            font=("Arial", 9, "bold"),
            bg="white",
        ).pack(fill=tk.X, pady=(1, 1))

        obstacle_outer = tk.Frame(self.jog_column, bg="#A0A0A0", pady=4, padx=6)
        obstacle_outer.pack(fill=tk.X, pady=(0, 2))

        tk.Label(
            obstacle_outer,
            text="Position Obstacle",
            bg="#A0A0A0",
            fg="black",
            font=("Arial", 8, "bold"),
        ).pack(fill=tk.X, pady=(0, 2))

        obstacle_inputs = tk.Frame(obstacle_outer, bg="#A0A0A0")
        obstacle_inputs.pack(fill=tk.X)
        for column in range(3):
            obstacle_inputs.grid_columnconfigure(column, weight=1)

        self.fixed_obstacle_corner_entries = []
        for column, (axis, default) in enumerate((("X", "250"), ("Y", "250"), ("Z", "0"))):
            axis_frame = tk.Frame(obstacle_inputs, bg="#A0A0A0")
            axis_frame.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 2, 0))
            tk.Label(axis_frame, text=axis, bg="#A0A0A0", fg="black", font=("Arial", 9, "bold")).pack(side=tk.LEFT)
            entry = tk.Entry(axis_frame, width=5, justify="center", font=("Arial", 9))
            entry.insert(0, default)
            entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))
            self.fixed_obstacle_corner_entries.append(entry)

        tk.Label(
            obstacle_outer,
            text="Size: Length X+ | Width Y+ | Height Z+",
            bg="#A0A0A0",
            fg="black",
            font=("Arial", 8, "bold"),
        ).pack(fill=tk.X, pady=(3, 2))

        obstacle_size_inputs = tk.Frame(obstacle_outer, bg="#A0A0A0")
        obstacle_size_inputs.pack(fill=tk.X)
        for column in range(3):
            obstacle_size_inputs.grid_columnconfigure(column, weight=1)

        self.fixed_obstacle_size_entries = []
        for column, (axis, default) in enumerate((("X+", "50"), ("Y+", "50"), ("Z+", "50"))):
            axis_frame = tk.Frame(obstacle_size_inputs, bg="#A0A0A0")
            axis_frame.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 2, 0))
            tk.Label(axis_frame, text=axis, bg="#A0A0A0", fg="black", font=("Arial", 8, "bold")).pack(side=tk.LEFT)
            entry = tk.Entry(axis_frame, width=5, justify="center", font=("Arial", 9))
            entry.insert(0, default)
            entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))
            self.fixed_obstacle_size_entries.append(entry)

        self.fixed_obstacle_size_entries[-1].bind("<Return>", self.add_fixed_obstacle_from_gui)

        obstacle_buttons = tk.Frame(obstacle_outer, bg="#A0A0A0")
        obstacle_buttons.pack(fill=tk.X, pady=(3, 0))
        tk.Button(
            obstacle_buttons,
            text="Add Obstacle",
            command=self.add_fixed_obstacle_from_gui,
            **self.button_style(font=("Arial", 8, "bold")),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        tk.Button(
            obstacle_buttons,
            text="Remove Last",
            command=self.remove_last_fixed_obstacle_from_gui,
            **self.button_style(font=("Arial", 8, "bold")),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

        self.fixed_obstacle_count_var = tk.StringVar(value="")

        tk.Label(self.jog_column, text="Log", font=("Arial", 9, "bold"), bg="white").pack(fill=tk.X, pady=(1, 0))
        self.log_text = tk.Text(self.jog_column, height=18, width=36, font=("Consolas", 9), bg="#f4f4f4", wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=2)
        self.log_text.bind("<Key>", lambda e: "break")

        self.planning_column = tk.Frame(left_controls, bg="white")
        self.planning_column.pack(fill=tk.BOTH, expand=True)

        btn_act_style = self.button_style()
        btn_clear_style = self.button_style()

        tk.Label(self.planning_column, text="Trajectory Planning", font=("Arial", 9, "bold"), bg="white").pack(fill=tk.X, pady=(0, 1))
        planning_outer = tk.Frame(self.planning_column, bg="#A0A0A0", padx=6, pady=4)
        planning_outer.pack(fill=tk.X, pady=(0, 2))
        planning_outer.grid_columnconfigure(0, weight=1, uniform="planning")
        planning_outer.grid_columnconfigure(1, weight=1, uniform="planning")

        tk.Button(planning_outer, text="Add Waypoint", command=self.add_waypoint, **btn_act_style).grid(row=0, column=0, columnspan=2, sticky="ew", padx=2, pady=1)

        self.btn_grip = tk.Button(
            planning_outer,
            text="Grip",
            command=self.set_grip_point,
            **self.button_style()
        )
        self.btn_grip.grid(row=1, column=0, sticky=tk.EW, padx=(0, 2), pady=1)

        self.btn_release = tk.Button(
            planning_outer,
            text="Release",
            command=self.add_release_point,
            **self.button_style()
        )
        self.btn_release.grid(row=1, column=1, sticky=tk.EW, padx=(2, 0), pady=1)

        tk.Button(
            planning_outer,
            text="Delete Last Waypoint",
            command=self.remove_last_planning_point,
            **btn_clear_style
        ).grid(row=2, column=0, columnspan=2, sticky="ew", padx=2, pady=(3, 1))

        tk.Button(
            planning_outer,
            text="Clear All Waypoints",
            command=self.clear_all_waypoints,
            **btn_clear_style
        ).grid(row=3, column=0, columnspan=2, sticky="ew", padx=2, pady=1)

        tk.Button(
            planning_outer,
            text="Clear Trajectory",
            command=self.clear_data,
            **btn_clear_style
        ).grid(row=4, column=0, columnspan=2, sticky="ew", padx=2, pady=1)

        self.optimasi_indicator_label = tk.Label(
            planning_outer,
            text="OPTIMIZATION OFF",
            bg="#C62828",
            fg="white",
            font=("Arial", 9, "bold"),
            relief=tk.SUNKEN,
            padx=4,
            pady=2,
        )
        self.optimasi_indicator_label.grid(row=5, column=0, columnspan=2, sticky="ew", padx=2, pady=(3, 1))

        tk.Button(
            planning_outer,
            text="Generate Trajectory",
            command=self.generate_direct_trajectory,
            **self.button_style()
        ).grid(row=6, column=0, sticky=tk.EW, padx=(0, 2), pady=1)

        self.btn_optimasi = tk.Button(
            planning_outer,
            text="Generate With FHO",
            command=self.run_optimization,
            **self.button_style()
        )
        self.btn_optimasi.grid(row=6, column=1, sticky=tk.EW, padx=(2, 0), pady=1)

        tk.Label(self.planning_column, text="Automatic Control", font=("Arial", 9, "bold"), bg="white").pack(fill=tk.X, pady=(1, 1))
        auto_outer = tk.Frame(self.planning_column, bg="#A0A0A0", padx=6, pady=4)
        auto_outer.pack(fill=tk.X, pady=(0, 2))
        auto_outer.grid_columnconfigure(0, weight=1, uniform="auto")
        auto_outer.grid_columnconfigure(1, weight=1, uniform="auto")

        self.btn_cycle = tk.Button(
            auto_outer,
            text="CYCLE OFF",
            command=self.toggle_cycle_mode,
            **self.button_style()
        )
        self.btn_cycle.grid(row=0, column=0, sticky=tk.EW, padx=(0, 2), pady=1)

        tk.Button(
            auto_outer,
            text="STOP",
            command=self.stop_arduino_motion,
            **self.button_style()
        ).grid(row=0, column=1, sticky=tk.EW, padx=(2, 0), pady=1)

        tk.Button(auto_outer, text="RUN", command=self.run_sequence, **btn_act_style).grid(row=1, column=0, columnspan=2, sticky="ew", padx=2, pady=(3, 1))

        system_frame = tk.Frame(self.planning_column, bg="white")
        system_frame.pack(fill=tk.X, pady=(0, 2))
        system_frame.grid_columnconfigure(0, weight=1, uniform="system")
        system_frame.grid_columnconfigure(1, weight=1, uniform="system")

        self.btn_connect = tk.Button(
            system_frame,
            text="Connect Arduino",
            command=self.toggle_connection,
            **self.button_style()
        )
        self.btn_connect.grid(row=0, column=0, columnspan=2, sticky=tk.EW, padx=2, pady=1)

        tk.Button(
            system_frame,
            text="Save Trajektory",
            command=self.save_plot_image,
            **self.button_style()
        ).grid(row=1, column=0, columnspan=2, sticky="ew", padx=2, pady=1)

        tk.Button(
            system_frame,
            text="Load Trajektory",
            command=self.load_csv_pattern,
            **self.button_style()
        ).grid(row=2, column=0, columnspan=2, sticky="ew", padx=2, pady=1)

        tk.Button(
            system_frame,
            text="Reset View",
            command=self.reset_plot_perspective,
            **self.button_style()
        ).grid(row=3, column=0, columnspan=2, sticky="ew", padx=2, pady=1)

    def get_step_size(self):
        try:
            return max(0.1, float(self.step_entry.get()))*10
        except:
            return 1

    def log_msg(self, message):
        if threading.current_thread() is not threading.main_thread():
            self.root.after(0, self.log_msg, message)
            return
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def _invalidate_generated_trajectory_after_obstacle_change(self):
        self.optimized_trajectory = []
        self.optimized_prefix_trajectory = []
        self.optimized_grip_release_segments = []
        self.direct_sequence_mode = False
        self.direct_trajectory_segment_distances_mm = ()
        self.direct_trajectory_distance_mm = None
        self.direct_trajectory_axis_distances_mm = (0.0, 0.0, 0.0)
        self.is_optimasi_on = False

        if hasattr(self, "opt_line"):
            self.opt_line.set_data([], [])
            self.opt_line.set_3d_properties([])
        if hasattr(self, "_clear_fho_epoch_artists"):
            self._clear_fho_epoch_artists()
        if hasattr(self, "_clear_grip_release_line_artists"):
            self._clear_grip_release_line_artists()
        if hasattr(self, "_set_optimization_indicator"):
            self._set_optimization_indicator(False)

    def add_fixed_obstacle_from_gui(self, event=None):
        if self.is_running:
            self.log_msg("Cannot add obstacle while trajectory is running.")
            return

        try:
            lower_corner = tuple(float(entry.get()) for entry in self.fixed_obstacle_corner_entries)
            dimensions = tuple(float(entry.get()) for entry in self.fixed_obstacle_size_entries)
        except ValueError:
            self.log_msg("Obstacle rejected: coordinates and dimensions must be numeric.")
            return

        added, message = self.add_fixed_obstacle(lower_corner, dimensions)
        if added:
            self._invalidate_generated_trajectory_after_obstacle_change()
            self.fixed_obstacle_count_var.set("")
            self.canvas.draw_idle()
        self.log_msg(message)

    def remove_last_fixed_obstacle_from_gui(self):
        if self.is_running:
            self.log_msg("Cannot remove obstacle while trajectory is running.")
            return

        removed, message = self.remove_last_user_fixed_obstacle()
        if removed:
            self._invalidate_generated_trajectory_after_obstacle_change()
            self.fixed_obstacle_count_var.set("")
            self.canvas.draw_idle()
        self.log_msg(message)
