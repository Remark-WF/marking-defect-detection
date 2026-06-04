#!/usr/bin/env python3
"""Operator panel for controlling run_pipeline_orangepi.py.

Large-button UI for small displays.
Main panel contains controls only, logs are shown in a separate Console window.
"""

import os
import sys
import queue
import threading
import subprocess
from pathlib import Path
from datetime import datetime

import tkinter as tk
from tkinter import messagebox
from tkinter.scrolledtext import ScrolledText


APP_TITLE = "Label QC Operator Panel"
DEFAULT_SCRIPT = "run_pipeline_orangepi.py"

# Color palette requested by user: blue, white, purple.
COLOR_BG = "#F7F9FF"         # near white
COLOR_PRIMARY = "#0B5ED7"    # blue
COLOR_ACCENT = "#6F42C1"     # purple
COLOR_TEXT = "#FFFFFF"
COLOR_DARK = "#1F2937"
COLOR_IDLE = "#6B7280"
COLOR_RUNNING = "#0EA5E9"


class PipelineController:
    def __init__(self, workdir: Path, script_name: str = DEFAULT_SCRIPT):
        self.workdir = workdir
        self.script_path = self.workdir / script_name
        self.proc = None
        self.reader_thread = None
        self.log_queue = queue.Queue()
        self.stop_reader = threading.Event()

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, mode: str = "control"):
        if self.is_running():
            raise RuntimeError("Pipeline is already running")
        if not self.script_path.exists():
            raise FileNotFoundError(f"Script not found: {self.script_path}")

        env = os.environ.copy()
        env["PIPELINE_MODE"] = mode

        cmd = [sys.executable, str(self.script_path)]
        self.proc = subprocess.Popen(
            cmd,
            cwd=str(self.workdir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env,
        )

        self.stop_reader.clear()
        self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.reader_thread.start()

    def stop(self):
        if self.is_running():
            self.proc.terminate()

    def hard_kill(self):
        if self.is_running():
            self.proc.kill()

    def poll_exit_code(self):
        if self.proc is None:
            return None
        return self.proc.poll()

    def _reader_loop(self):
        assert self.proc is not None
        while not self.stop_reader.is_set():
            line = self.proc.stdout.readline() if self.proc.stdout else ""
            if line:
                self.log_queue.put(line.rstrip("\n"))
                continue
            if self.proc.poll() is not None:
                break

        if self.proc is not None:
            self.log_queue.put(f"[process] exited with code {self.proc.poll()}")


class ConsoleWindow(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Console")
        self.geometry("920x540")
        self.configure(bg=COLOR_BG)

        top = tk.Frame(self, bg=COLOR_BG)
        top.pack(fill=tk.X, padx=12, pady=10)

        tk.Label(
            top,
            text="Pipeline Console",
            bg=COLOR_BG,
            fg=COLOR_ACCENT,
            font=("Segoe UI", 16, "bold"),
        ).pack(side=tk.LEFT)

        tk.Button(
            top,
            text="Clear",
            command=self.clear,
            bg=COLOR_PRIMARY,
            fg=COLOR_TEXT,
            font=("Segoe UI", 12, "bold"),
            relief=tk.FLAT,
            padx=18,
            pady=8,
            activebackground=COLOR_ACCENT,
            activeforeground=COLOR_TEXT,
        ).pack(side=tk.RIGHT)

        self.log_box = ScrolledText(
            self,
            wrap=tk.WORD,
            font=("Consolas", 11),
            bg="#FFFFFF",
            fg=COLOR_DARK,
            insertbackground=COLOR_DARK,
        )
        self.log_box.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

    def append(self, msg: str):
        self.log_box.insert(tk.END, msg + "\n")
        self.log_box.see(tk.END)

    def clear(self):
        self.log_box.delete("1.0", tk.END)


class OperatorPanel(tk.Tk):
    def __init__(self, workdir: Path):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("980x620")
        self.minsize(860, 520)
        self.configure(bg=COLOR_BG)

        self.workdir = workdir
        self.controller = PipelineController(workdir=workdir)
        self.console = None

        self.mode_var = tk.StringVar(value="control")
        self.status_var = tk.StringVar(value="Status: Idle")

        self._build_ui()
        self._set_status("Idle", COLOR_IDLE)
        self.after(120, self._pump_logs)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        root = tk.Frame(self, bg=COLOR_BG)
        root.pack(fill=tk.BOTH, expand=True, padx=16, pady=14)

        header = tk.Frame(root, bg=COLOR_BG)
        header.pack(fill=tk.X)

        tk.Label(
            header,
            text="Label QC Control",
            bg=COLOR_BG,
            fg=COLOR_ACCENT,
            font=("Segoe UI", 24, "bold"),
        ).pack(side=tk.LEFT)

        self.status_label = tk.Label(
            header,
            textvariable=self.status_var,
            bg=COLOR_BG,
            fg=COLOR_IDLE,
            font=("Segoe UI", 16, "bold"),
        )
        self.status_label.pack(side=tk.RIGHT)

        mode_panel = tk.Frame(root, bg=COLOR_BG)
        mode_panel.pack(fill=tk.X, pady=(16, 14))

        tk.Label(
            mode_panel,
            text="Mode:",
            bg=COLOR_BG,
            fg=COLOR_DARK,
            font=("Segoe UI", 16, "bold"),
        ).pack(side=tk.LEFT, padx=(0, 12))

        for val, title in (("control", "CONTROL"), ("calibration", "CALIBRATION")):
            tk.Radiobutton(
                mode_panel,
                text=title,
                value=val,
                variable=self.mode_var,
                bg=COLOR_BG,
                fg=COLOR_DARK,
                selectcolor="#FFFFFF",
                activebackground=COLOR_BG,
                activeforeground=COLOR_ACCENT,
                font=("Segoe UI", 14, "bold"),
                padx=8,
            ).pack(side=tk.LEFT, padx=(0, 18))

        btn_grid = tk.Frame(root, bg=COLOR_BG)
        btn_grid.pack(fill=tk.BOTH, expand=True)

        btn_cfg = {
            "font": ("Segoe UI", 20, "bold"),
            "fg": COLOR_TEXT,
            "relief": tk.FLAT,
            "width": 16,
            "height": 2,
            "activeforeground": COLOR_TEXT,
            "cursor": "hand2",
        }

        self.start_btn = tk.Button(
            btn_grid,
            text="START",
            command=self._on_start,
            bg=COLOR_PRIMARY,
            activebackground=COLOR_ACCENT,
            **btn_cfg,
        )
        self.start_btn.grid(row=0, column=0, padx=12, pady=12, sticky="nsew")

        self.stop_btn = tk.Button(
            btn_grid,
            text="STOP",
            command=self._on_stop,
            bg=COLOR_ACCENT,
            activebackground=COLOR_PRIMARY,
            **btn_cfg,
        )
        self.stop_btn.grid(row=0, column=1, padx=12, pady=12, sticky="nsew")

        self.kill_btn = tk.Button(
            btn_grid,
            text="FORCE STOP",
            command=self._on_kill,
            bg="#4338CA",
            activebackground=COLOR_ACCENT,
            **btn_cfg,
        )
        self.kill_btn.grid(row=1, column=0, padx=12, pady=12, sticky="nsew")

        self.console_btn = tk.Button(
            btn_grid,
            text="OPEN CONSOLE",
            command=self._open_console,
            bg="#2563EB",
            activebackground=COLOR_ACCENT,
            **btn_cfg,
        )
        self.console_btn.grid(row=1, column=1, padx=12, pady=12, sticky="nsew")

        btn_grid.grid_columnconfigure(0, weight=1)
        btn_grid.grid_columnconfigure(1, weight=1)
        btn_grid.grid_rowconfigure(0, weight=1)
        btn_grid.grid_rowconfigure(1, weight=1)

        note = (
            "Calibration mode currently starts the same pipeline process. "
            "This UI is ready for future dedicated calibration state-machine."
        )
        tk.Label(
            root,
            text=note,
            bg=COLOR_BG,
            fg=COLOR_DARK,
            font=("Segoe UI", 11),
            wraplength=920,
            justify=tk.LEFT,
        ).pack(fill=tk.X, pady=(8, 0))

    def _set_status(self, text: str, color: str):
        self.status_var.set(f"Status: {text}")
        self.status_label.configure(fg=color)

    def _timestamp(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    def _log(self, msg: str):
        line = f"[{self._timestamp()}] {msg}"
        if self.console is None or not self.console.winfo_exists():
            self._open_console()
        self.console.append(line)

    def _open_console(self):
        if self.console is None or not self.console.winfo_exists():
            self.console = ConsoleWindow(self)
            self.console.append(f"[{self._timestamp()}] [panel] workdir: {self.workdir}")
            self.console.append(f"[{self._timestamp()}] [panel] script: {self.controller.script_path.name}")
        else:
            self.console.deiconify()
            self.console.lift()
            self.console.focus_force()

    def _on_start(self):
        mode = self.mode_var.get().strip() or "control"
        try:
            self.controller.start(mode=mode)
        except Exception as exc:
            messagebox.showerror("Start error", str(exc))
            return

        self._set_status(f"Running ({mode})", COLOR_RUNNING)
        self._log(f"[panel] started pipeline in mode={mode}")

    def _on_stop(self):
        if not self.controller.is_running():
            self._log("[panel] stop requested, but process is not running")
            return
        self.controller.stop()
        self._log("[panel] graceful stop requested")

    def _on_kill(self):
        if not self.controller.is_running():
            self._log("[panel] force stop requested, but process is not running")
            return
        self.controller.hard_kill()
        self._log("[panel] process killed")

    def _pump_logs(self):
        drained = False
        while True:
            try:
                line = self.controller.log_queue.get_nowait()
            except queue.Empty:
                break
            drained = True
            self._log(line)

        code = self.controller.poll_exit_code()
        if code is None:
            if self.controller.is_running():
                self._set_status("Running", COLOR_RUNNING)
        else:
            self._set_status("Idle", COLOR_IDLE)
            if drained:
                self._log(f"[panel] ready for next run (exit={code})")

        self.after(120, self._pump_logs)

    def _on_close(self):
        if self.controller.is_running():
            if not messagebox.askyesno("Exit", "Pipeline is still running. Stop and close?"):
                return
            self.controller.stop()
        self.destroy()


def main():
    workdir = Path(__file__).resolve().parent
    app = OperatorPanel(workdir=workdir)
    app.mainloop()


if __name__ == "__main__":
    main()