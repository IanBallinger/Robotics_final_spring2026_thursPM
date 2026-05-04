"""Task graph timeline visualizer.

Provides a lightweight Tkinter UI running on a dedicated thread.
Supports:
- Simulation playback mode (slider-driven)
- Live mode with optional "follow latest" toggle and time-travel slider
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any, Dict, List, Optional


class TaskGraphStateVisualizer:
    """Threaded task graph visualizer with timeline playback."""

    def __init__(self, mode: str = "simulate", title: str = "Task Graph Visualizer"):
        mode_norm = str(mode).strip().lower()
        if mode_norm not in {"simulate", "live"}:
            raise ValueError("mode must be one of: simulate, live")

        self.mode = mode_norm
        self.title = title
        self._queue: queue.Queue = queue.Queue()
        self._history: List[Dict[str, Any]] = []
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._closed_event = threading.Event()

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run_ui, daemon=True)
        self._thread.start()

    def push_snapshot(self, snapshot: Dict[str, Any]):
        if snapshot is None:
            return
        self._queue.put({"type": "snapshot", "payload": snapshot})

    def stop(self):
        self._stop_event.set()
        self._queue.put({"type": "stop"})

    def wait_until_closed(self, timeout: Optional[float] = None) -> bool:
        return self._closed_event.wait(timeout=timeout)

    def _run_ui(self):
        try:
            import tkinter as tk
            from tkinter import ttk
        except Exception:
            # GUI is unavailable in this environment.
            self._closed_event.set()
            return

        root = tk.Tk()
        root.title(self.title)
        root.geometry("1180x760")

        top_frame = ttk.Frame(root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)

        mode_label = ttk.Label(top_frame, text=f"mode: {self.mode}")
        mode_label.pack(side=tk.LEFT)

        live_follow_var = tk.BooleanVar(value=(self.mode == "live"))
        if self.mode == "live":
            ttk.Checkbutton(
                top_frame,
                text="follow latest (live priority)",
                variable=live_follow_var,
            ).pack(side=tk.LEFT, padx=(12, 0))

        status_var = tk.StringVar(value="waiting for snapshots...")
        status_label = ttk.Label(top_frame, textvariable=status_var)
        status_label.pack(side=tk.RIGHT)

        slider_var = tk.IntVar(value=0)
        slider = ttk.Scale(root, from_=0, to=0, orient=tk.HORIZONTAL, variable=slider_var)
        slider.pack(side=tk.TOP, fill=tk.X, padx=8)

        content = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
        content.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=8)

        left = ttk.Frame(content)
        right = ttk.Frame(content)
        content.add(left, weight=2)
        content.add(right, weight=3)

        summary_text = tk.Text(left, wrap=tk.WORD, height=30)
        summary_text.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(right, bg="#171717", highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)

        selected_index = 0

        def clamp_index(idx: int) -> int:
            if not self._history:
                return 0
            return max(0, min(idx, len(self._history) - 1))

        def color_for_status(state: str) -> str:
            if state == "completed":
                return "#2f9e44"
            if state == "running":
                return "#1c7ed6"
            if state == "runnable":
                return "#f08c00"
            if state == "blocked":
                return "#c92a2a"
            return "#868e96"

        def render_snapshot(idx: int):
            nonlocal selected_index
            if not self._history:
                summary_text.delete("1.0", tk.END)
                summary_text.insert(tk.END, "No timeline data yet.")
                canvas.delete("all")
                status_var.set("waiting for snapshots...")
                return

            selected_index = clamp_index(idx)
            slider_var.set(selected_index)
            snap = self._history[selected_index]

            step = snap.get("step", selected_index)
            event = str(snap.get("event", ""))
            tick = snap.get("tick")
            score = snap.get("earned_points_total", 0.0)
            pending = snap.get("pending_count", 0)
            running = snap.get("running_by_arm", {})
            blocked = snap.get("blocked_arms", {})
            resources = snap.get("resource_state", {})
            resource_cfg = snap.get("resource_constraints", {})
            msg = str(snap.get("message", ""))

            status_var.set(
                f"step {selected_index + 1}/{len(self._history)} | event={event} | "
                f"pending={pending} | score={score}"
            )

            lines = [
                f"step index: {selected_index}",
                f"step id: {step}",
                f"event: {event}",
                f"tick: {tick}",
                f"pending_count: {pending}",
                f"running_by_arm: {running}",
                f"blocked_arms: {blocked}",
                f"resource_state: {resources}",
                f"resource_constraints: {resource_cfg}",
                f"earned_points_total: {score}",
                "",
                f"message: {msg}",
                "",
                "tasks:",
            ]

            tasks = snap.get("tasks", [])
            for task in tasks:
                lines.append(
                    f"- {task.get('task_id','')} | {task.get('name','')} | "
                    f"state={task.get('state','')} | arm={task.get('arm','any')} | "
                    f"priority={task.get('priority_score',0.0)}"
                )

            summary_text.delete("1.0", tk.END)
            summary_text.insert(tk.END, "\n".join(lines))

            canvas.delete("all")
            width = max(240, canvas.winfo_width())
            col_count = 4
            node_w = max(190, int((width - 40) / col_count) - 10)
            node_h = 56
            x_gap = 10
            y_gap = 14
            x0 = 12
            y0 = 12

            positions: Dict[str, tuple[int, int, int, int]] = {}
            for i, task in enumerate(tasks):
                row = i // col_count
                col = i % col_count
                x = x0 + col * (node_w + x_gap)
                y = y0 + row * (node_h + y_gap)
                task_id = str(task.get("task_id", ""))
                positions[task_id] = (x, y, x + node_w, y + node_h)

            # Draw dependency links first.
            for task in tasks:
                tid = str(task.get("task_id", ""))
                x1y1 = positions.get(tid)
                if x1y1 is None:
                    continue
                tx0, ty0, tx1, ty1 = x1y1
                for prereq in task.get("prerequisites", []):
                    prereq_id = str(prereq)
                    px0y0 = positions.get(prereq_id)
                    if px0y0 is None:
                        continue
                    px0, py0, px1, py1 = px0y0
                    canvas.create_line(
                        (px0 + px1) // 2,
                        py1,
                        (tx0 + tx1) // 2,
                        ty0,
                        fill="#495057",
                        width=2,
                    )

            # Draw nodes.
            for task in tasks:
                tid = str(task.get("task_id", ""))
                rect = positions.get(tid)
                if rect is None:
                    continue
                rx0, ry0, rx1, ry1 = rect
                state = str(task.get("state", "pending"))
                fill = color_for_status(state)
                canvas.create_rectangle(rx0, ry0, rx1, ry1, fill=fill, outline="#ced4da", width=1)
                canvas.create_text(
                    rx0 + 6,
                    ry0 + 8,
                    anchor=tk.NW,
                    fill="#ffffff",
                    text=str(task.get("name", ""))[:30],
                    font=("Segoe UI", 9, "bold"),
                )
                canvas.create_text(
                    rx0 + 6,
                    ry0 + 28,
                    anchor=tk.NW,
                    fill="#f1f3f5",
                    text=f"{tid[:26]}",
                    font=("Consolas", 8),
                )
                canvas.create_text(
                    rx1 - 6,
                    ry1 - 6,
                    anchor=tk.SE,
                    fill="#ffffff",
                    text=state,
                    font=("Segoe UI", 8),
                )

        def on_slider(_evt=None):
            idx = int(float(slider_var.get()))
            render_snapshot(idx)

        slider.configure(command=lambda _v: on_slider())

        def process_queue():
            nonlocal selected_index
            got_any = False
            while True:
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    break

                got_any = True
                typ = item.get("type")
                if typ == "stop":
                    root.after(0, root.destroy)
                    return
                if typ == "snapshot":
                    snap = dict(item.get("payload") or {})
                    snap.setdefault("timestamp", time.time())
                    self._history.append(snap)

            if got_any:
                slider.configure(to=max(0, len(self._history) - 1))
                if self.mode == "live" and live_follow_var.get():
                    selected_index = max(0, len(self._history) - 1)
                    render_snapshot(selected_index)
                else:
                    render_snapshot(selected_index)

            if self._stop_event.is_set():
                root.after(0, root.destroy)
                return

            root.after(80, process_queue)

        def on_close():
            self._stop_event.set()
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", on_close)
        root.after(100, process_queue)
        render_snapshot(0)
        root.mainloop()
        self._closed_event.set()
