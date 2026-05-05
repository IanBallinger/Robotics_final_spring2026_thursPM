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
import math
from typing import Any, Callable, Dict, List, Optional


class TaskGraphStateVisualizer:
    """Threaded task graph visualizer with timeline playback."""

    def __init__(
        self,
        mode: str = "simulate",
        title: str = "Task Graph Visualizer",
        control_callback: Optional[Callable[[str, Optional[Dict[str, Any]], Dict[str, Any]], str]] = None,
    ):
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
        self._control_callback = control_callback

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

        controls_frame = ttk.Frame(root)
        controls_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 8))

        selected_task_var = tk.StringVar(value="")
        action_status_var = tk.StringVar(value="Controls ready.")
        layout_mode_var = tk.StringVar(value="grid")
        bimanual_attempt_var = tk.BooleanVar(value=False)

        ttk.Label(controls_frame, text="Selected task id:").pack(side=tk.LEFT)
        task_id_entry = ttk.Entry(controls_frame, textvariable=selected_task_var, width=34)
        task_id_entry.pack(side=tk.LEFT, padx=(4, 10))

        def _find_selected_task() -> Optional[Dict[str, Any]]:
            if not self._history:
                return None
            snap = self._history[selected_index]
            selected_task_id = str(selected_task_var.get()).strip()
            if not selected_task_id:
                return None
            for task in snap.get("tasks", []):
                if str(task.get("task_id", "")).strip() == selected_task_id:
                    return dict(task)
            return None

        def _run_control_action(action_name: str):
            if self._control_callback is None:
                action_status_var.set("No control callback configured for this viewer.")
                return
            snapshot = self._history[selected_index] if self._history else {}
            selected_task = _find_selected_task()
            if selected_task is not None and action_name == "run_selected_task" and bimanual_attempt_var.get():
                selected_task = dict(selected_task)
                task_params = dict(selected_task.get("params", {}) or {})
                task_params["attempt_bimanual"] = True
                selected_task["params"] = task_params
            try:
                msg = self._control_callback(action_name, selected_task, dict(snapshot))
                action_status_var.set(str(msg))
            except Exception as exc:
                action_status_var.set(f"Action failed: {exc}")

        ttk.Button(
            controls_frame,
            text="Start Waypoint Recording",
            command=lambda: _run_control_action("start_recording"),
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(
            controls_frame,
            text="Run Selected Task",
            command=lambda: _run_control_action("run_selected_task"),
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Checkbutton(
            controls_frame,
            text="Attempt bimanual (live)",
            variable=bimanual_attempt_var,
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(
            controls_frame,
            text="Start Waypoint Tuning",
            command=lambda: _run_control_action("start_tuning"),
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(
            controls_frame,
            text="Pause/Halt Robot",
            command=lambda: _run_control_action("pause_halt"),
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(
            controls_frame,
            text="Resume Scheduler",
            command=lambda: _run_control_action("resume"),
        ).pack(side=tk.LEFT, padx=(0, 6))
        layout_button = ttk.Button(controls_frame, text="Layout: Grid")
        layout_button.pack(side=tk.LEFT, padx=(0, 6))
        ttk.Label(controls_frame, textvariable=action_status_var).pack(side=tk.LEFT, padx=(10, 0))

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
        node_rects_by_task_id: Dict[str, tuple[int, int, int, int]] = {}
        force_positions: Dict[str, List[float]] = {}
        force_velocities: Dict[str, List[float]] = {}
        dragged_task_id: Optional[str] = None
        hovered_task_id: Optional[str] = None

        def _toggle_layout_mode():
            cur = layout_mode_var.get().strip().lower()
            if cur == "grid":
                layout_mode_var.set("force")
                layout_button.configure(text="Layout: Force")
                action_status_var.set("Force-directed layout enabled (drag nodes to demo graph structure).")
            else:
                layout_mode_var.set("grid")
                layout_button.configure(text="Layout: Grid")
                action_status_var.set("Grid layout enabled.")
            render_snapshot(selected_index)

        layout_button.configure(command=_toggle_layout_mode)

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
            nonlocal selected_index, hovered_task_id
            if not self._history:
                summary_text.delete("1.0", tk.END)
                summary_text.insert(tk.END, "No timeline data yet.")
                canvas.delete("all")
                status_var.set("waiting for snapshots...")
                return

            selected_index = clamp_index(idx)
            slider_var.set(selected_index)
            snap = self._history[selected_index]

            if selected_task_var.get().strip() == "":
                running_task_id = ""
                for running_id in (snap.get("running_by_arm", {}) or {}).values():
                    if running_id:
                        running_task_id = str(running_id)
                        break
                if running_task_id:
                    selected_task_var.set(running_task_id)
                elif snap.get("tasks"):
                    selected_task_var.set(str(snap.get("tasks", [{}])[0].get("task_id", "")))

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
            height = max(220, canvas.winfo_height())
            col_count = 4
            node_w = max(190, int((width - 40) / col_count) - 10)
            node_h = 56
            x_gap = 10
            y_gap = 14
            x0 = 12
            y0 = 12

            positions: Dict[str, tuple[int, int, int, int]] = {}
            if layout_mode_var.get().strip().lower() == "force":
                task_ids = [str(task.get("task_id", "")) for task in tasks if str(task.get("task_id", ""))]
                task_set = set(task_ids)

                # Keep only active nodes in simulation state.
                for stale in list(force_positions.keys()):
                    if stale not in task_set:
                        force_positions.pop(stale, None)
                        force_velocities.pop(stale, None)

                if task_ids:
                    cx = width * 0.5
                    cy = height * 0.5
                    radius = max(90.0, min(width, height) * 0.28)
                    for i, tid in enumerate(task_ids):
                        if tid in force_positions:
                            continue
                        theta = (2.0 * math.pi * i) / max(1, len(task_ids))
                        force_positions[tid] = [cx + radius * math.cos(theta), cy + radius * math.sin(theta)]
                        force_velocities[tid] = [0.0, 0.0]

                    links: List[tuple[str, str]] = []
                    for task in tasks:
                        tid = str(task.get("task_id", ""))
                        if not tid:
                            continue
                        for prereq in task.get("prerequisites", []):
                            pid = str(prereq)
                            if pid and pid in task_set:
                                links.append((pid, tid))

                    repulsion = 9000.0
                    spring_k = 0.018
                    rest_len = float(max(70.0, node_w * 0.62))
                    damping = 0.83
                    border = 24.0

                    for i, aid in enumerate(task_ids):
                        ap = force_positions[aid]
                        av = force_velocities[aid]
                        fx = 0.0
                        fy = 0.0

                        for bid in task_ids[i + 1 :]:
                            bp = force_positions[bid]
                            dx = ap[0] - bp[0]
                            dy = ap[1] - bp[1]
                            dist2 = max(40.0, dx * dx + dy * dy)
                            dist = math.sqrt(dist2)
                            f = repulsion / dist2
                            nx = dx / dist
                            ny = dy / dist
                            fx += nx * f
                            fy += ny * f
                            bv = force_velocities[bid]
                            bv[0] -= nx * f
                            bv[1] -= ny * f

                        for src, dst in links:
                            if aid != src and aid != dst:
                                continue
                            other = dst if aid == src else src
                            op = force_positions.get(other)
                            if op is None:
                                continue
                            dx = op[0] - ap[0]
                            dy = op[1] - ap[1]
                            dist = max(1.0, math.sqrt(dx * dx + dy * dy))
                            pull = (dist - rest_len) * spring_k
                            fx += (dx / dist) * pull
                            fy += (dy / dist) * pull

                        # Centering force to keep graph visible.
                        fx += (cx - ap[0]) * 0.003
                        fy += (cy - ap[1]) * 0.003

                        if dragged_task_id == aid:
                            av[0] = 0.0
                            av[1] = 0.0
                            continue

                        av[0] = (av[0] + fx) * damping
                        av[1] = (av[1] + fy) * damping
                        ap[0] += av[0]
                        ap[1] += av[1]

                        ap[0] = min(max(border, ap[0]), max(border, width - border))
                        ap[1] = min(max(border, ap[1]), max(border, height - border))

                    for tid in task_ids:
                        cxn, cyn = force_positions[tid]
                        x = int(cxn - (node_w / 2.0))
                        y = int(cyn - (node_h / 2.0))
                        positions[tid] = (x, y, x + node_w, y + node_h)
            else:
                for i, task in enumerate(tasks):
                    row = i // col_count
                    col = i % col_count
                    x = x0 + col * (node_w + x_gap)
                    y = y0 + row * (node_h + y_gap)
                    task_id = str(task.get("task_id", ""))
                    positions[task_id] = (x, y, x + node_w, y + node_h)

            node_rects_by_task_id.clear()
            node_rects_by_task_id.update(positions)

            connected_edges: set[tuple[str, str]] = set()
            connected_nodes: set[str] = set()
            if hovered_task_id:
                connected_nodes.add(hovered_task_id)
                for task in tasks:
                    tid = str(task.get("task_id", ""))
                    for prereq in task.get("prerequisites", []):
                        prereq_id = str(prereq)
                        if tid == hovered_task_id or prereq_id == hovered_task_id:
                            connected_edges.add((prereq_id, tid))
                            connected_nodes.add(tid)
                            connected_nodes.add(prereq_id)

            # Draw dependency links first.
            for task in tasks:
                tid = str(task.get("task_id", ""))
                x1y1 = positions.get(tid)
                if x1y1 is None:
                    continue
                tx0, ty0, tx1, _ty1 = x1y1
                for prereq in task.get("prerequisites", []):
                    prereq_id = str(prereq)
                    px0y0 = positions.get(prereq_id)
                    if px0y0 is None:
                        continue
                    px0, _py0, px1, py1 = px0y0
                    is_hover_edge = (prereq_id, tid) in connected_edges
                    canvas.create_line(
                        (px0 + px1) // 2,
                        py1,
                        (tx0 + tx1) // 2,
                        ty0,
                        fill="#74c0fc" if is_hover_edge else "#495057",
                        width=4 if is_hover_edge else 2,
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
                if hovered_task_id and tid not in connected_nodes:
                    fill = "#495057"
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

        def on_canvas_click(event):
            nonlocal dragged_task_id
            ex = int(event.x)
            ey = int(event.y)
            for task_id, (x0r, y0r, x1r, y1r) in node_rects_by_task_id.items():
                if x0r <= ex <= x1r and y0r <= ey <= y1r:
                    selected_task_var.set(task_id)
                    action_status_var.set(f"Selected task: {task_id}")
                    if layout_mode_var.get().strip().lower() == "force":
                        dragged_task_id = task_id
                    return

        def on_canvas_hover(event):
            nonlocal hovered_task_id
            ex = int(event.x)
            ey = int(event.y)
            hit_task_id: Optional[str] = None
            for task_id, (x0r, y0r, x1r, y1r) in node_rects_by_task_id.items():
                if x0r <= ex <= x1r and y0r <= ey <= y1r:
                    hit_task_id = task_id
                    break
            if hit_task_id != hovered_task_id:
                hovered_task_id = hit_task_id
                render_snapshot(selected_index)

        def on_canvas_drag(event):
            if layout_mode_var.get().strip().lower() != "force":
                return
            if dragged_task_id is None:
                return
            if dragged_task_id not in force_positions:
                return
            force_positions[dragged_task_id][0] = float(int(event.x))
            force_positions[dragged_task_id][1] = float(int(event.y))
            vel = force_velocities.get(dragged_task_id)
            if vel is not None:
                vel[0] = 0.0
                vel[1] = 0.0
            render_snapshot(selected_index)

        def on_canvas_release(_event):
            nonlocal dragged_task_id
            dragged_task_id = None

        canvas.bind("<Button-1>", on_canvas_click)
        canvas.bind("<Motion>", on_canvas_hover)
        canvas.bind("<B1-Motion>", on_canvas_drag)
        canvas.bind("<ButtonRelease-1>", on_canvas_release)

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
            elif self._history and layout_mode_var.get().strip().lower() == "force":
                # Keep animating force layout even when no new snapshots arrive.
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
