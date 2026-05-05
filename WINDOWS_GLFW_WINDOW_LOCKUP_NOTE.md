# Windows GLFW Window Lockup Note

## Scope
This note documents a Windows-specific GLMakie or GLFW windowing lockup observed in the Julia recorder and offline history viewers.

## Symptom
- The machine appears to hard-lock at the desktop or compositor level.
- CPU, RAM, and disk usage can remain normal.
- The trigger is window state changes, especially:
  - resizing
  - maximizing or fullscreen-like transitions
  - dragging a window partly out of display bounds and then resizing

## Affected scripts
- [live_plot_runner.jl](live_plot_runner.jl)
- [waypoint_offline_history_viewer.jl](waypoint_offline_history_viewer.jl)

## Root cause summary
The failure is consistent with a native window-state transition issue in GLFW on this Windows setup, rather than ordinary application resource saturation.

## Applied mitigation
A hard native GLFW window lock is applied immediately after window creation in both scripts:
- disable window resizing attribute
- clear maximized state
- set fixed size limits where min and max are equal to the current window size

This mitigation is implemented in helper functions named enforce_window_lock! in both scripts.

## Operational guidance
- Keep lock-windowed mode enabled by default on this machine.
- Avoid re-enabling free resize unless testing with caution.
- If lockups return even without resize events, switch to a non-GL windowing path for this workflow.

## Related context
Additional startup and loop-throttling safety improvements were made in the same period to reduce startup friction and event-loop pressure, but the decisive crash trigger here was resize or bounds transition behavior.
