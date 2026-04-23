# Makie Live Plotting Exploration (Windows + Julia 1.12)

## Goal
Establish a reliable way to render live 3D TCP pose data in a separate window on Windows.

## Final Outcome
Live rendering works reliably with a standalone Julia script and single-thread execution.

- Script: live_plot_runner.jl
- Run command: julia --threads 1 live_plot_runner.jl --host 127.0.0.1 --port 9999 --refresh-every 1

## Salient Discoveries

### 1) Run GLMakie on a single Julia thread
GLFW/GLMakie calls must execute on thread 1 on Windows. Running with default threading can trigger errors like:

- ThreadAssertionError: Code must run on thread 1 but ran on thread N

Use:

- julia --threads 1 ...

### 2) Jupyter execution model was a major destabilizer
Notebook task scheduling and window/render lifecycle handling made live GL window behavior inconsistent. Moving to a standalone script significantly improved reliability.

### 3) Separate window rendering works with GLMakie
For windowed output, use GLMakie screen display (instead of notebook inline rendering patterns).

### 4) Async UDP + renderloop interactions are fragile
Several async variants produced hard-to-debug issues (render starvation, interrupt handling side effects, cleanup races). A pure single-thread event flow is more predictable for this use case.

### 5) Atomic data updates matter for plotting stability
Updating points and colors as separate observables can transiently desynchronize lengths during redraw. Using a single tuple observable for both arrays avoids mismatched state windows.

### 6) Explicit cooperative yielding helps redraw responsiveness
Under high packet rates, adding yield points in the processing loop helps the GL renderloop run and refresh frames.

### 7) Force limits refresh when data arrives
Calling autolimits! on updates helps ensure incoming points are visible in-axis during live updates.

### 8) UDP packets were arriving even when RTDE ended with errors
The Python recorder could still stream UDP while later exiting with code 1 due to RTDE/network exceptions. Packet arrival and recorder process exit code should be treated as separate signals.

### 9) poll_fd location and signatures
On Julia 1.12, poll_fd is in FileWatching (not Base), with signatures for RawFD / WindowsRawSocket.

### 10) PowerShell quoting repeatedly caused false-negative Julia -e tests
Many parse errors came from shell quoting/escaping, not Julia code correctness. Prefer one of these when testing with -e in PowerShell:

- Single-quoted outer command with normal Julia double quotes inside
- A PowerShell here-string assigned to a variable and passed to julia -e

## Recommended Operational Workflow

1. Start Julia plot runner first:
   - julia --threads 1 live_plot_runner.jl --host 127.0.0.1 --port 9999 --refresh-every 1
2. Start Python UDP stream sender:
   - python UR5/demo_4_16/record.py -o task2.csv --stream-udp-host 127.0.0.1 --stream-udp-port 9999
3. Verify console message:
   - First UDP packet received; live plot updating.
4. For production, increase refresh_every if needed for lower redraw overhead.

## Remaining Caveat
Ctrl+C teardown around GLMakie/GLFW can still produce noisy interrupt/logging behavior in some paths on Windows. Window-close shutdown is generally cleaner than SIGINT-driven shutdown.
