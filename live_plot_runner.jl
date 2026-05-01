import Pkg

# GLMakie/GLFW on Windows requires all GL calls on thread 1.
# Julia must be started with --threads 1:
#   julia --threads 1 live_plot_runner.jl
if Threads.nthreads() > 1
    println("ERROR: Run with julia --threads 1 to avoid GLFW threading issues.")
    println("  julia --threads 1 live_plot_runner.jl --host 127.0.0.1 --port 9999")
    exit(1)
end

# Install required packages if missing.
for pkg in ["GLMakie", "Colors", "JSON3", "JLD2"]
    if Base.find_package(pkg) === nothing
        Pkg.add(pkg)
    end
end

using Sockets
using JSON3
using GLMakie
using Colors
using JLD2
using Dates

function safe_task_name_for_filename(task_name::AbstractString, fallback::AbstractString = "unnamed_task")
    raw = strip(task_name)
    if isempty(raw)
        raw = fallback
    end
    safe = replace(raw, r"[^A-Za-z0-9_.-]+" => "_")
    safe = strip(safe, ['.', '_', '-'])
    return isempty(safe) ? "unnamed_task" : safe
end

function arg_value(flag::String, default)
    idx = findfirst(==(flag), ARGS)
    if idx === nothing || idx == length(ARGS)
        return default
    end
    return ARGS[idx + 1]
end

function arg_int(flag::String, default::Int)
    raw = arg_value(flag, string(default))
    try
        return parse(Int, raw)
    catch
        println("Invalid value for ", flag, ": ", raw, ". Using ", default, ".")
        return default
    end
end

function csv_escape(s)
    txt = string(s)
    if occursin('"', txt)
        txt = replace(txt, '"' => "\"\"")
    end
    if occursin(',', txt) || occursin('"', txt) || occursin('\n', txt)
        return "\"" * txt * "\""
    end
    return txt
end

function append_named_waypoint_row(path::String, row::Dict{String, Any})
    values = [
        get(row, "waypoint_index", ""),
        get(row, "waypoint_name", ""),
        get(row, "task_id", ""),
        get(row, "task_name", ""),
        get(row, "dependent_item_label", ""),
        get(row, "left_gripper_open", ""),
        get(row, "right_gripper_open", ""),
        get(row, "left_x", ""), get(row, "left_y", ""), get(row, "left_z", ""),
        get(row, "left_rx", ""), get(row, "left_ry", ""), get(row, "left_rz", ""),
        get(row, "left_q_0", ""), get(row, "left_q_1", ""), get(row, "left_q_2", ""),
        get(row, "left_q_3", ""), get(row, "left_q_4", ""), get(row, "left_q_5", ""),
        get(row, "right_x", ""), get(row, "right_y", ""), get(row, "right_z", ""),
        get(row, "right_rx", ""), get(row, "right_ry", ""), get(row, "right_rz", ""),
        get(row, "right_q_0", ""), get(row, "right_q_1", ""), get(row, "right_q_2", ""),
        get(row, "right_q_3", ""), get(row, "right_q_4", ""), get(row, "right_q_5", ""),
        get(row, "left_distance_to_dependent_m", ""),
        get(row, "right_distance_to_dependent_m", ""),
        get(row, "left_offset_dx", ""), get(row, "left_offset_dy", ""), get(row, "left_offset_dz", ""),
        get(row, "right_offset_dx", ""), get(row, "right_offset_dy", ""), get(row, "right_offset_dz", ""),
        get(row, "waypoint_mark_time", ""),
    ]
    open(path, "a") do io
        println(io, join(csv_escape.(values), ","))
    end
end

function ensure_named_waypoints_header(path::String)
    if isfile(path)
        return
    end
    open(path, "w") do io
        println(io,
            "waypoint_index,waypoint_name,task_id,task_name,dependent_item_label," *
            "left_gripper_open,right_gripper_open," *
            "left_x,left_y,left_z,left_rx,left_ry,left_rz," *
            "left_q_0,left_q_1,left_q_2,left_q_3,left_q_4,left_q_5," *
            "right_x,right_y,right_z,right_rx,right_ry,right_rz," *
            "right_q_0,right_q_1,right_q_2,right_q_3,right_q_4,right_q_5," *
            "left_distance_to_dependent_m,right_distance_to_dependent_m," *
            "left_offset_dx,left_offset_dy,left_offset_dz,right_offset_dx,right_offset_dy,right_offset_dz," *
            "waypoint_mark_time"
        )
    end
end

function arg_host(default::String)
    raw = arg_value("--host", default)
    try
        return parse(IPAddr, raw)
    catch
        println("Invalid host '", raw, "'. Falling back to ", default)
        return parse(IPAddr, default)
    end
end

function norm01(v::Float64, lo::Float64, hi::Float64)
    return hi == lo ? 0.5 : (v - lo) / (hi - lo)
end

function mat_vec_mul_row(v::NTuple{3, Float64}, m::Matrix{Float64})
    # Row-vector multiply: [x y z] * M
    return (
        v[1] * m[1, 1] + v[2] * m[2, 1] + v[3] * m[3, 1],
        v[1] * m[1, 2] + v[2] * m[2, 2] + v[3] * m[3, 2],
        v[1] * m[1, 3] + v[2] * m[2, 3] + v[3] * m[3, 3],
    )
end

function base_to_global_task_xyz(base_xyz::NTuple{3, Float64}, arm::Symbol)
    # Mirrors Supervisor.compute_task_frames() constants and orientation.
    dy_t = 0.225 / 2 + 0.540 / 2
    dz_t = -0.753

    if arm == :left
        dx_t = 0.090 / 2 + 0.010 + 0.110
        r_task_to_base = Float64[
            0.707 0.0 -0.707;
            0.0 -1.0 0.0;
            -0.707 0.0 -0.707;
        ]
        trans_base_to_task = mat_vec_mul_row((dx_t, dy_t, dz_t), r_task_to_base)
        p_rel = (
            base_xyz[1] - trans_base_to_task[1],
            base_xyz[2] - trans_base_to_task[2],
            base_xyz[3] - trans_base_to_task[3],
        )
        # Inverse of row-vector transform p_base = p_task * R + t is p_task = (p_base - t) * R'.
        p_task = mat_vec_mul_row(p_rel, transpose(r_task_to_base))
        return p_task
    elseif arm == :right
        dx_t = -(0.090 / 2 + 0.010 + 0.110)
        r_task_to_base = Float64[
            0.707 0.0 0.707;
            0.0 -1.0 0.0;
            0.707 0.0 -0.707;
        ]
        trans_base_to_task = mat_vec_mul_row((dx_t, dy_t, dz_t), r_task_to_base)
        p_rel = (
            base_xyz[1] - trans_base_to_task[1],
            base_xyz[2] - trans_base_to_task[2],
            base_xyz[3] - trans_base_to_task[3],
        )
        p_task = mat_vec_mul_row(p_rel, transpose(r_task_to_base))
        return p_task
    else
        return base_xyz
    end
end

function main()
    host = arg_host("127.0.0.1")
    port = arg_int("--port", 9999)
    max_points = arg_int("--max-points", 400000000)
    refresh_every = arg_int("--refresh-every", 5)
    output_path = arg_value("--jld2-file", arg_value("--output", ""))
    if !isempty(output_path)
        println("Warning: --jld2-file/--output is deprecated and ignored. JLD2 name is derived from task name and saved in ./traces.")
    end
    named_waypoints_csv = arg_value("--named-waypoints-csv", "named_waypoints.csv")
    active_named_waypoints_csv = named_waypoints_csv
    active_task_name = ""

    GLMakie.activate!()

    fig = Figure(size = (1200, 860))
    ax = Axis3(
        fig[1, 1],
        xlabel = "X [m]",
        ylabel = "Y [m]",
        zlabel = "Z [m]",
        title = "Live Left/Right TCP (Global Task XYZ) & Camera Vision Poses"
    )

    Label(fig[2, 1], "Waypoint Name (press Enter to assign to oldest pending mark):", halign = :left)
    waypoint_name_box = Textbox(
        fig[3, 1],
        placeholder = "e.g. pregrasp_bowl",
        validator = r"^[A-Za-z0-9_.-]+$",
        defocus_on_submit = false,
        reset_on_defocus = true,
        tellwidth = true,
    )
    waypoint_status = Label(fig[4, 1], "Pending waypoint marks: 0", halign = :left)

    # Three observables: left TCP, right TCP, and vision detections.
    left_tcp_data_obs = Observable((Point3f[], RGBf[]))
    right_tcp_data_obs = Observable((Point3f[], RGBf[]))
    vision_data_obs = Observable((Point3f[], RGBf[]))

    scatter!(ax, @lift($left_tcp_data_obs[1]); color = @lift($left_tcp_data_obs[2]), markersize = 5, marker = :circle, label = "Left TCP")
    scatter!(ax, @lift($right_tcp_data_obs[1]); color = @lift($right_tcp_data_obs[2]), markersize = 5, marker = :utriangle, label = "Right TCP")
    scatter!(ax, @lift($vision_data_obs[1]); color = @lift($vision_data_obs[2]), markersize = 10, marker = :rect, label = "Vision")

    axislegend(ax, position = :rt)

    screen = GLMakie.Screen(start_renderloop = true)
    display(screen, fig)

    # Single UDP socket - receives both TCP pose packets and vision packets
    sock = UDPSocket()
    bind(sock, host, port)
    println("Listening on ", host, ":", port, " (TCP poses + vision detections)")
    println("Close the plot window or press Ctrl+C to stop.")
    println("Named waypoint CSV: ", named_waypoints_csv)

    ensure_named_waypoints_header(active_named_waypoints_csv)

    # Left TCP pose data (arm/base frame)
    left_tcp_xs = Float32[];  left_tcp_ys = Float32[];  left_tcp_zs = Float32[]
    # Left TCP pose data (shared global task frame)
    left_tcp_global_xs = Float32[];  left_tcp_global_ys = Float32[];  left_tcp_global_zs = Float32[]
    left_tcp_rxs = Float64[]; left_tcp_rys = Float64[]; left_tcp_rzs = Float64[]
    left_tcp_timestamps = Float64[]
    left_tcp_cols = RGBf[]
    left_tcp_rx_min = Inf;  left_tcp_rx_max = -Inf
    left_tcp_ry_min = Inf;  left_tcp_ry_max = -Inf
    left_tcp_rz_min = Inf;  left_tcp_rz_max = -Inf
    left_tcp_packet_count = 0

    # Right TCP pose data (arm/base frame)
    right_tcp_xs = Float32[];  right_tcp_ys = Float32[];  right_tcp_zs = Float32[]
    # Right TCP pose data (shared global task frame)
    right_tcp_global_xs = Float32[];  right_tcp_global_ys = Float32[];  right_tcp_global_zs = Float32[]
    right_tcp_rxs = Float64[]; right_tcp_rys = Float64[]; right_tcp_rzs = Float64[]
    right_tcp_timestamps = Float64[]
    right_tcp_cols = RGBf[]
    right_tcp_rx_min = Inf;  right_tcp_rx_max = -Inf
    right_tcp_ry_min = Inf;  right_tcp_ry_max = -Inf
    right_tcp_rz_min = Inf;  right_tcp_rz_max = -Inf
    right_tcp_packet_count = 0

    # Vision data
    vision_xs = Float32[];  vision_ys = Float32[];  vision_zs = Float32[]
    vision_color_names = String[]
    vision_cols = RGBf[]
    vision_packet_count = 0

    first_left_tcp_logged = false
    first_right_tcp_logged = false
    first_vision_logged = false
    pending_waypoints = Dict{String, Any}[]

    on(waypoint_name_box.stored_string) do s
        if s === nothing
            return
        end
        waypoint_name = strip(String(s))
        if isempty(waypoint_name)
            return
        end
        if isempty(pending_waypoints)
            waypoint_status.text[] = "No pending waypoint marks."
            return
        end

        wp = popfirst!(pending_waypoints)
        wp["waypoint_name"] = waypoint_name
        append_named_waypoint_row(active_named_waypoints_csv, wp)
        waypoint_status.text[] = "Saved waypoint '" * waypoint_name * "'. Pending waypoint marks: " * string(length(pending_waypoints))
        waypoint_name_box.displayed_string[] = nothing
    end

    # Color map for vision detections by color name
    color_map = Dict(
        "red"    => RGBf(0.9, 0.1, 0.1),
        "yellow" => RGBf(0.9, 0.85, 0.0),
        "green"  => RGBf(0.1, 0.8, 0.1),
        "blue"   => RGBf(0.1, 0.3, 0.9),
        "purple" => RGBf(0.6, 0.1, 0.8),
        "tan"    => RGBf(0.82, 0.71, 0.55),
    )

    # === PURE SINGLE-THREAD EVENT LOOP ===
    # Based on learnings from docs/explorations/makie.md:
    # - Single Julia thread enforced (julia --threads 1)
    # - recvfrom() for non-blocking UDP on Windows (readavailable() does not work for UDP)
    # - Explicit yield() ensures GLMakie's internal renderloop gets CPU cycles
    # - No @async/@spawn/@Task to avoid fragile async+renderloop interactions
    # - All data updates atomic via tuple observables (prevents sync races)

    Base.exit_on_sigint(false)
    try
        while isopen(screen)
            # Receive next UDP packet (recvfrom works for UDP on Windows; wraps any error as "no data")
            addr, data = try
                recvfrom(sock)
            catch
                (nothing, UInt8[])
            end

            if !isempty(data)
                pkt = try JSON3.read(String(data)) catch; nothing end

                if pkt !== nothing
                    # --- Waypoint mark packet ---
                    if haskey(pkt, :packet_type) && String(pkt.packet_type) == "waypoint_mark"
                        packet_task_name = String(get(pkt, :task_name, ""))
                        if !isempty(packet_task_name)
                            active_task_name = packet_task_name
                        end

                        packet_waypoint_csv = String(get(pkt, :named_waypoints_csv, active_named_waypoints_csv))
                        if !isempty(packet_waypoint_csv) && packet_waypoint_csv != active_named_waypoints_csv
                            active_named_waypoints_csv = packet_waypoint_csv
                            ensure_named_waypoints_header(active_named_waypoints_csv)
                            println("Waypoint CSV path switched to: ", active_named_waypoints_csv)
                        end

                        left_pose = haskey(pkt, :left_actual_TCP_pose) ? collect(Float64.(pkt.left_actual_TCP_pose)) : Float64[]
                        right_pose = haskey(pkt, :right_actual_TCP_pose) ? collect(Float64.(pkt.right_actual_TCP_pose)) : Float64[]
                        left_q = haskey(pkt, :left_actual_q) ? collect(Float64.(pkt.left_actual_q)) : Float64[]
                        right_q = haskey(pkt, :right_actual_q) ? collect(Float64.(pkt.right_actual_q)) : Float64[]
                        left_offset = haskey(pkt, :left_offset_to_dependent_xyz) && pkt.left_offset_to_dependent_xyz !== nothing ? collect(Float64.(pkt.left_offset_to_dependent_xyz)) : Float64[]
                        right_offset = haskey(pkt, :right_offset_to_dependent_xyz) && pkt.right_offset_to_dependent_xyz !== nothing ? collect(Float64.(pkt.right_offset_to_dependent_xyz)) : Float64[]

                        row = Dict{String, Any}(
                            "waypoint_index" => get(pkt, :waypoint_index, 0),
                            "task_id" => String(get(pkt, :task_id, "")),
                            "task_name" => String(get(pkt, :task_name, "")),
                            "dependent_item_label" => String(get(pkt, :dependent_item_label, "")),
                            "left_gripper_open" => get(pkt, :left_gripper_open, ""),
                            "right_gripper_open" => get(pkt, :right_gripper_open, ""),
                            "left_x" => length(left_pose) >= 1 ? left_pose[1] : "",
                            "left_y" => length(left_pose) >= 2 ? left_pose[2] : "",
                            "left_z" => length(left_pose) >= 3 ? left_pose[3] : "",
                            "left_rx" => length(left_pose) >= 4 ? left_pose[4] : "",
                            "left_ry" => length(left_pose) >= 5 ? left_pose[5] : "",
                            "left_rz" => length(left_pose) >= 6 ? left_pose[6] : "",
                            "left_q_0" => length(left_q) >= 1 ? left_q[1] : "",
                            "left_q_1" => length(left_q) >= 2 ? left_q[2] : "",
                            "left_q_2" => length(left_q) >= 3 ? left_q[3] : "",
                            "left_q_3" => length(left_q) >= 4 ? left_q[4] : "",
                            "left_q_4" => length(left_q) >= 5 ? left_q[5] : "",
                            "left_q_5" => length(left_q) >= 6 ? left_q[6] : "",
                            "right_x" => length(right_pose) >= 1 ? right_pose[1] : "",
                            "right_y" => length(right_pose) >= 2 ? right_pose[2] : "",
                            "right_z" => length(right_pose) >= 3 ? right_pose[3] : "",
                            "right_rx" => length(right_pose) >= 4 ? right_pose[4] : "",
                            "right_ry" => length(right_pose) >= 5 ? right_pose[5] : "",
                            "right_rz" => length(right_pose) >= 6 ? right_pose[6] : "",
                            "right_q_0" => length(right_q) >= 1 ? right_q[1] : "",
                            "right_q_1" => length(right_q) >= 2 ? right_q[2] : "",
                            "right_q_2" => length(right_q) >= 3 ? right_q[3] : "",
                            "right_q_3" => length(right_q) >= 4 ? right_q[4] : "",
                            "right_q_4" => length(right_q) >= 5 ? right_q[5] : "",
                            "right_q_5" => length(right_q) >= 6 ? right_q[6] : "",
                            "left_distance_to_dependent_m" => get(pkt, :left_distance_to_dependent_m, ""),
                            "right_distance_to_dependent_m" => get(pkt, :right_distance_to_dependent_m, ""),
                            "left_offset_dx" => length(left_offset) >= 1 ? left_offset[1] : "",
                            "left_offset_dy" => length(left_offset) >= 2 ? left_offset[2] : "",
                            "left_offset_dz" => length(left_offset) >= 3 ? left_offset[3] : "",
                            "right_offset_dx" => length(right_offset) >= 1 ? right_offset[1] : "",
                            "right_offset_dy" => length(right_offset) >= 2 ? right_offset[2] : "",
                            "right_offset_dz" => length(right_offset) >= 3 ? right_offset[3] : "",
                            "waypoint_mark_time" => get(pkt, :waypoint_mark_time, ""),
                        )
                        push!(pending_waypoints, row)
                        waypoint_status.text[] = "Pending waypoint marks: " * string(length(pending_waypoints))

                    # --- Left/Right TCP pose packet ---
                    elseif haskey(pkt, :left_actual_TCP_pose) || haskey(pkt, :right_actual_TCP_pose) || haskey(pkt, :actual_TCP_pose)
                        if haskey(pkt, :left_actual_TCP_pose) || haskey(pkt, :actual_TCP_pose)
                            left_pose = haskey(pkt, :left_actual_TCP_pose) ? pkt.left_actual_TCP_pose : pkt.actual_TCP_pose
                            if length(left_pose) >= 6
                                x_arm, y_arm, z_arm = Float32(left_pose[1]), Float32(left_pose[2]), Float32(left_pose[3])
                                gx, gy, gz = base_to_global_task_xyz((Float64(x_arm), Float64(y_arm), Float64(z_arm)), :left)
                                x, y, z = Float32(gx), Float32(gy), Float32(gz)
                                rx, ry, rz = Float64(left_pose[4]), Float64(left_pose[5]), Float64(left_pose[6])

                                if isfinite(x_arm) && isfinite(y_arm) && isfinite(z_arm) && isfinite(x) && isfinite(y) && isfinite(z)
                                    push!(left_tcp_xs, x_arm); push!(left_tcp_ys, y_arm); push!(left_tcp_zs, z_arm)
                                    push!(left_tcp_global_xs, x); push!(left_tcp_global_ys, y); push!(left_tcp_global_zs, z)
                                    push!(left_tcp_rxs, rx); push!(left_tcp_rys, ry); push!(left_tcp_rzs, rz)
                                    ts = Float64(get(pkt, :left_timestamp, get(pkt, :timestamp, 0.0)))
                                    push!(left_tcp_timestamps, ts)

                                    left_tcp_rx_min = min(left_tcp_rx_min, rx); left_tcp_rx_max = max(left_tcp_rx_max, rx)
                                    left_tcp_ry_min = min(left_tcp_ry_min, ry); left_tcp_ry_max = max(left_tcp_ry_max, ry)
                                    left_tcp_rz_min = min(left_tcp_rz_min, rz); left_tcp_rz_max = max(left_tcp_rz_max, rz)

                                    push!(left_tcp_cols, RGBf(
                                        norm01(rx, left_tcp_rx_min, left_tcp_rx_max),
                                        norm01(ry, left_tcp_ry_min, left_tcp_ry_max),
                                        norm01(rz, left_tcp_rz_min, left_tcp_rz_max)
                                    ))

                                    if length(left_tcp_xs) > max_points
                                        popfirst!(left_tcp_xs); popfirst!(left_tcp_ys); popfirst!(left_tcp_zs); popfirst!(left_tcp_cols)
                                        popfirst!(left_tcp_global_xs); popfirst!(left_tcp_global_ys); popfirst!(left_tcp_global_zs)
                                        popfirst!(left_tcp_rxs); popfirst!(left_tcp_rys); popfirst!(left_tcp_rzs)
                                        popfirst!(left_tcp_timestamps)
                                    end

                                    left_tcp_packet_count += 1
                                    println("[Left TCP #$left_tcp_packet_count] pos=($x, $y, $z) m  rot=($rx, $ry, $rz) rad")

                                    if !first_left_tcp_logged
                                        first_left_tcp_logged = true
                                        println("First LEFT TCP pose received.")
                                    end
                                    if left_tcp_packet_count % refresh_every == 0 || !first_left_tcp_logged
                                        left_tcp_data_obs[] = (Point3f.(left_tcp_global_xs, left_tcp_global_ys, left_tcp_global_zs), copy(left_tcp_cols))
                                        autolimits!(ax)
                                    end
                                end
                            end
                        end

                        if haskey(pkt, :right_actual_TCP_pose)
                            right_pose = pkt.right_actual_TCP_pose
                            if length(right_pose) >= 6
                                x_arm, y_arm, z_arm = Float32(right_pose[1]), Float32(right_pose[2]), Float32(right_pose[3])
                                gx, gy, gz = base_to_global_task_xyz((Float64(x_arm), Float64(y_arm), Float64(z_arm)), :right)
                                x, y, z = Float32(gx), Float32(gy), Float32(gz)
                                rx, ry, rz = Float64(right_pose[4]), Float64(right_pose[5]), Float64(right_pose[6])

                                if isfinite(x_arm) && isfinite(y_arm) && isfinite(z_arm) && isfinite(x) && isfinite(y) && isfinite(z)
                                    push!(right_tcp_xs, x_arm); push!(right_tcp_ys, y_arm); push!(right_tcp_zs, z_arm)
                                    push!(right_tcp_global_xs, x); push!(right_tcp_global_ys, y); push!(right_tcp_global_zs, z)
                                    push!(right_tcp_rxs, rx); push!(right_tcp_rys, ry); push!(right_tcp_rzs, rz)
                                    ts = Float64(get(pkt, :right_timestamp, get(pkt, :timestamp, 0.0)))
                                    push!(right_tcp_timestamps, ts)

                                    right_tcp_rx_min = min(right_tcp_rx_min, rx); right_tcp_rx_max = max(right_tcp_rx_max, rx)
                                    right_tcp_ry_min = min(right_tcp_ry_min, ry); right_tcp_ry_max = max(right_tcp_ry_max, ry)
                                    right_tcp_rz_min = min(right_tcp_rz_min, rz); right_tcp_rz_max = max(right_tcp_rz_max, rz)

                                    push!(right_tcp_cols, RGBf(
                                        norm01(rx, right_tcp_rx_min, right_tcp_rx_max),
                                        norm01(ry, right_tcp_ry_min, right_tcp_ry_max),
                                        norm01(rz, right_tcp_rz_min, right_tcp_rz_max)
                                    ))

                                    if length(right_tcp_xs) > max_points
                                        popfirst!(right_tcp_xs); popfirst!(right_tcp_ys); popfirst!(right_tcp_zs); popfirst!(right_tcp_cols)
                                        popfirst!(right_tcp_global_xs); popfirst!(right_tcp_global_ys); popfirst!(right_tcp_global_zs)
                                        popfirst!(right_tcp_rxs); popfirst!(right_tcp_rys); popfirst!(right_tcp_rzs)
                                        popfirst!(right_tcp_timestamps)
                                    end

                                    right_tcp_packet_count += 1
                                    println("[Right TCP #$right_tcp_packet_count] pos=($x, $y, $z) m  rot=($rx, $ry, $rz) rad")

                                    if !first_right_tcp_logged
                                        first_right_tcp_logged = true
                                        println("First RIGHT TCP pose received.")
                                    end
                                    if right_tcp_packet_count % refresh_every == 0 || !first_right_tcp_logged
                                        right_tcp_data_obs[] = (Point3f.(right_tcp_global_xs, right_tcp_global_ys, right_tcp_global_zs), copy(right_tcp_cols))
                                        autolimits!(ax)
                                    end
                                end
                            end
                        end

                    # --- Vision detection packet ---
                    elseif haskey(pkt, :position)
                        pos = pkt.position
                        if length(pos) >= 3
                            x, y, z = Float32(pos[1]), Float32(pos[2]), Float32(pos[3])

                            if isfinite(x) && isfinite(y) && isfinite(z)
                                push!(vision_xs, x); push!(vision_ys, y); push!(vision_zs, z)

                                color_name = get(pkt, :color, "unknown")
                                col = get(color_map, color_name, RGBf(0.5, 0.5, 0.5))
                                push!(vision_cols, col)
                                push!(vision_color_names, color_name)

                                if length(vision_xs) > max_points
                                    popfirst!(vision_xs); popfirst!(vision_ys); popfirst!(vision_zs)
                                    popfirst!(vision_cols); popfirst!(vision_color_names)
                                end

                                vision_packet_count += 1
                                println("[Vision #$vision_packet_count] pos=($x, $y, $z) m  color=$color_name")

                                if !first_vision_logged
                                    first_vision_logged = true
                                    println("First vision detection received.")
                                end
                                if vision_packet_count % refresh_every == 0 || !first_vision_logged
                                    vision_data_obs[] = (Point3f.(vision_xs, vision_ys, vision_zs), copy(vision_cols))
                                    autolimits!(ax)
                                end
                            end
                        end
                    end
                end
            end

            # Yield to GLMakie's renderloop
            yield()
        end
    catch err
        err isa InterruptException || rethrow()
        println("Interrupted by user.")
    finally
        Base.disable_sigint() do
            try; close(sock); catch; end
            try; GLMakie.closeall(); catch; end
        end
        Base.exit_on_sigint(true)

        # Save recorded data to JLD2
        traces_dir = "traces"
        try
            mkpath(traces_dir)
        catch e
            println("WARNING: could not create traces directory '$traces_dir': $e")
        end

        task_slug = safe_task_name_for_filename(active_task_name)
        save_path = joinpath(traces_dir, "trace_$(task_slug).jld2")
        try
            jldsave(save_path;
                # Existing keys preserved as arm/base-frame coordinates for backward compatibility.
                left_tcp_x = left_tcp_xs, left_tcp_y = left_tcp_ys, left_tcp_z = left_tcp_zs,
                left_tcp_global_x = left_tcp_global_xs, left_tcp_global_y = left_tcp_global_ys, left_tcp_global_z = left_tcp_global_zs,
                left_tcp_rx = left_tcp_rxs, left_tcp_ry = left_tcp_rys, left_tcp_rz = left_tcp_rzs,
                left_tcp_timestamp = left_tcp_timestamps,
                right_tcp_x = right_tcp_xs, right_tcp_y = right_tcp_ys, right_tcp_z = right_tcp_zs,
                right_tcp_global_x = right_tcp_global_xs, right_tcp_global_y = right_tcp_global_ys, right_tcp_global_z = right_tcp_global_zs,
                right_tcp_rx = right_tcp_rxs, right_tcp_ry = right_tcp_rys, right_tcp_rz = right_tcp_rzs,
                right_tcp_timestamp = right_tcp_timestamps,
                # Backward-compat aliases for older readers expecting tcp_* keys.
                tcp_x = left_tcp_xs, tcp_y = left_tcp_ys, tcp_z = left_tcp_zs,
                tcp_global_x = left_tcp_global_xs, tcp_global_y = left_tcp_global_ys, tcp_global_z = left_tcp_global_zs,
                tcp_rx = left_tcp_rxs, tcp_ry = left_tcp_rys, tcp_rz = left_tcp_rzs,
                tcp_timestamp = left_tcp_timestamps,
                vision_x = vision_xs, vision_y = vision_ys, vision_z = vision_zs,
                vision_color = vision_color_names
            )
            println("Data saved to: $save_path  ",
                    "($(length(left_tcp_xs)) left TCP pts, $(length(right_tcp_xs)) right TCP pts, $(length(vision_xs)) vision pts)")
        catch e
            println("WARNING: could not save data: $e")
        end
    end

    println("Live plot runner stopped.")
end

main()
