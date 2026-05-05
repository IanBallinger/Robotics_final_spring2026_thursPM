# GLMakie/GLFW on Windows requires all GL calls on thread 1.
# Julia must be started with --threads 1:
#   julia --threads 1 live_plot_runner.jl
if Threads.nthreads() > 1
    println("ERROR: Run with julia --threads 1 to avoid GLFW threading issues.")
    println("  julia --threads 1 live_plot_runner.jl --host 127.0.0.1 --port 9999")
    exit(1)
end

using Sockets
using JSON3
using GLMakie
using Colors
using JLD2
using Dates
using Printf

# Frame transform helpers mirroring Supervisor.compute_task_frames().
function mat_vec_mul_row(v::NTuple{3, Float64}, m::AbstractMatrix{<:Real})
    return (
        v[1] * m[1, 1] + v[2] * m[2, 1] + v[3] * m[3, 1],
        v[1] * m[1, 2] + v[2] * m[2, 2] + v[3] * m[3, 2],
        v[1] * m[1, 3] + v[2] * m[2, 3] + v[3] * m[3, 3],
    )
end

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

function arg_float(flag::String, default::Float64)
    raw = arg_value(flag, string(default))
    try
        return parse(Float64, raw)
    catch
        println("Invalid value for ", flag, ": ", raw, ". Using ", default, ".")
        return default
    end
end

function arg_bool(flag::String, default::Bool)
    raw = lowercase(strip(string(arg_value(flag, default ? "true" : "false"))))
    if raw in ("1", "true", "yes", "on")
        return true
    elseif raw in ("0", "false", "no", "off")
        return false
    else
        println("Invalid value for ", flag, ": ", raw, ". Using ", default, ".")
        return default
    end
end

function enforce_window_lock!(screen)
    glfw = GLMakie.GLFW
    win = GLMakie.to_native(screen)

    # Force-disable resize/maximize at the native GLFW layer.
    glfw.SetWindowAttrib(win, glfw.RESIZABLE, false)
    try
        glfw.SetWindowAttrib(win, glfw.MAXIMIZED, false)
    catch
    end

    ww, wh = glfw.GetWindowSize(win)
    glfw.SetWindowSizeLimits(win, ww, wh, ww, wh)
    return nothing
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
        get(row, "left_gripper_open_pct", ""),
        get(row, "right_gripper_open_pct", ""),
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
        get(row, "left_task_x", ""), get(row, "left_task_y", ""), get(row, "left_task_z", ""),
        get(row, "right_task_x", ""), get(row, "right_task_y", ""), get(row, "right_task_z", ""),
        get(row, "left_global_x", ""), get(row, "left_global_y", ""), get(row, "left_global_z", ""),
        get(row, "right_global_x", ""), get(row, "right_global_y", ""), get(row, "right_global_z", ""),
        get(row, "tracked_items_json", ""),
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
            "left_gripper_open,right_gripper_open,left_gripper_open_pct,right_gripper_open_pct," *
            "left_x,left_y,left_z,left_rx,left_ry,left_rz," *
            "left_q_0,left_q_1,left_q_2,left_q_3,left_q_4,left_q_5," *
            "right_x,right_y,right_z,right_rx,right_ry,right_rz," *
            "right_q_0,right_q_1,right_q_2,right_q_3,right_q_4,right_q_5," *
            "left_distance_to_dependent_m,right_distance_to_dependent_m," *
            "left_offset_dx,left_offset_dy,left_offset_dz,right_offset_dx,right_offset_dy,right_offset_dz," *
            "left_task_x,left_task_y,left_task_z,right_task_x,right_task_y,right_task_z," *
            "left_global_x,left_global_y,left_global_z,right_global_x,right_global_y,right_global_z," *
            "tracked_items_json," *
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

function format_joint_bar(q::Float64, lo::Float64, hi::Float64; width::Int = 16)
    if !isfinite(q)
        return repeat(" ", width), 0.5
    end
    frac = hi == lo ? 0.5 : (q - lo) / (hi - lo)
    frac_clamped = clamp(frac, 0.0, 1.0)
    marker_idx = clamp(Int(round(frac_clamped * (width - 1))) + 1, 1, width)
    chars = fill(' ', width)
    for i in 1:(marker_idx - 1)
        chars[i] = '-'
    end
    chars[marker_idx] = 'x'
    return String(chars), frac_clamped
end

function format_joint_tracker_block(
    arm_name::String,
    q_vals::Vector{Float64},
    joint_limits::Vector{Tuple{Float64, Float64}},
    joint_names::Vector{String}
)
    lines = ["$arm_name joint range tracker", "lower lim [bar] upper lim"]
    warnings = String[]

    for i in 1:min(length(q_vals), length(joint_limits), length(joint_names))
        q = q_vals[i]
        lo, hi = joint_limits[i]
        bar, frac = format_joint_bar(q, lo, hi; width = 16)
        pct = Int(round(frac * 100))
        warn_frac = i >= 4 ? 0.18 : 0.10
        near_limit = isfinite(q) && (frac <= warn_frac || frac >= 1.0 - warn_frac)
        side = frac <= 0.5 ? "LOW" : "HIGH"

        push!(lines, @sprintf("%-10s [%-16s] %3d%%  q=%+0.2f", joint_names[i], bar, pct, q))
        if near_limit
            push!(warnings, @sprintf("%s %s near %s limit (%+0.2f rad)", arm_name, joint_names[i], side, q))
        end
    end

    return join(lines, "\n"), warnings
end

function load_joint_limits_from_python(default_limits::Vector{Tuple{Float64, Float64}})
    python_exe = arg_value("--python-exe", "python")
    py_code = (
        "import json,sys; " *
        "sys.path.insert(0, 'UR5'); " *
        "from joint_limits import UR5_JOINT_LIMITS_RAD; " *
        "print(json.dumps(UR5_JOINT_LIMITS_RAD))"
    )

    try
        out = read(`$python_exe -c $py_code`, String)
        parsed = JSON3.read(strip(out))
        limits = Tuple{Float64, Float64}[]
        for item in parsed
            if length(item) < 2
                continue
            end
            lo = Float64(item[1])
            hi = Float64(item[2])
            push!(limits, (lo, hi))
        end

        if length(limits) == 6
            println("Loaded joint limits from Python module UR5/joint_limits.py")
            return limits
        else
            println("Warning: Python joint-limits payload malformed; using fallback ±2pi")
            return default_limits
        end
    catch e
        println("Warning: could not load joint limits from Python (", e, "); using fallback ±2pi")
        return default_limits
    end
end

function write_observed_joint_limits(
    path::String,
    joint_names::Vector{String},
    joint_limits::Vector{Tuple{Float64, Float64}},
    observed_min::Vector{Float64},
    observed_max::Vector{Float64},
)
    payload = Dict(
        "updated_at" => string(Dates.now()),
        "joint_names" => joint_names,
        "joint_limits_rad" => [[lim[1], lim[2]] for lim in joint_limits],
        "observed_min_rad" => observed_min,
        "observed_max_rad" => observed_max,
    )

    mkpath(dirname(path))
    open(path, "w") do io
        write(io, JSON3.write(payload))
        write(io, "\n")
    end
end

function load_observed_joint_limits_from_disk(
    path::String,
    baseline_limits::Vector{Tuple{Float64, Float64}},
)
    merged = copy(baseline_limits)
    observed_min = [lim[1] for lim in merged]
    observed_max = [lim[2] for lim in merged]

    if !isfile(path)
        return merged, observed_min, observed_max
    end

    try
        raw = read(path, String)
        parsed = JSON3.read(raw)

        limits_payload = haskey(parsed, :joint_limits_rad) ? parsed.joint_limits_rad : nothing
        if limits_payload !== nothing
            for i in 1:min(length(merged), length(limits_payload))
                lim = limits_payload[i]
                if length(lim) >= 2
                    lo = Float64(lim[1])
                    hi = Float64(lim[2])
                    merged[i] = (min(merged[i][1], lo), max(merged[i][2], hi))
                end
            end
        end

        if haskey(parsed, :observed_min_rad)
            min_payload = parsed.observed_min_rad
            for i in 1:min(length(observed_min), length(min_payload))
                observed_min[i] = Float64(min_payload[i])
            end
        end

        if haskey(parsed, :observed_max_rad)
            max_payload = parsed.observed_max_rad
            for i in 1:min(length(observed_max), length(max_payload))
                observed_max[i] = Float64(max_payload[i])
            end
        end

        for i in 1:length(merged)
            merged[i] = (min(merged[i][1], observed_min[i]), max(merged[i][2], observed_max[i]))
            observed_min[i] = min(observed_min[i], merged[i][1])
            observed_max[i] = max(observed_max[i], merged[i][2])
        end

        println("Loaded observed joint limits from disk: ", path)
    catch e
        println("Warning: could not parse observed joint limits file '", path, "': ", e)
    end

    return merged, observed_min, observed_max
end

function update_limits_from_sample!(
    joint_limits::Vector{Tuple{Float64, Float64}},
    observed_min::Vector{Float64},
    observed_max::Vector{Float64},
    q_vals::Vector{Float64},
)
    changed = false

    for i in 1:min(length(joint_limits), length(q_vals))
        q = q_vals[i]
        if !isfinite(q)
            continue
        end

        if q < observed_min[i]
            observed_min[i] = q
            changed = true
        end
        if q > observed_max[i]
            observed_max[i] = q
            changed = true
        end

        lo, hi = joint_limits[i]
        new_lo = min(lo, q)
        new_hi = max(hi, q)
        if new_lo != lo || new_hi != hi
            joint_limits[i] = (new_lo, new_hi)
            changed = true
        end
    end

    return changed
end

function jld2_read_numeric_vector(data, keys::Vector{String}, ::Type{T}) where T<:Real
    for key in keys
        if !haskey(data, key)
            continue
        end
        raw = data[key]
        if !(raw isa AbstractVector)
            continue
        end

        out = T[]
        for v in raw
            try
                fv = Float64(v)
                if isfinite(fv)
                    push!(out, T(fv))
                end
            catch
                # Skip malformed entries.
            end
        end
        return out
    end
    return T[]
end

function jld2_read_string_vector(data, keys::Vector{String})
    for key in keys
        if !haskey(data, key)
            continue
        end
        raw = data[key]
        if !(raw isa AbstractVector)
            continue
        end
        return [string(v) for v in raw]
    end
    return String[]
end

function take_first_n(v::AbstractVector, n::Int)
    if n <= 0
        return eltype(v)[]
    end
    return collect(v[1:n])
end

function build_pose_colors(rxs::Vector{Float64}, rys::Vector{Float64}, rzs::Vector{Float64})
    n = min(length(rxs), length(rys), length(rzs))
    if n == 0
        return RGBf[]
    end

    rx_min, rx_max = minimum(rxs[1:n]), maximum(rxs[1:n])
    ry_min, ry_max = minimum(rys[1:n]), maximum(rys[1:n])
    rz_min, rz_max = minimum(rzs[1:n]), maximum(rzs[1:n])

    cols = RGBf[]
    for i in 1:n
        push!(cols, RGBf(
            norm01(rxs[i], rx_min, rx_max),
            norm01(rys[i], ry_min, ry_max),
            norm01(rzs[i], rz_min, rz_max),
        ))
    end
    return cols
end

function load_bootstrap_trace_jld2(path::String)
    clean = strip(path)
    if isempty(clean) || lowercase(clean) in ["none", "off", "false"]
        return nothing
    end

    if !isfile(clean)
        println("Bootstrap JLD2 not found: ", clean)
        return nothing
    end

    try
        data = JLD2.load(clean)

        left_tcp_x = jld2_read_numeric_vector(data, ["left_tcp_x", "tcp_x"], Float32)
        left_tcp_y = jld2_read_numeric_vector(data, ["left_tcp_y", "tcp_y"], Float32)
        left_tcp_z = jld2_read_numeric_vector(data, ["left_tcp_z", "tcp_z"], Float32)
        left_global_x = jld2_read_numeric_vector(data, ["left_tcp_global_x", "tcp_global_x", "left_task_x"], Float32)
        left_global_y = jld2_read_numeric_vector(data, ["left_tcp_global_y", "tcp_global_y", "left_task_y"], Float32)
        left_global_z = jld2_read_numeric_vector(data, ["left_tcp_global_z", "tcp_global_z", "left_task_z"], Float32)
        left_task_x = jld2_read_numeric_vector(data, ["left_task_x", "left_tcp_global_x", "tcp_global_x"], Float32)
        left_task_y = jld2_read_numeric_vector(data, ["left_task_y", "left_tcp_global_y", "tcp_global_y"], Float32)
        left_task_z = jld2_read_numeric_vector(data, ["left_task_z", "left_tcp_global_z", "tcp_global_z"], Float32)
        left_rx = jld2_read_numeric_vector(data, ["left_tcp_rx", "tcp_rx"], Float64)
        left_ry = jld2_read_numeric_vector(data, ["left_tcp_ry", "tcp_ry"], Float64)
        left_rz = jld2_read_numeric_vector(data, ["left_tcp_rz", "tcp_rz"], Float64)
        left_ts = jld2_read_numeric_vector(data, ["left_tcp_timestamp", "tcp_timestamp"], Float64)

        if isempty(left_global_x); left_global_x = copy(left_tcp_x); end
        if isempty(left_global_y); left_global_y = copy(left_tcp_y); end
        if isempty(left_global_z); left_global_z = copy(left_tcp_z); end
        if isempty(left_task_x); left_task_x = copy(left_global_x); end
        if isempty(left_task_y); left_task_y = copy(left_global_y); end
        if isempty(left_task_z); left_task_z = copy(left_global_z); end
        if isempty(left_ts)
            left_ts = [Float64(i - 1) for i in 1:length(left_tcp_x)]
        end

        left_n = minimum([
            length(left_tcp_x), length(left_tcp_y), length(left_tcp_z),
            length(left_global_x), length(left_global_y), length(left_global_z),
            length(left_task_x), length(left_task_y), length(left_task_z),
            length(left_rx), length(left_ry), length(left_rz), length(left_ts),
        ])

        right_tcp_x = jld2_read_numeric_vector(data, ["right_tcp_x"], Float32)
        right_tcp_y = jld2_read_numeric_vector(data, ["right_tcp_y"], Float32)
        right_tcp_z = jld2_read_numeric_vector(data, ["right_tcp_z"], Float32)
        right_global_x = jld2_read_numeric_vector(data, ["right_tcp_global_x", "right_task_x"], Float32)
        right_global_y = jld2_read_numeric_vector(data, ["right_tcp_global_y", "right_task_y"], Float32)
        right_global_z = jld2_read_numeric_vector(data, ["right_tcp_global_z", "right_task_z"], Float32)
        right_task_x = jld2_read_numeric_vector(data, ["right_task_x", "right_tcp_global_x"], Float32)
        right_task_y = jld2_read_numeric_vector(data, ["right_task_y", "right_tcp_global_y"], Float32)
        right_task_z = jld2_read_numeric_vector(data, ["right_task_z", "right_tcp_global_z"], Float32)
        right_rx = jld2_read_numeric_vector(data, ["right_tcp_rx"], Float64)
        right_ry = jld2_read_numeric_vector(data, ["right_tcp_ry"], Float64)
        right_rz = jld2_read_numeric_vector(data, ["right_tcp_rz"], Float64)
        right_ts = jld2_read_numeric_vector(data, ["right_tcp_timestamp"], Float64)

        if isempty(right_global_x); right_global_x = copy(right_tcp_x); end
        if isempty(right_global_y); right_global_y = copy(right_tcp_y); end
        if isempty(right_global_z); right_global_z = copy(right_tcp_z); end
        if isempty(right_task_x); right_task_x = copy(right_global_x); end
        if isempty(right_task_y); right_task_y = copy(right_global_y); end
        if isempty(right_task_z); right_task_z = copy(right_global_z); end
        if isempty(right_ts)
            right_ts = [Float64(i - 1) for i in 1:length(right_tcp_x)]
        end

        right_n = minimum([
            length(right_tcp_x), length(right_tcp_y), length(right_tcp_z),
            length(right_global_x), length(right_global_y), length(right_global_z),
            length(right_task_x), length(right_task_y), length(right_task_z),
            length(right_rx), length(right_ry), length(right_rz), length(right_ts),
        ])

        vision_x = jld2_read_numeric_vector(data, ["vision_x"], Float32)
        vision_y = jld2_read_numeric_vector(data, ["vision_y"], Float32)
        vision_z = jld2_read_numeric_vector(data, ["vision_z"], Float32)
        vision_ts = jld2_read_numeric_vector(data, ["vision_timestamp"], Float64)
        vision_labels = jld2_read_string_vector(data, ["vision_label"])
        vision_colors = jld2_read_string_vector(data, ["vision_color"])

        if isempty(vision_ts)
            vision_ts = [Float64(i - 1) for i in 1:length(vision_x)]
        end
        if isempty(vision_labels)
            vision_labels = ["" for _ in 1:length(vision_x)]
        end
        if isempty(vision_colors)
            vision_colors = ["unknown" for _ in 1:length(vision_x)]
        end
        vision_n = minimum([
            length(vision_x), length(vision_y), length(vision_z), length(vision_ts),
            length(vision_labels), length(vision_colors),
        ])

        left_q0 = jld2_read_numeric_vector(data, ["left_q_0"], Float64)
        left_q1 = jld2_read_numeric_vector(data, ["left_q_1"], Float64)
        left_q2 = jld2_read_numeric_vector(data, ["left_q_2"], Float64)
        left_q3 = jld2_read_numeric_vector(data, ["left_q_3"], Float64)
        left_q4 = jld2_read_numeric_vector(data, ["left_q_4"], Float64)
        left_q5 = jld2_read_numeric_vector(data, ["left_q_5"], Float64)
        left_q_n = minimum([length(left_q0), length(left_q1), length(left_q2), length(left_q3), length(left_q4), length(left_q5)])

        right_q0 = jld2_read_numeric_vector(data, ["right_q_0"], Float64)
        right_q1 = jld2_read_numeric_vector(data, ["right_q_1"], Float64)
        right_q2 = jld2_read_numeric_vector(data, ["right_q_2"], Float64)
        right_q3 = jld2_read_numeric_vector(data, ["right_q_3"], Float64)
        right_q4 = jld2_read_numeric_vector(data, ["right_q_4"], Float64)
        right_q5 = jld2_read_numeric_vector(data, ["right_q_5"], Float64)
        right_q_n = minimum([length(right_q0), length(right_q1), length(right_q2), length(right_q3), length(right_q4), length(right_q5)])

        payload = Dict{String, Any}(
            "left_tcp_x" => take_first_n(left_tcp_x, left_n),
            "left_tcp_y" => take_first_n(left_tcp_y, left_n),
            "left_tcp_z" => take_first_n(left_tcp_z, left_n),
            "left_global_x" => take_first_n(left_global_x, left_n),
            "left_global_y" => take_first_n(left_global_y, left_n),
            "left_global_z" => take_first_n(left_global_z, left_n),
            "left_task_x" => take_first_n(left_task_x, left_n),
            "left_task_y" => take_first_n(left_task_y, left_n),
            "left_task_z" => take_first_n(left_task_z, left_n),
            "left_rx" => take_first_n(left_rx, left_n),
            "left_ry" => take_first_n(left_ry, left_n),
            "left_rz" => take_first_n(left_rz, left_n),
            "left_ts" => take_first_n(left_ts, left_n),
            "left_cols" => build_pose_colors(take_first_n(left_rx, left_n), take_first_n(left_ry, left_n), take_first_n(left_rz, left_n)),
            "right_tcp_x" => take_first_n(right_tcp_x, right_n),
            "right_tcp_y" => take_first_n(right_tcp_y, right_n),
            "right_tcp_z" => take_first_n(right_tcp_z, right_n),
            "right_global_x" => take_first_n(right_global_x, right_n),
            "right_global_y" => take_first_n(right_global_y, right_n),
            "right_global_z" => take_first_n(right_global_z, right_n),
            "right_task_x" => take_first_n(right_task_x, right_n),
            "right_task_y" => take_first_n(right_task_y, right_n),
            "right_task_z" => take_first_n(right_task_z, right_n),
            "right_rx" => take_first_n(right_rx, right_n),
            "right_ry" => take_first_n(right_ry, right_n),
            "right_rz" => take_first_n(right_rz, right_n),
            "right_ts" => take_first_n(right_ts, right_n),
            "right_cols" => build_pose_colors(take_first_n(right_rx, right_n), take_first_n(right_ry, right_n), take_first_n(right_rz, right_n)),
            "vision_x" => take_first_n(vision_x, vision_n),
            "vision_y" => take_first_n(vision_y, vision_n),
            "vision_z" => take_first_n(vision_z, vision_n),
            "vision_ts" => take_first_n(vision_ts, vision_n),
            "vision_labels" => take_first_n(vision_labels, vision_n),
            "vision_color_names" => take_first_n(vision_colors, vision_n),
            "left_q0" => take_first_n(left_q0, left_q_n),
            "left_q1" => take_first_n(left_q1, left_q_n),
            "left_q2" => take_first_n(left_q2, left_q_n),
            "left_q3" => take_first_n(left_q3, left_q_n),
            "left_q4" => take_first_n(left_q4, left_q_n),
            "left_q5" => take_first_n(left_q5, left_q_n),
            "right_q0" => take_first_n(right_q0, right_q_n),
            "right_q1" => take_first_n(right_q1, right_q_n),
            "right_q2" => take_first_n(right_q2, right_q_n),
            "right_q3" => take_first_n(right_q3, right_q_n),
            "right_q4" => take_first_n(right_q4, right_q_n),
            "right_q5" => take_first_n(right_q5, right_q_n),
        )

        println(
            "Bootstrapped from JLD2 ", clean,
            " (left=", left_n,
            ", right=", right_n,
            ", vision=", vision_n,
            ")"
        )
        return payload
    catch e
        println("Warning: failed to bootstrap from JLD2 '", clean, "': ", e)
        return nothing
    end
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
    # Keep bounded history by default. Very large point clouds can stall GL on some Windows drivers.
    max_points = arg_int("--max-points", 200000)
    refresh_every = arg_int("--refresh-every", 5)
    packet_log_every = max(1, arg_int("--packet-log-every", 100))
    observed_limits_save_interval_s = max(0.1, arg_float("--observed-limits-save-sec", 2.0))
    idle_sleep_s = max(0.0, arg_float("--idle-sleep-ms", 2.0) / 1000.0)
    lock_windowed = arg_bool("--lock-windowed", true)
    output_path = arg_value("--jld2-file", arg_value("--output", ""))
    if !isempty(output_path)
        println("Warning: --jld2-file/--output is deprecated and ignored. JLD2 name is derived from task name and saved in ./traces.")
    end
    named_waypoints_csv = arg_value("--named-waypoints-csv", "named_waypoints.csv")
    bootstrap_jld2_path = arg_value("--bootstrap-jld2", "")
    active_named_waypoints_csv = named_waypoints_csv
    active_task_name = ""
    active_task_id = ""
    active_dependent_item_label = ""

    GLMakie.activate!()

    fig = Figure(size = (2400, 1300))
    ax = Axis3(
        fig[1, 1],
        xlabel = "X [m]",
        ylabel = "Y [m]",
        zlabel = "Z [m]",
        title = "Live Left/Right TCP (Global Task XYZ) & Camera Vision Poses",
        titlegap = 10
    )

    side_panel = GridLayout(fig[1, 2])

    Label(side_panel[5, 1], "Waypoint Name (press Enter to assign to oldest pending mark):", halign = :left)
    waypoint_name_box = Textbox(
        side_panel[6, 1],
        placeholder = "e.g. pregrasp_bowl",
        validator = r"^[A-Za-z0-9_.-]+$",
        defocus_on_submit = false,
        reset_on_defocus = true,
        tellwidth = true,
    )
    waypoint_status = Label(side_panel[7, 1], "Pending waypoint marks: 0", halign = :left)

    # Joint-limit tracker panel (bar-style textual UI for operator range awareness).
    joint_limits_fallback = [
        (-2pi, 2pi),
        (-2pi, 2pi),
        (-2pi, 2pi),
        (-2pi, 2pi),
        (-2pi, 2pi),
        (-2pi, 2pi),
    ]
    fast_start = arg_bool("--fast-start", false)
    joint_limits = fast_start ? joint_limits_fallback : load_joint_limits_from_python(joint_limits_fallback)
    if fast_start
        println("Fast start enabled: skipping Python joint-limit bootstrap (using fallback/observed limits).")
    end
    joint_names = ["q0_base", "q1_shldr", "q2_elbow", "q3_wrist1", "q4_wrist2", "q5_wrist3"]
    observed_limits_path = arg_value("--observed-joint-limits-file", joinpath("traces", "observed_joint_limits.json"))
    joint_limits, observed_joint_min, observed_joint_max = load_observed_joint_limits_from_disk(observed_limits_path, joint_limits)
    limits_dirty = false
    last_limits_save_time = Ref(time())
    latest_left_q = fill(NaN, 6)
    latest_right_q = fill(NaN, 6)

    left_joint_tracker_text = Observable("LEFT joint range tracker awaiting data...")
    right_joint_tracker_text = Observable("RIGHT joint range tracker awaiting data...")
    joint_warning_text = Observable("Joint range status: awaiting data")
    joint_warning_color = Observable(RGBf(0.85, 0.85, 0.1))

    Label(side_panel[1, 1], "Joint Limit Tracker", halign = :left)
    Label(side_panel[2, 1], left_joint_tracker_text, halign = :left, tellwidth = false)
    Label(side_panel[3, 1], right_joint_tracker_text, halign = :left, tellwidth = false)
    Label(side_panel[4, 1], joint_warning_text, color = joint_warning_color, halign = :left, tellwidth = false)
    colsize!(fig.layout, 1, Relative(0.86))
    colsize!(fig.layout, 2, Relative(0.14))
    rowsize!(fig.layout, 1, Relative(1.0))
    rowsize!(side_panel, 1, Relative(0.06))
    rowsize!(side_panel, 2, Relative(0.25))
    rowsize!(side_panel, 3, Relative(0.25))
    rowsize!(side_panel, 4, Relative(0.10))
    rowsize!(side_panel, 5, Relative(0.08))
    rowsize!(side_panel, 6, Relative(0.08))
    rowsize!(side_panel, 7, Relative(0.18))

    # Three observables: left TCP, right TCP, and vision detections.
    left_tcp_data_obs = Observable((Point3f[], RGBf[]))
    right_tcp_data_obs = Observable((Point3f[], RGBf[]))
    vision_data_obs = Observable((Point3f[], RGBAf[]))

    scatter!(ax, @lift($left_tcp_data_obs[1]); color = @lift($left_tcp_data_obs[2]), markersize = 5, marker = :circle, label = "Left TCP")
    scatter!(ax, @lift($right_tcp_data_obs[1]); color = @lift($right_tcp_data_obs[2]), markersize = 5, marker = :utriangle, label = "Right TCP")
    scatter!(ax, @lift($vision_data_obs[1]); color = @lift($vision_data_obs[2]), markersize = 10, marker = :rect, label = "Vision")

    axislegend(ax, position = :lt)

    screen = GLMakie.Screen(
        start_renderloop = true,
        fullscreen = false,
        resizable = !lock_windowed,
        title = lock_windowed ? "Live Plot Runner (Windowed Lock)" : "Live Plot Runner",
    )
    display(screen, fig)
    if lock_windowed
        enforce_window_lock!(screen)
        println("Windowed lock enabled: fullscreen/maximize disabled to avoid GLFW window-state lockups on Windows.")
    end

    # Single TCP listener receives all pose/vision/waypoint packets from Python.
    server = listen(host, port)
    println("Listening on ", host, ":", port, " (TCP stream: poses + vision + waypoints)")
    println("Waiting for TCP stream client connection...")
    sock = accept(server)
    println("TCP stream client connected.")
    println("Close the plot window or press Ctrl+C to stop.")
    println("Named waypoint CSV: ", named_waypoints_csv)
    println("Observed joint limits file: ", observed_limits_path)

    ensure_named_waypoints_header(active_named_waypoints_csv)

    # Left TCP pose data (arm/base frame)
    left_tcp_xs = Float32[];  left_tcp_ys = Float32[];  left_tcp_zs = Float32[]
    # Left TCP pose data (shared global task frame)
    left_tcp_global_xs = Float32[];  left_tcp_global_ys = Float32[];  left_tcp_global_zs = Float32[]
    left_tcp_rxs = Float64[]; left_tcp_rys = Float64[]; left_tcp_rzs = Float64[]
    left_tcp_timestamps = Float64[]
    left_q0 = Float64[]; left_q1 = Float64[]; left_q2 = Float64[]
    left_q3 = Float64[]; left_q4 = Float64[]; left_q5 = Float64[]
    left_task_xs = Float32[]; left_task_ys = Float32[]; left_task_zs = Float32[]
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
    right_q0 = Float64[]; right_q1 = Float64[]; right_q2 = Float64[]
    right_q3 = Float64[]; right_q4 = Float64[]; right_q5 = Float64[]
    right_task_xs = Float32[]; right_task_ys = Float32[]; right_task_zs = Float32[]
    right_tcp_cols = RGBf[]
    right_tcp_rx_min = Inf;  right_tcp_rx_max = -Inf
    right_tcp_ry_min = Inf;  right_tcp_ry_max = -Inf
    right_tcp_rz_min = Inf;  right_tcp_rz_max = -Inf
    right_tcp_packet_count = 0

    # Vision data
    vision_xs = Float32[];  vision_ys = Float32[];  vision_zs = Float32[]
    vision_timestamps = Float64[]
    vision_labels = String[]
    vision_color_names = String[]
    vision_cols = RGBAf[]
    vision_packet_count = 0

    # Color map for vision detections by color name
    color_map = Dict(
        "red"    => RGBAf(0.9, 0.1, 0.1, 1.0),
        "yellow" => RGBAf(0.9, 0.85, 0.0, 1.0),
        "green"  => RGBAf(0.1, 0.8, 0.1, 1.0),
        "blue"   => RGBAf(0.1, 0.3, 0.9, 1.0),
        "purple" => RGBAf(0.6, 0.1, 0.8, 1.0),
        "tan"    => RGBAf(0.82, 0.71, 0.55, 1.0),
    )
    color_priority = ["red", "yellow", "green", "blue", "purple", "tan"]
    stale_vision_color = RGBAf(0.45, 0.45, 0.45, 0.5)
    unknown_latest_vision_color = RGBAf(0.92, 0.92, 0.92, 1.0)

    function resolve_base_color_name(raw_name::AbstractString)
        txt = lowercase(strip(String(raw_name)))
        if isempty(txt)
            return "unknown"
        end
        for cname in color_priority
            if occursin(cname, txt)
                return cname
            end
        end
        return txt
    end

    function refresh_vision_colors!()
        empty!(vision_cols)
        n = length(vision_color_names)
        if n == 0
            return
        end
        for i in 1:n
            if i == n
                cname = resolve_base_color_name(vision_color_names[i])
                push!(vision_cols, get(color_map, cname, unknown_latest_vision_color))
            else
                push!(vision_cols, stale_vision_color)
            end
        end
    end

    bootstrap = load_bootstrap_trace_jld2(bootstrap_jld2_path)
    if bootstrap !== nothing
        append!(left_tcp_xs, bootstrap["left_tcp_x"])
        append!(left_tcp_ys, bootstrap["left_tcp_y"])
        append!(left_tcp_zs, bootstrap["left_tcp_z"])
        append!(left_tcp_global_xs, bootstrap["left_global_x"])
        append!(left_tcp_global_ys, bootstrap["left_global_y"])
        append!(left_tcp_global_zs, bootstrap["left_global_z"])
        append!(left_task_xs, bootstrap["left_task_x"])
        append!(left_task_ys, bootstrap["left_task_y"])
        append!(left_task_zs, bootstrap["left_task_z"])
        append!(left_tcp_rxs, bootstrap["left_rx"])
        append!(left_tcp_rys, bootstrap["left_ry"])
        append!(left_tcp_rzs, bootstrap["left_rz"])
        append!(left_tcp_timestamps, bootstrap["left_ts"])
        append!(left_tcp_cols, bootstrap["left_cols"])

        append!(right_tcp_xs, bootstrap["right_tcp_x"])
        append!(right_tcp_ys, bootstrap["right_tcp_y"])
        append!(right_tcp_zs, bootstrap["right_tcp_z"])
        append!(right_tcp_global_xs, bootstrap["right_global_x"])
        append!(right_tcp_global_ys, bootstrap["right_global_y"])
        append!(right_tcp_global_zs, bootstrap["right_global_z"])
        append!(right_task_xs, bootstrap["right_task_x"])
        append!(right_task_ys, bootstrap["right_task_y"])
        append!(right_task_zs, bootstrap["right_task_z"])
        append!(right_tcp_rxs, bootstrap["right_rx"])
        append!(right_tcp_rys, bootstrap["right_ry"])
        append!(right_tcp_rzs, bootstrap["right_rz"])
        append!(right_tcp_timestamps, bootstrap["right_ts"])
        append!(right_tcp_cols, bootstrap["right_cols"])

        append!(vision_xs, bootstrap["vision_x"])
        append!(vision_ys, bootstrap["vision_y"])
        append!(vision_zs, bootstrap["vision_z"])
        append!(vision_timestamps, bootstrap["vision_ts"])
        append!(vision_labels, bootstrap["vision_labels"])
        append!(vision_color_names, bootstrap["vision_color_names"])
        refresh_vision_colors!()

        append!(left_q0, bootstrap["left_q0"])
        append!(left_q1, bootstrap["left_q1"])
        append!(left_q2, bootstrap["left_q2"])
        append!(left_q3, bootstrap["left_q3"])
        append!(left_q4, bootstrap["left_q4"])
        append!(left_q5, bootstrap["left_q5"])
        append!(right_q0, bootstrap["right_q0"])
        append!(right_q1, bootstrap["right_q1"])
        append!(right_q2, bootstrap["right_q2"])
        append!(right_q3, bootstrap["right_q3"])
        append!(right_q4, bootstrap["right_q4"])
        append!(right_q5, bootstrap["right_q5"])

        if !isempty(left_tcp_global_xs)
            left_tcp_data_obs[] = (Point3f.(left_tcp_global_xs, left_tcp_global_ys, left_tcp_global_zs), copy(left_tcp_cols))
            left_tcp_packet_count = length(left_tcp_global_xs)
            left_tcp_rx_min = minimum(left_tcp_rxs)
            left_tcp_rx_max = maximum(left_tcp_rxs)
            left_tcp_ry_min = minimum(left_tcp_rys)
            left_tcp_ry_max = maximum(left_tcp_rys)
            left_tcp_rz_min = minimum(left_tcp_rzs)
            left_tcp_rz_max = maximum(left_tcp_rzs)
            if length(left_q0) > 0
                latest_left_q .= [left_q0[end], left_q1[end], left_q2[end], left_q3[end], left_q4[end], left_q5[end]]
            end
        end

        if !isempty(right_tcp_global_xs)
            right_tcp_data_obs[] = (Point3f.(right_tcp_global_xs, right_tcp_global_ys, right_tcp_global_zs), copy(right_tcp_cols))
            right_tcp_packet_count = length(right_tcp_global_xs)
            right_tcp_rx_min = minimum(right_tcp_rxs)
            right_tcp_rx_max = maximum(right_tcp_rxs)
            right_tcp_ry_min = minimum(right_tcp_rys)
            right_tcp_ry_max = maximum(right_tcp_rys)
            right_tcp_rz_min = minimum(right_tcp_rzs)
            right_tcp_rz_max = maximum(right_tcp_rzs)
            if length(right_q0) > 0
                latest_right_q .= [right_q0[end], right_q1[end], right_q2[end], right_q3[end], right_q4[end], right_q5[end]]
            end
        end

        if !isempty(vision_xs)
            vision_data_obs[] = (Point3f.(vision_xs, vision_ys, vision_zs), copy(vision_cols))
            vision_packet_count = length(vision_xs)
        end

        if !isempty(left_tcp_global_xs) || !isempty(right_tcp_global_xs) || !isempty(vision_xs)
            autolimits!(ax)
        end

        left_block, left_warnings = format_joint_tracker_block("LEFT", latest_left_q, joint_limits, joint_names)
        right_block, right_warnings = format_joint_tracker_block("RIGHT", latest_right_q, joint_limits, joint_names)
        left_joint_tracker_text[] = left_block
        right_joint_tracker_text[] = right_block

        all_warnings = vcat(left_warnings, right_warnings)
        if isempty(all_warnings)
            joint_warning_text[] = "Joint range status: OK"
            joint_warning_color[] = RGBf(0.2, 0.8, 0.2)
        else
            joint_warning_text[] = "WARNING: " * join(all_warnings, " | ")
            joint_warning_color[] = RGBf(0.9, 0.25, 0.25)
        end
    end

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
        task_display = !isempty(strip(active_task_name)) ? active_task_name : (isempty(strip(active_task_id)) ? "<unknown-task>" : active_task_id)
        dep_display = isempty(strip(active_dependent_item_label)) ? "none" : active_dependent_item_label
        waypoint_status.text[] = "Saved waypoint '" * waypoint_name * "' for " * task_display * " (dep=" * dep_display * "). Pending waypoint marks: " * string(length(pending_waypoints))
        waypoint_name_box.displayed_string[] = nothing
    end

    # === PURE SINGLE-THREAD EVENT LOOP ===
    # - Single Julia thread enforced (julia --threads 1)
    # - TCP stream packets are newline-delimited JSON from Python
    # - Explicit yield() ensures GLMakie's internal renderloop gets CPU cycles
    # - No @async/@spawn/@Task to avoid fragile async+renderloop interactions
    # - All data updates atomic via tuple observables (prevents sync races)

    tcp_rx_buffer = ""

    Base.exit_on_sigint(false)
    try
        while isopen(screen)
            packet_lines = String[]
            if bytesavailable(sock) > 0
                chunk = try
                    String(readavailable(sock))
                catch
                    ""
                end
                if !isempty(chunk)
                    tcp_rx_buffer *= chunk
                    while true
                        nl = findfirst('\n', tcp_rx_buffer)
                        nl === nothing && break
                        line = strip(tcp_rx_buffer[1:(nl - 1)])
                        if nl >= lastindex(tcp_rx_buffer)
                            tcp_rx_buffer = ""
                        else
                            tcp_rx_buffer = tcp_rx_buffer[(nl + 1):end]
                        end
                        if !isempty(line)
                            push!(packet_lines, line)
                        end
                    end
                end
            elseif eof(sock)
                println("TCP stream disconnected.")
                break
            end

            had_packets = !isempty(packet_lines)
            for raw_line in packet_lines
                pkt = try JSON3.read(raw_line) catch; nothing end

                if pkt !== nothing
                    # --- Waypoint mark packet ---
                    if haskey(pkt, :packet_type) && String(pkt.packet_type) == "waypoint_mark"
                        packet_task_id = String(get(pkt, :task_id, ""))
                        packet_task_name = String(get(pkt, :task_name, ""))
                        packet_dependent_item_label = String(get(pkt, :dependent_item_label, ""))

                        if !isempty(packet_task_id)
                            active_task_id = packet_task_id
                        end
                        if !isempty(packet_task_name)
                            active_task_name = packet_task_name
                        end
                        if !isempty(packet_dependent_item_label)
                            active_dependent_item_label = packet_dependent_item_label
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
                        left_task = haskey(pkt, :left_task_xyz) ? collect(Float64.(pkt.left_task_xyz)) : Float64[]
                        right_task = haskey(pkt, :right_task_xyz) ? collect(Float64.(pkt.right_task_xyz)) : Float64[]
                        left_global = haskey(pkt, :left_global_xyz) ? collect(Float64.(pkt.left_global_xyz)) : left_task
                        right_global = haskey(pkt, :right_global_xyz) ? collect(Float64.(pkt.right_global_xyz)) : right_task
                        tracked_items_json = haskey(pkt, :tracked_items) ? JSON3.write(pkt.tracked_items) : ""

                        row = Dict{String, Any}(
                            "waypoint_index" => get(pkt, :waypoint_index, 0),
                            "task_id" => (!isempty(packet_task_id) ? packet_task_id : active_task_id),
                            "task_name" => (!isempty(packet_task_name) ? packet_task_name : active_task_name),
                            "dependent_item_label" => (!isempty(packet_dependent_item_label) ? packet_dependent_item_label : active_dependent_item_label),
                            "left_gripper_open" => get(pkt, :left_gripper_open, ""),
                            "right_gripper_open" => get(pkt, :right_gripper_open, ""),
                            "left_gripper_open_pct" => get(pkt, :left_gripper_open_pct, ""),
                            "right_gripper_open_pct" => get(pkt, :right_gripper_open_pct, ""),
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
                            "left_task_x" => length(left_task) >= 1 ? left_task[1] : "",
                            "left_task_y" => length(left_task) >= 2 ? left_task[2] : "",
                            "left_task_z" => length(left_task) >= 3 ? left_task[3] : "",
                            "right_task_x" => length(right_task) >= 1 ? right_task[1] : "",
                            "right_task_y" => length(right_task) >= 2 ? right_task[2] : "",
                            "right_task_z" => length(right_task) >= 3 ? right_task[3] : "",
                            "left_global_x" => length(left_global) >= 1 ? left_global[1] : "",
                            "left_global_y" => length(left_global) >= 2 ? left_global[2] : "",
                            "left_global_z" => length(left_global) >= 3 ? left_global[3] : "",
                            "right_global_x" => length(right_global) >= 1 ? right_global[1] : "",
                            "right_global_y" => length(right_global) >= 2 ? right_global[2] : "",
                            "right_global_z" => length(right_global) >= 3 ? right_global[3] : "",
                            "tracked_items_json" => tracked_items_json,
                            "waypoint_mark_time" => get(pkt, :waypoint_mark_time, ""),
                        )
                        push!(pending_waypoints, row)
                        task_display = !isempty(strip(active_task_name)) ? active_task_name : (isempty(strip(active_task_id)) ? "<unknown-task>" : active_task_id)
                        dep_display = isempty(strip(active_dependent_item_label)) ? "none" : active_dependent_item_label
                        waypoint_status.text[] = "Pending waypoint marks: " * string(length(pending_waypoints)) * " | task=" * task_display * " dep=" * dep_display

                    # --- Left/Right TCP pose packet ---
                    elseif haskey(pkt, :left_actual_TCP_pose) || haskey(pkt, :right_actual_TCP_pose) || haskey(pkt, :actual_TCP_pose)
                        if haskey(pkt, :left_actual_TCP_pose) || haskey(pkt, :actual_TCP_pose)
                            left_pose = haskey(pkt, :left_actual_TCP_pose) ? pkt.left_actual_TCP_pose : pkt.actual_TCP_pose
                            if length(left_pose) >= 6
                                x_arm, y_arm, z_arm = Float32(left_pose[1]), Float32(left_pose[2]), Float32(left_pose[3])
                                if haskey(pkt, :left_global_xyz) && length(pkt.left_global_xyz) >= 3
                                    gx, gy, gz = Float64(pkt.left_global_xyz[1]), Float64(pkt.left_global_xyz[2]), Float64(pkt.left_global_xyz[3])
                                else
                                    gx, gy, gz = base_to_global_task_xyz((Float64(x_arm), Float64(y_arm), Float64(z_arm)), :left)
                                end
                                if haskey(pkt, :left_task_xyz) && length(pkt.left_task_xyz) >= 3
                                    tx, ty, tz = Float32(pkt.left_task_xyz[1]), Float32(pkt.left_task_xyz[2]), Float32(pkt.left_task_xyz[3])
                                else
                                    tx, ty, tz = Float32(gx), Float32(gy), Float32(gz)
                                end
                                x, y, z = Float32(gx), Float32(gy), Float32(gz)
                                rx, ry, rz = Float64(left_pose[4]), Float64(left_pose[5]), Float64(left_pose[6])

                                if isfinite(x_arm) && isfinite(y_arm) && isfinite(z_arm) && isfinite(x) && isfinite(y) && isfinite(z)
                                    push!(left_tcp_xs, x_arm); push!(left_tcp_ys, y_arm); push!(left_tcp_zs, z_arm)
                                    push!(left_tcp_global_xs, x); push!(left_tcp_global_ys, y); push!(left_tcp_global_zs, z)
                                    push!(left_task_xs, tx); push!(left_task_ys, ty); push!(left_task_zs, tz)
                                    push!(left_tcp_rxs, rx); push!(left_tcp_rys, ry); push!(left_tcp_rzs, rz)
                                    ts = Float64(get(pkt, :left_timestamp, get(pkt, :timestamp, 0.0)))
                                    push!(left_tcp_timestamps, ts)
                                    if haskey(pkt, :left_actual_q) && length(pkt.left_actual_q) >= 6
                                        push!(left_q0, Float64(pkt.left_actual_q[1])); push!(left_q1, Float64(pkt.left_actual_q[2])); push!(left_q2, Float64(pkt.left_actual_q[3]))
                                        push!(left_q3, Float64(pkt.left_actual_q[4])); push!(left_q4, Float64(pkt.left_actual_q[5])); push!(left_q5, Float64(pkt.left_actual_q[6]))
                                        latest_left_q .= [
                                            Float64(pkt.left_actual_q[1]),
                                            Float64(pkt.left_actual_q[2]),
                                            Float64(pkt.left_actual_q[3]),
                                            Float64(pkt.left_actual_q[4]),
                                            Float64(pkt.left_actual_q[5]),
                                            Float64(pkt.left_actual_q[6]),
                                        ]
                                        if update_limits_from_sample!(joint_limits, observed_joint_min, observed_joint_max, latest_left_q)
                                            limits_dirty = true
                                            now_s = time()
                                            if now_s - last_limits_save_time[] >= observed_limits_save_interval_s
                                                try
                                                    write_observed_joint_limits(observed_limits_path, joint_names, joint_limits, observed_joint_min, observed_joint_max)
                                                    limits_dirty = false
                                                    last_limits_save_time[] = now_s
                                                catch e
                                                    println("Warning: could not write observed joint limits: ", e)
                                                end
                                            end
                                        end
                                    else
                                        push!(left_q0, NaN); push!(left_q1, NaN); push!(left_q2, NaN); push!(left_q3, NaN); push!(left_q4, NaN); push!(left_q5, NaN)
                                    end

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
                                        popfirst!(left_task_xs); popfirst!(left_task_ys); popfirst!(left_task_zs)
                                        popfirst!(left_tcp_rxs); popfirst!(left_tcp_rys); popfirst!(left_tcp_rzs)
                                        popfirst!(left_tcp_timestamps)
                                        popfirst!(left_q0); popfirst!(left_q1); popfirst!(left_q2); popfirst!(left_q3); popfirst!(left_q4); popfirst!(left_q5)
                                    end

                                    left_tcp_packet_count += 1
                                    if left_tcp_packet_count % packet_log_every == 0
                                        println("[Left TCP #$left_tcp_packet_count] pos=($x, $y, $z) m  rot=($rx, $ry, $rz) rad")
                                    end

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
                                if haskey(pkt, :right_global_xyz) && length(pkt.right_global_xyz) >= 3
                                    gx, gy, gz = Float64(pkt.right_global_xyz[1]), Float64(pkt.right_global_xyz[2]), Float64(pkt.right_global_xyz[3])
                                else
                                    gx, gy, gz = base_to_global_task_xyz((Float64(x_arm), Float64(y_arm), Float64(z_arm)), :right)
                                end
                                if haskey(pkt, :right_task_xyz) && length(pkt.right_task_xyz) >= 3
                                    tx, ty, tz = Float32(pkt.right_task_xyz[1]), Float32(pkt.right_task_xyz[2]), Float32(pkt.right_task_xyz[3])
                                else
                                    tx, ty, tz = Float32(gx), Float32(gy), Float32(gz)
                                end
                                x, y, z = Float32(gx), Float32(gy), Float32(gz)
                                rx, ry, rz = Float64(right_pose[4]), Float64(right_pose[5]), Float64(right_pose[6])

                                if isfinite(x_arm) && isfinite(y_arm) && isfinite(z_arm) && isfinite(x) && isfinite(y) && isfinite(z)
                                    push!(right_tcp_xs, x_arm); push!(right_tcp_ys, y_arm); push!(right_tcp_zs, z_arm)
                                    push!(right_tcp_global_xs, x); push!(right_tcp_global_ys, y); push!(right_tcp_global_zs, z)
                                    push!(right_task_xs, tx); push!(right_task_ys, ty); push!(right_task_zs, tz)
                                    push!(right_tcp_rxs, rx); push!(right_tcp_rys, ry); push!(right_tcp_rzs, rz)
                                    ts = Float64(get(pkt, :right_timestamp, get(pkt, :timestamp, 0.0)))
                                    push!(right_tcp_timestamps, ts)
                                    if haskey(pkt, :right_actual_q) && length(pkt.right_actual_q) >= 6
                                        push!(right_q0, Float64(pkt.right_actual_q[1])); push!(right_q1, Float64(pkt.right_actual_q[2])); push!(right_q2, Float64(pkt.right_actual_q[3]))
                                        push!(right_q3, Float64(pkt.right_actual_q[4])); push!(right_q4, Float64(pkt.right_actual_q[5])); push!(right_q5, Float64(pkt.right_actual_q[6]))
                                        latest_right_q .= [
                                            Float64(pkt.right_actual_q[1]),
                                            Float64(pkt.right_actual_q[2]),
                                            Float64(pkt.right_actual_q[3]),
                                            Float64(pkt.right_actual_q[4]),
                                            Float64(pkt.right_actual_q[5]),
                                            Float64(pkt.right_actual_q[6]),
                                        ]
                                        if update_limits_from_sample!(joint_limits, observed_joint_min, observed_joint_max, latest_right_q)
                                            limits_dirty = true
                                            now_s = time()
                                            if now_s - last_limits_save_time[] >= observed_limits_save_interval_s
                                                try
                                                    write_observed_joint_limits(observed_limits_path, joint_names, joint_limits, observed_joint_min, observed_joint_max)
                                                    limits_dirty = false
                                                    last_limits_save_time[] = now_s
                                                catch e
                                                    println("Warning: could not write observed joint limits: ", e)
                                                end
                                            end
                                        end
                                    else
                                        push!(right_q0, NaN); push!(right_q1, NaN); push!(right_q2, NaN); push!(right_q3, NaN); push!(right_q4, NaN); push!(right_q5, NaN)
                                    end

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
                                        popfirst!(right_task_xs); popfirst!(right_task_ys); popfirst!(right_task_zs)
                                        popfirst!(right_tcp_rxs); popfirst!(right_tcp_rys); popfirst!(right_tcp_rzs)
                                        popfirst!(right_tcp_timestamps)
                                        popfirst!(right_q0); popfirst!(right_q1); popfirst!(right_q2); popfirst!(right_q3); popfirst!(right_q4); popfirst!(right_q5)
                                    end

                                    right_tcp_packet_count += 1
                                    if right_tcp_packet_count % packet_log_every == 0
                                        println("[Right TCP #$right_tcp_packet_count] pos=($x, $y, $z) m  rot=($rx, $ry, $rz) rad")
                                    end

                                    if !first_right_tcp_logged
                                        first_right_tcp_logged = true
                                        println("First RIGHT TCP pose received.")
                                    end
                                    if right_tcp_packet_count % refresh_every == 0 || !first_right_tcp_logged
                                        right_tcp_data_obs[] = (Point3f.(right_tcp_global_xs, right_tcp_global_ys, right_tcp_global_zs), copy(right_tcp_cols))
                                        autolimits!(ax)

                                        left_block, left_warnings = format_joint_tracker_block("LEFT", latest_left_q, joint_limits, joint_names)
                                        right_block, right_warnings = format_joint_tracker_block("RIGHT", latest_right_q, joint_limits, joint_names)
                                        left_joint_tracker_text[] = left_block
                                        right_joint_tracker_text[] = right_block

                                        all_warnings = vcat(left_warnings, right_warnings)
                                        if isempty(all_warnings)
                                            joint_warning_text[] = "Joint range status: OK"
                                            joint_warning_color[] = RGBf(0.2, 0.8, 0.2)
                                        else
                                            joint_warning_text[] = "WARNING: " * join(all_warnings, " | ")
                                            joint_warning_color[] = RGBf(0.9, 0.25, 0.25)
                                        end
                                    end
                                end
                            end
                        end

                    # --- Vision frame packet (all tracked items in FOV) ---
                    elseif haskey(pkt, :packet_type) && String(pkt.packet_type) == "vision_frame" && haskey(pkt, :detections)
                        frame_ts = Float64(get(pkt, :timestamp, 0.0))
                        for det in pkt.detections
                            if !haskey(det, :position)
                                continue
                            end
                            pos = det.position
                            if length(pos) < 3
                                continue
                            end

                            x, y, z = Float32(pos[1]), Float32(pos[2]), Float32(pos[3])

                            # For XZ-camera streams, keep Python-provided y (calibrated offset)
                            # and only fall back z from pos[2] when needed for legacy packets.
                            axis_pair_second = ""
                            if haskey(det, :axis_pair) && length(det.axis_pair) >= 2
                                axis_pair_second = lowercase(String(det.axis_pair[2]))
                            end
                            if axis_pair_second == "z"
                                z = abs(Float32(pos[3])) > Float32(1e-6) ? Float32(pos[3]) : Float32(pos[2])
                            end

                            if !(isfinite(x) && isfinite(y) && isfinite(z))
                                continue
                            end

                            push!(vision_xs, x); push!(vision_ys, y); push!(vision_zs, z)
                            push!(vision_timestamps, frame_ts)
                            label_name = haskey(det, :label) ? String(det.label) : ""
                            color_name = haskey(det, :color) ? String(det.color) : "unknown"
                            spec_key_log = haskey(det, :spec_key) ? Int(det.spec_key) : -1
                            push!(vision_labels, label_name)
                            push!(vision_color_names, color_name)

                            if length(vision_xs) > max_points
                                popfirst!(vision_xs); popfirst!(vision_ys); popfirst!(vision_zs)
                                popfirst!(vision_timestamps)
                                popfirst!(vision_labels)
                                popfirst!(vision_color_names)
                            end

                            refresh_vision_colors!()

                            vision_packet_count += 1
                            if vision_packet_count % packet_log_every == 0
                                println("[Vision #$vision_packet_count] pos=($x, $y, $z) m  label=$label_name color=$color_name spec_key=$spec_key_log")
                            end
                        end

                        if !first_vision_logged
                            first_vision_logged = true
                            println("First vision frame received.")
                        end
                        if vision_packet_count % refresh_every == 0 || !first_vision_logged
                            vision_data_obs[] = (Point3f.(vision_xs, vision_ys, vision_zs), copy(vision_cols))
                            autolimits!(ax)
                        end

                    # --- Legacy single-object vision packet ---
                    elseif haskey(pkt, :position)
                        pos = pkt.position
                        if length(pos) >= 3
                            x, y, z = Float32(pos[1]), Float32(pos[2]), Float32(pos[3])

                            if isfinite(x) && isfinite(y) && isfinite(z)
                                push!(vision_xs, x); push!(vision_ys, y); push!(vision_zs, z)
                                push!(vision_timestamps, Float64(get(pkt, :timestamp, 0.0)))

                                color_name = get(pkt, :color, "unknown")
                                push!(vision_labels, String(get(pkt, :label, "")))
                                push!(vision_color_names, color_name)

                                if length(vision_xs) > max_points
                                    popfirst!(vision_xs); popfirst!(vision_ys); popfirst!(vision_zs)
                                    popfirst!(vision_timestamps)
                                    popfirst!(vision_labels)
                                    popfirst!(vision_color_names)
                                end

                                refresh_vision_colors!()

                                vision_packet_count += 1
                                if vision_packet_count % packet_log_every == 0
                                    println("[Vision #$vision_packet_count] pos=($x, $y, $z) m  color=$color_name")
                                end

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
            if !had_packets && idle_sleep_s > 0.0
                sleep(idle_sleep_s)
            end
        end
    catch err
        err isa InterruptException || rethrow()
        println("Interrupted by user.")
    finally
        Base.disable_sigint() do
            try; close(sock); catch; end
            try; close(server); catch; end
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

        task_slug = safe_task_name_for_filename(active_task_name, active_task_id)
        save_path = joinpath(traces_dir, "trace_$(task_slug).jld2")
        try
            jldsave(save_path;
                # Existing keys preserved as arm/base-frame coordinates for backward compatibility.
                left_tcp_x = left_tcp_xs, left_tcp_y = left_tcp_ys, left_tcp_z = left_tcp_zs,
                left_task_x = left_task_xs, left_task_y = left_task_ys, left_task_z = left_task_zs,
                left_tcp_global_x = left_tcp_global_xs, left_tcp_global_y = left_tcp_global_ys, left_tcp_global_z = left_tcp_global_zs,
                left_tcp_rx = left_tcp_rxs, left_tcp_ry = left_tcp_rys, left_tcp_rz = left_tcp_rzs,
                left_tcp_timestamp = left_tcp_timestamps,
                left_q_0 = left_q0, left_q_1 = left_q1, left_q_2 = left_q2,
                left_q_3 = left_q3, left_q_4 = left_q4, left_q_5 = left_q5,
                right_tcp_x = right_tcp_xs, right_tcp_y = right_tcp_ys, right_tcp_z = right_tcp_zs,
                right_task_x = right_task_xs, right_task_y = right_task_ys, right_task_z = right_task_zs,
                right_tcp_global_x = right_tcp_global_xs, right_tcp_global_y = right_tcp_global_ys, right_tcp_global_z = right_tcp_global_zs,
                right_tcp_rx = right_tcp_rxs, right_tcp_ry = right_tcp_rys, right_tcp_rz = right_tcp_rzs,
                right_tcp_timestamp = right_tcp_timestamps,
                right_q_0 = right_q0, right_q_1 = right_q1, right_q_2 = right_q2,
                right_q_3 = right_q3, right_q_4 = right_q4, right_q_5 = right_q5,
                # Backward-compat aliases for older readers expecting tcp_* keys.
                tcp_x = left_tcp_xs, tcp_y = left_tcp_ys, tcp_z = left_tcp_zs,
                tcp_global_x = left_tcp_global_xs, tcp_global_y = left_tcp_global_ys, tcp_global_z = left_tcp_global_zs,
                tcp_rx = left_tcp_rxs, tcp_ry = left_tcp_rys, tcp_rz = left_tcp_rzs,
                tcp_timestamp = left_tcp_timestamps,
                vision_x = vision_xs, vision_y = vision_ys, vision_z = vision_zs,
                vision_timestamp = vision_timestamps,
                vision_label = vision_labels,
                vision_color = vision_color_names
            )
            println("Data saved to: $save_path  ",
                    "($(length(left_tcp_xs)) left TCP pts, $(length(right_tcp_xs)) right TCP pts, $(length(vision_xs)) vision pts)")
        catch e
            println("WARNING: could not save data: $e")
        end

        if limits_dirty
            try
                write_observed_joint_limits(observed_limits_path, joint_names, joint_limits, observed_joint_min, observed_joint_max)
                println("Observed joint limits updated at shutdown: ", observed_limits_path)
            catch e
                println("WARNING: could not save observed joint limits at shutdown: $e")
            end
        end
    end

    println("Live plot runner stopped.")
end

main()
