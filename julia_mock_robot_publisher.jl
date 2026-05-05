using JSON3
using Dates

function arg_value(flag::String, default::String)
    idx = findfirst(==(flag), ARGS)
    if idx === nothing || idx == length(ARGS)
        return default
    end
    return ARGS[idx + 1]
end

function arg_float(flag::String, default::Float64)
    raw = arg_value(flag, string(default))
    try
        return parse(Float64, raw)
    catch
        return default
    end
end

function read_json_dict(path::String)
    if !isfile(path)
        return Dict{String, Any}()
    end
    try
        txt = read(path, String)
        obj = JSON3.read(txt)
        return Dict{String, Any}(pairs(obj))
    catch
        return Dict{String, Any}()
    end
end

function write_json_atomic(path::String, payload::Dict{String, Any})
    tmp = path * "." * string(getpid()) * "." * string(Threads.threadid()) * ".tmp"

    for _ in 1:4
        try
            open(tmp, "w") do io
                JSON3.pretty(io, payload)
            end
            mv(tmp, path; force = true)
            return
        catch
            try
                isfile(tmp) && rm(tmp; force = true)
            catch
            end
            sleep(0.01)
        end
    end

    # Fallback when rename replacement is transiently locked on Windows.
    open(path, "w") do io
        JSON3.pretty(io, payload)
    end
end

function clamp01(x)
    return min(1.0, max(0.0, x))
end

function main()
    state_path = arg_value("--state", joinpath("traces", "mock_robot_state.json"))
    hz = max(1.0, arg_float("--hz", 20.0))
    dt = 1.0 / hz

    mkpath(dirname(state_path))

    println("Julia mock robot publisher")
    println("State file: " * state_path)
    println("Rate: " * string(hz) * " Hz")
    println("Reading last Python command from state and writing simulated robot state.")
    println("Stop with Ctrl+C.")

    t0 = time()
    while true
        t = time() - t0
        data = read_json_dict(state_path)

        pose = get(data, "pose", Any[0.45, -0.20, 0.45, 2.20, -2.20, 0.00])
        q = get(data, "q", Any[0.00, -1.57, 1.57, -1.57, -1.57, 0.00])
        open_pct = Float64(get(data, "gripper_open_pct", 100.0))
        force_pct = Float64(get(data, "gripper_force_pct", 100.0))
        last_cmd = String(get(data, "last_command", "none"))

        p = [Float64(pose[min(i, length(pose))]) for i in 1:6]
        qv = [Float64(q[min(i, length(q))]) for i in 1:6]

        # Add a small deterministic dither so UI has visible live updates.
        wobble = 0.0025 * sin(2pi * 0.35 * t)
        p_live = copy(p)
        p_live[1] += wobble
        p_live[2] += 0.0015 * cos(2pi * 0.27 * t)
        p_live[6] += 0.01 * sin(2pi * 0.19 * t)

        q_live = copy(qv)
        q_live[1] += 0.01 * sin(2pi * 0.25 * t)
        q_live[6] += 0.015 * cos(2pi * 0.18 * t)

        payload = Dict{String, Any}(
            "timestamp" => time(),
            "iso_time" => Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS.sss"),
            "source" => "julia_mock_robot",
            "pose" => p_live,
            "q" => q_live,
            "gripper_open_pct" => clamp01(open_pct / 100.0) * 100.0,
            "gripper_force_pct" => clamp01(force_pct / 100.0) * 100.0,
            "last_command" => last_cmd,
            "sim_note" => "offline mock stream",
        )
        write_json_atomic(state_path, payload)

        sleep(dt)
    end
end

main()
