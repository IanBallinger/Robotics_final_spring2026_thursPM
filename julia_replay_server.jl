import Pkg

for pkg in ["HTTP", "JSON3", "JLD2"]
    if Base.find_package(pkg) === nothing
        Pkg.add(pkg)
    end
end

using HTTP
using JSON3
using JLD2

function arg_value(flag::String, default)
    idx = findfirst(==(flag), ARGS)
    if idx === nothing || idx == length(ARGS)
        return default
    end
    return ARGS[idx + 1]
end

function parse_int(v, default::Int)
    try
        return parse(Int, string(v))
    catch
        return default
    end
end

function to_f64(x)
    try
        return Float64(x)
    catch
        return NaN
    end
end

function normalize_waypoints(raw_poses)::Vector{Vector{Float64}}
    waypoints = Vector{Vector{Float64}}()
    for pose in raw_poses
        try
            vals = Float64[to_f64(v) for v in pose]
            if length(vals) >= 6 && all(isfinite, vals[1:6])
                push!(waypoints, vals[1:6])
            end
        catch
            # skip malformed pose
        end
    end
    return waypoints
end

function smooth_waypoints(in_pts::Vector{Vector{Float64}}, window::Int)::Vector{Vector{Float64}}
    n = length(in_pts)
    if n == 0 || window <= 1
        return in_pts
    end

    half = window ÷ 2
    out = Vector{Vector{Float64}}(undef, n)

    for i in 1:n
        j0 = max(1, i - half)
        j1 = min(n, i + half)
        count = j1 - j0 + 1
        acc = zeros(Float64, 6)

        for j in j0:j1
            @inbounds acc .+= in_pts[j]
        end

        out[i] = acc ./ count
    end

    return out
end

function downsample_waypoints(in_pts::Vector{Vector{Float64}}, step::Int)::Vector{Vector{Float64}}
    s = max(1, step)
    if s == 1 || isempty(in_pts)
        return in_pts
    end

    out = Vector{Vector{Float64}}()
    for i in 1:s:length(in_pts)
        push!(out, in_pts[i])
    end

    if out[end] != in_pts[end]
        push!(out, in_pts[end])
    end

    return out
end

function json_response(status::Int, payload)
    return HTTP.Response(
        status,
        ["Content-Type" => "application/json"],
        JSON3.write(payload),
    )
end

function error_response(status::Int, message::String)
    return json_response(status, Dict("error" => message))
end

function parse_request_json(req::HTTP.Request)
    try
        body = String(req.body)
        isempty(body) && return Dict{String, Any}()
        obj = JSON3.read(body)
        return Dict{String, Any}(pairs(obj))
    catch
        return nothing
    end
end

function plan_from_request(data::Dict{String, Any})
    raw = get(data, "raw_poses", Any[])
    if !(raw isa AbstractVector)
        return nothing, "raw_poses must be an array"
    end

    waypoints = normalize_waypoints(raw)
    if isempty(waypoints)
        return nothing, "no valid 6D poses in raw_poses"
    end

    planner_downsample = parse_int(get(data, "planner_downsample", 1), 1)
    smooth_window = parse_int(get(data, "smooth_window", 5), 5)

    ds = downsample_waypoints(waypoints, planner_downsample)
    smoothed = smooth_waypoints(ds, smooth_window)

    meta = Dict(
        "input_waypoints" => length(waypoints),
        "downsampled_waypoints" => length(ds),
        "output_waypoints" => length(smoothed),
        "planner_downsample" => planner_downsample,
        "smooth_window" => smooth_window,
        "trace_csv_path" => string(get(data, "trace_csv_path", "")),
        "trace_side" => string(get(data, "trace_side", "")),
    )

    return Dict("waypoints" => smoothed, "meta" => meta), ""
end

function read_numeric_vector(data, keys::Vector{String})::Vector{Float64}
    for key in keys
        if !haskey(data, key)
            continue
        end
        raw = data[key]
        if !(raw isa AbstractVector)
            continue
        end
        out = Float64[]
        for v in raw
            f = to_f64(v)
            if isfinite(f)
                push!(out, f)
            end
        end
        return out
    end
    return Float64[]
end

function plan_from_jld2_request(data::Dict{String, Any})
    jld2_path = strip(string(get(data, "jld2_path", "")))
    if isempty(jld2_path)
        return nothing, "jld2_path is required"
    end
    if !isfile(jld2_path)
        return nothing, "jld2_path not found: $(jld2_path)"
    end

    trace_side = lowercase(strip(string(get(data, "trace_side", "left"))))
    side = trace_side in ["right", "r"] ? "right" : "left"

    planner_downsample = parse_int(get(data, "planner_downsample", 1), 1)
    smooth_window = parse_int(get(data, "smooth_window", 5), 5)

    jld = try
        JLD2.load(jld2_path)
    catch e
        return nothing, "failed to read JLD2: $(e)"
    end

    x = read_numeric_vector(jld, side == "right" ? ["right_tcp_x"] : ["left_tcp_x", "tcp_x"])
    y = read_numeric_vector(jld, side == "right" ? ["right_tcp_y"] : ["left_tcp_y", "tcp_y"])
    z = read_numeric_vector(jld, side == "right" ? ["right_tcp_z"] : ["left_tcp_z", "tcp_z"])
    rx = read_numeric_vector(jld, side == "right" ? ["right_tcp_rx"] : ["left_tcp_rx", "tcp_rx"])
    ry = read_numeric_vector(jld, side == "right" ? ["right_tcp_ry"] : ["left_tcp_ry", "tcp_ry"])
    rz = read_numeric_vector(jld, side == "right" ? ["right_tcp_rz"] : ["left_tcp_rz", "tcp_rz"])

    n = minimum([length(x), length(y), length(z), length(rx), length(ry), length(rz)])
    if n == 0
        return nothing, "no valid pose data for side=$(side) in $(jld2_path)"
    end

    raw_poses = Vector{Vector{Float64}}()
    for i in 1:n
        push!(raw_poses, [x[i], y[i], z[i], rx[i], ry[i], rz[i]])
    end

    ds = downsample_waypoints(raw_poses, planner_downsample)
    smoothed = smooth_waypoints(ds, smooth_window)

    meta = Dict(
        "source" => "jld2",
        "jld2_path" => jld2_path,
        "trace_side" => side,
        "input_waypoints" => length(raw_poses),
        "downsampled_waypoints" => length(ds),
        "output_waypoints" => length(smoothed),
        "planner_downsample" => planner_downsample,
        "smooth_window" => smooth_window,
    )

    return Dict("waypoints" => smoothed, "meta" => meta), ""
end

function handler(req::HTTP.Request)
    method = String(req.method)
    path = HTTP.URIs.URI(req.target).path

    if method == "GET" && (path == "/" || path == "/health")
        return json_response(200, Dict(
            "ok" => true,
            "service" => "julia_replay_server",
            "routes" => [
                "GET /health",
                "POST /replay/plan",
                "POST /plan_replay",
                "POST /trace/plan_jld2",
            ],
        ))
    end

    if method == "POST" && (path == "/replay/plan" || path == "/plan_replay")
        data = parse_request_json(req)
        data === nothing && return error_response(400, "invalid JSON body")

        result, err = plan_from_request(data)
        result === nothing && return error_response(400, err)

        return json_response(200, result)
    end

    if method == "POST" && path == "/trace/plan_jld2"
        data = parse_request_json(req)
        data === nothing && return error_response(400, "invalid JSON body")

        result, err = plan_from_jld2_request(data)
        result === nothing && return error_response(400, err)

        return json_response(200, result)
    end

    return error_response(404, "unknown endpoint")
end

function main()
    host = arg_value("--host", "127.0.0.1")
    port = parse_int(arg_value("--port", "8081"), 8081)

    println("Starting Julia replay server on http://$(host):$(port)")
    println("Endpoints: GET /health, POST /replay/plan, POST /plan_replay, POST /trace/plan_jld2")

    HTTP.serve(handler, host, port; verbose=true)
end

main()
