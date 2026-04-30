import Pkg

for pkg in ["HTTP", "JSON3"]
    if Base.find_package(pkg) === nothing
        Pkg.add(pkg)
    end
end

using HTTP
using JSON3

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

function handler(req::HTTP.Request)
    method = String(req.method)
    path = HTTP.URIs.URI(req.target).path

    if method == "GET" && (path == "/" || path == "/health")
        return json_response(200, Dict(
            "ok" => true,
            "service" => "julia_replay_server",
            "routes" => ["GET /health", "POST /replay/plan", "POST /plan_replay"],
        ))
    end

    if method == "POST" && (path == "/replay/plan" || path == "/plan_replay")
        data = parse_request_json(req)
        data === nothing && return error_response(400, "invalid JSON body")

        result, err = plan_from_request(data)
        result === nothing && return error_response(400, err)

        return json_response(200, result)
    end

    return error_response(404, "unknown endpoint")
end

function main()
    host = arg_value("--host", "127.0.0.1")
    port = parse_int(arg_value("--port", "8081"), 8081)

    println("Starting Julia replay server on http://$(host):$(port)")
    println("Endpoints: GET /health, POST /replay/plan, POST /plan_replay")

    HTTP.serve(handler, host, port; verbose=true)
end

main()
