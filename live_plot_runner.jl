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

function main()
    host = arg_host("127.0.0.1")
    port = arg_int("--port", 9999)
    max_points = arg_int("--max-points", 400000000)
    refresh_every = arg_int("--refresh-every", 5)
    output_path = arg_value("--jld2-file", arg_value("--output", ""))

    GLMakie.activate!()

    fig = Figure(size = (1100, 800))
    ax = Axis3(
        fig[1, 1],
        xlabel = "X [m]",
        ylabel = "Y [m]",
        zlabel = "Z [m]",
        title = "Live Left/Right TCP (XYZ) & Camera Vision Poses"
    )

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

    # Left TCP pose data
    left_tcp_xs = Float32[];  left_tcp_ys = Float32[];  left_tcp_zs = Float32[]
    left_tcp_rxs = Float64[]; left_tcp_rys = Float64[]; left_tcp_rzs = Float64[]
    left_tcp_timestamps = Float64[]
    left_tcp_cols = RGBf[]
    left_tcp_rx_min = Inf;  left_tcp_rx_max = -Inf
    left_tcp_ry_min = Inf;  left_tcp_ry_max = -Inf
    left_tcp_rz_min = Inf;  left_tcp_rz_max = -Inf
    left_tcp_packet_count = 0

    # Right TCP pose data
    right_tcp_xs = Float32[];  right_tcp_ys = Float32[];  right_tcp_zs = Float32[]
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
                    # --- Left/Right TCP pose packet ---
                    if haskey(pkt, :left_actual_TCP_pose) || haskey(pkt, :right_actual_TCP_pose) || haskey(pkt, :actual_TCP_pose)
                        if haskey(pkt, :left_actual_TCP_pose) || haskey(pkt, :actual_TCP_pose)
                            left_pose = haskey(pkt, :left_actual_TCP_pose) ? pkt.left_actual_TCP_pose : pkt.actual_TCP_pose
                            if length(left_pose) >= 6
                                x, y, z = Float32(left_pose[1]), Float32(left_pose[2]), Float32(left_pose[3])
                                rx, ry, rz = Float64(left_pose[4]), Float64(left_pose[5]), Float64(left_pose[6])

                                if isfinite(x) && isfinite(y) && isfinite(z)
                                    push!(left_tcp_xs, x); push!(left_tcp_ys, y); push!(left_tcp_zs, z)
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
                                        left_tcp_data_obs[] = (Point3f.(left_tcp_xs, left_tcp_ys, left_tcp_zs), copy(left_tcp_cols))
                                        autolimits!(ax)
                                    end
                                end
                            end
                        end

                        if haskey(pkt, :right_actual_TCP_pose)
                            right_pose = pkt.right_actual_TCP_pose
                            if length(right_pose) >= 6
                                x, y, z = Float32(right_pose[1]), Float32(right_pose[2]), Float32(right_pose[3])
                                rx, ry, rz = Float64(right_pose[4]), Float64(right_pose[5]), Float64(right_pose[6])

                                if isfinite(x) && isfinite(y) && isfinite(z)
                                    push!(right_tcp_xs, x); push!(right_tcp_ys, y); push!(right_tcp_zs, z)
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
                                        right_tcp_data_obs[] = (Point3f.(right_tcp_xs, right_tcp_ys, right_tcp_zs), copy(right_tcp_cols))
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
        save_path = if isempty(output_path)
            ts = replace(string(now()), ":" => "-", "." => "-")
            "live_plot_$(ts).jld2"
        else
            output_path
        end
        try
            jldsave(save_path;
                left_tcp_x = left_tcp_xs, left_tcp_y = left_tcp_ys, left_tcp_z = left_tcp_zs,
                left_tcp_rx = left_tcp_rxs, left_tcp_ry = left_tcp_rys, left_tcp_rz = left_tcp_rzs,
                left_tcp_timestamp = left_tcp_timestamps,
                right_tcp_x = right_tcp_xs, right_tcp_y = right_tcp_ys, right_tcp_z = right_tcp_zs,
                right_tcp_rx = right_tcp_rxs, right_tcp_ry = right_tcp_rys, right_tcp_rz = right_tcp_rzs,
                right_tcp_timestamp = right_tcp_timestamps,
                # Backward-compat aliases for older readers expecting tcp_* keys.
                tcp_x = left_tcp_xs, tcp_y = left_tcp_ys, tcp_z = left_tcp_zs,
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
