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
for pkg in ["GLMakie", "Colors", "JSON3"]
    if Base.find_package(pkg) === nothing
        Pkg.add(pkg)
    end
end

using Sockets
using JSON3
using GLMakie
using Colors

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
    port_vision = 9998  # Camera vision port
    max_points = arg_int("--max-points", 4000)
    refresh_every = arg_int("--refresh-every", 5)

    GLMakie.activate!()

    fig = Figure(size = (1100, 800))
    ax = Axis3(
        fig[1, 1],
        xlabel = "X [m]",
        ylabel = "Y [m]",
        zlabel = "Z [m]",
        title = "Live TCP Position (XYZ) & Camera Vision Poses"
    )

    # Two observables: one for TCP poses (small), one for camera vision (large, different color)
    tcp_data_obs = Observable((Point3f[], RGBf[]))
    vision_data_obs = Observable((Point3f[], RGBf[]))
    
    # TCP poses: small markers, colors from orientation
    scatter!(ax, @lift($tcp_data_obs[1]); color = @lift($tcp_data_obs[2]), markersize = 5, label = "TCP Pose")
    
    # Camera vision poses: larger markers, colors based on position magnitude
    scatter!(ax, @lift($vision_data_obs[1]); color = @lift($vision_data_obs[2]), markersize = 8, marker = :rect, label = "Vision Pose")
    
    axislegend(ax, position = :rt)

    screen = GLMakie.Screen(start_renderloop = true)
    display(screen, fig)

    # Bind both sockets (UDP sockets will use non-blocking readavailable() in main loop)
    sock_tcp = UDPSocket()
    sock_vision = UDPSocket()
    bind(sock_tcp, host, port)
    bind(sock_vision, host, port_vision)
    println("Listening for TCP poses on ", host, ":", port)
    println("Listening for vision poses on ", host, ":", port_vision)
    println("Close the plot window or press Ctrl+C to stop.")

    # TCP data
    tcp_xs = Float32[]
    tcp_ys = Float32[]
    tcp_zs = Float32[]
    tcp_cols = RGBf[]

    tcp_rx_min = Inf
    tcp_rx_max = -Inf
    tcp_ry_min = Inf
    tcp_ry_max = -Inf
    tcp_rz_min = Inf
    tcp_rz_max = -Inf

    # Vision data
    vision_xs = Float32[]
    vision_ys = Float32[]
    vision_zs = Float32[]
    vision_cols = RGBf[]

    vision_mag_min = Inf
    vision_mag_max = -Inf

    first_tcp_logged = false
    first_vision_logged = false

    # === PURE SINGLE-THREAD EVENT LOOP ===
    # Based on learnings from docs/explorations/makie.md:
    # - Single Julia thread enforced (julia --threads 1)
    # - Non-blocking readavailable() for UDP sockets prevents blocking the renderloop
    # - Explicit yield() ensures GLMakie's internal renderloop gets CPU cycles
    # - No @async/@spawn/@Task to avoid fragile async+renderloop interactions
    # - All data updates atomic via tuple observables (prevents sync races)
    
    Base.exit_on_sigint(false)
    try
        while isopen(screen)
            # Try to receive from TCP socket (non-blocking)
            try
                data = readavailable(sock_tcp)
                if !isempty(data)
                    msg = String(data)
                    pkt = try JSON3.read(msg) catch nothing end
                    
                    if pkt !== nothing && haskey(pkt, :actual_TCP_pose)
                        pose = pkt.actual_TCP_pose
                        length(pose) >= 6 && begin
                            x = Float32(pose[1])
                            y = Float32(pose[2])
                            z = Float32(pose[3])
                            rx = Float64(pose[4])
                            ry = Float64(pose[5])
                            rz = Float64(pose[6])
                            
                            if isfinite(x) && isfinite(y) && isfinite(z) && isfinite(rx) && isfinite(ry) && isfinite(rz)
                                push!(tcp_xs, x)
                                push!(tcp_ys, y)
                                push!(tcp_zs, z)
                                
                                tcp_rx_min = min(tcp_rx_min, rx)
                                tcp_rx_max = max(tcp_rx_max, rx)
                                tcp_ry_min = min(tcp_ry_min, ry)
                                tcp_ry_max = max(tcp_ry_max, ry)
                                tcp_rz_min = min(tcp_rz_min, rz)
                                tcp_rz_max = max(tcp_rz_max, rz)
                                
                                push!(tcp_cols, RGBf(
                                    norm01(rx, tcp_rx_min, tcp_rx_max),
                                    norm01(ry, tcp_ry_min, tcp_ry_max),
                                    norm01(rz, tcp_rz_min, tcp_rz_max)
                                ))
                                
                                if length(tcp_xs) > max_points
                                    popfirst!(tcp_xs)
                                    popfirst!(tcp_ys)
                                    popfirst!(tcp_zs)
                                    popfirst!(tcp_cols)
                                end
                                
                                tcp_packet_count += 1
                                
                                # Log TCP data as sanity check
                                println("[TCP #$tcp_packet_count] Position: ($x, $y, $z) m | Rotation: ($rx, $ry, $rz) rad")
                                
                                if !first_tcp_logged
                                    println("First TCP pose received; live plot updating.")
                                    first_tcp_logged = true
                                    tcp_data_obs[] = (Point3f.(tcp_xs, tcp_ys, tcp_zs), copy(tcp_cols))
                                    autolimits!(ax)
                                elseif tcp_packet_count % refresh_every == 0
                                    tcp_data_obs[] = (Point3f.(tcp_xs, tcp_ys, tcp_zs), copy(tcp_cols))
                                    autolimits!(ax)
                                end
                            end
                        end
                    end
                end
            catch e
                e isa EOFError && nothing  # Expected when no data available
            end
            
            # Try to receive from Vision socket (non-blocking)
            try
                data = readavailable(sock_vision)
                if !isempty(data)
                    msg = String(data)
                    pkt = try JSON3.read(msg) catch nothing end
                    
                    if pkt !== nothing && haskey(pkt, :position)
                        pos = pkt.position
                        length(pos) >= 3 && begin
                            x = Float32(pos[1])
                            y = Float32(pos[2])
                            z = Float32(pos[3])
                            
                            if isfinite(x) && isfinite(y) && isfinite(z)
                                push!(vision_xs, x)
                                push!(vision_ys, y)
                                push!(vision_zs, z)
                                
                                # Color by distance from origin
                                mag = sqrt(x^2 + y^2 + z^2)
                                vision_mag_min = min(vision_mag_min, mag)
                                vision_mag_max = max(vision_mag_max, mag)
                                
                                push!(vision_cols, RGBf(
                                    norm01(Float64(x), Float64(vision_mag_min), Float64(vision_mag_max)),
                                    norm01(Float64(y), Float64(vision_mag_min), Float64(vision_mag_max)),
                                    norm01(Float64(z), Float64(vision_mag_min), Float64(vision_mag_max))
                                ))
                                
                                if length(vision_xs) > max_points
                                    popfirst!(vision_xs)
                                    popfirst!(vision_ys)
                                    popfirst!(vision_zs)
                                    popfirst!(vision_cols)
                                end
                                
                                vision_packet_count += 1
                                
                                # Log vision data as sanity check
                                color_name = get(pkt, :color, "unknown")
                                println("[Vision #$vision_packet_count] Position: ($x, $y, $z) m | Color: $color_name")
                                
                                if !first_vision_logged
                                    println("First vision pose received; visualizing camera detections.")
                                    first_vision_logged = true
                                    vision_data_obs[] = (Point3f.(vision_xs, vision_ys, vision_zs), copy(vision_cols))
                                    autolimits!(ax)
                                elseif vision_packet_count % refresh_every == 0
                                    vision_data_obs[] = (Point3f.(vision_xs, vision_ys, vision_zs), copy(vision_cols))
                                    autolimits!(ax)
                                end
                            end
                        end
                    end
                end
            catch e
                e isa EOFError && nothing  # Expected when no data available
            end
            
            # Give GLMakie's renderloop a chance to draw frames
            yield()
        end
    catch err
        err isa InterruptException || rethrow()
        println("Interrupted by user.")
    finally
        Base.disable_sigint() do
            try; close(sock_tcp); catch; end
            try; close(sock_vision); catch; end
            try; GLMakie.closeall(); catch; end
        end
        Base.exit_on_sigint(true)
    end

    println("Live plot runner stopped.")
end

main()
