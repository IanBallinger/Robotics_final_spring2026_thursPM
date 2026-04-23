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
    max_points = arg_int("--max-points", 4000)
    refresh_every = arg_int("--refresh-every", 5)

    GLMakie.activate!()

    fig = Figure(size = (1100, 800))
    ax = Axis3(
        fig[1, 1],
        xlabel = "X [m]",
        ylabel = "Y [m]",
        zlabel = "Z [m]",
        title = "Live TCP Position (XYZ), color from orientation (Rx,Ry,Rz)"
    )

    # Single observable holding (points, colors) because both come from the same underlying 6-dof data
    data_obs = Observable((Point3f[], RGBf[]))
    scatter!(ax, @lift($data_obs[1]); color = @lift($data_obs[2]), markersize = 5)

    screen = GLMakie.Screen(start_renderloop = true)
    display(screen, fig)

    sock = UDPSocket()
    bind(sock, host, port)
    println("Listening for UDP packets on ", host, ":", port)
    println("Close the plot window or press Ctrl+C to stop.")

    xs = Float32[]
    ys = Float32[]
    zs = Float32[]
    cols = RGBf[]

    rx_min = Inf
    rx_max = -Inf
    ry_min = Inf
    ry_max = -Inf
    rz_min = Inf
    rz_max = -Inf

    packet_count = 0
    first_packet_logged = false

    # Pure single-thread loop: blocking recv() yields cooperatively to other
    # tasks (including GLMakie's renderloop) while waiting for UDP input.
    Base.exit_on_sigint(false)
    try
        while isopen(screen)
            msg = String(recv(sock))
            msg == "__STOP__" && continue

            pkt = try
                JSON3.read(msg)
            catch
                nothing
            end

            if pkt === nothing || !haskey(pkt, :actual_TCP_pose)
                continue
            end
            pose = pkt.actual_TCP_pose
            length(pose) < 6 && continue

            x = Float32(pose[1])
            y = Float32(pose[2])
            z = Float32(pose[3])
            rx = Float64(pose[4])
            ry = Float64(pose[5])
            rz = Float64(pose[6])

            # Skip bad packets so one invalid value does not poison limits.
            if !(isfinite(x) && isfinite(y) && isfinite(z) && isfinite(rx) && isfinite(ry) && isfinite(rz))
                yield()
                continue
            end

            push!(xs, x)
            push!(ys, y)
            push!(zs, z)

            rx_min = min(rx_min, rx)
            rx_max = max(rx_max, rx)
            ry_min = min(ry_min, ry)
            ry_max = max(ry_max, ry)
            rz_min = min(rz_min, rz)
            rz_max = max(rz_max, rz)

            push!(cols, RGBf(
                norm01(rx, rx_min, rx_max),
                norm01(ry, ry_min, ry_max),
                norm01(rz, rz_min, rz_max)
            ))

            if length(xs) > max_points
                popfirst!(xs)
                popfirst!(ys)
                popfirst!(zs)
                popfirst!(cols)
            end

            packet_count += 1
            if !first_packet_logged
                println("First UDP packet received; live plot updating.")
                first_packet_logged = true
                data_obs[] = (Point3f.(xs, ys, zs), copy(cols))
                autolimits!(ax)
                yield()
                continue
            end

            if packet_count % refresh_every == 0
                data_obs[] = (Point3f.(xs, ys, zs), copy(cols))
                autolimits!(ax)
            end

            # Give GLMakie's renderloop a chance to draw frames under high UDP rate.
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
    end

    println("Live plot runner stopped.")
end

main()
