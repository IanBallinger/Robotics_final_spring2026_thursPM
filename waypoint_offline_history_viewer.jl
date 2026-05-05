if Threads.nthreads() > 1
    println("ERROR: Run with --threads 1 for GLMakie/GLFW stability on Windows.")
    exit(1)
end

using GLMakie
using JLD2

function arg_value(flag::String, default::String)
    idx = findfirst(==(flag), ARGS)
    if idx === nothing || idx == length(ARGS)
        return default
    end
    return ARGS[idx + 1]
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

    glfw.SetWindowAttrib(win, glfw.RESIZABLE, false)
    try
        glfw.SetWindowAttrib(win, glfw.MAXIMIZED, false)
    catch
    end

    ww, wh = glfw.GetWindowSize(win)
    glfw.SetWindowSizeLimits(win, ww, wh, ww, wh)
    return nothing
end

function jld2_read_numeric_vector(data, keys::Vector{String}, T::Type)
    for key in keys
        if haskey(data, key)
            raw = data[key]
            try
                return collect(T.(raw))
            catch
                try
                    return vec(collect(T.(raw)))
                catch
                end
            end
        end
    end
    return T[]
end

function jld2_read_string_vector(data, keys::Vector{String})
    for key in keys
        if haskey(data, key)
            raw = data[key]
            try
                return [string(v) for v in raw]
            catch
            end
        end
    end
    return String[]
end

function resolve_jld2_path()
    explicit = arg_value("--jld2", "")
    if !isempty(strip(explicit))
        return explicit
    end

    traces_dir = joinpath(pwd(), "traces")
    candidates = String[]
    if isdir(traces_dir)
        for name in readdir(traces_dir)
            if endswith(lowercase(name), ".jld2")
                push!(candidates, joinpath(traces_dir, name))
            end
        end
    end

    if !isempty(candidates)
        sort!(candidates, by = p -> stat(p).mtime, rev = true)
        return candidates[1]
    end

    fallback = "first_full_run.jld2"
    return fallback
end

function load_history(jld2_path::String)
    if !isfile(jld2_path)
        error("JLD2 file not found: $jld2_path")
    end

    data = JLD2.load(jld2_path)

    left_x = jld2_read_numeric_vector(data, ["left_tcp_global_x", "left_task_x", "left_tcp_x"], Float32)
    left_y = jld2_read_numeric_vector(data, ["left_tcp_global_y", "left_task_y", "left_tcp_y"], Float32)
    left_z = jld2_read_numeric_vector(data, ["left_tcp_global_z", "left_task_z", "left_tcp_z"], Float32)

    right_x = jld2_read_numeric_vector(data, ["right_tcp_global_x", "right_task_x", "right_tcp_x"], Float32)
    right_y = jld2_read_numeric_vector(data, ["right_tcp_global_y", "right_task_y", "right_tcp_y"], Float32)
    right_z = jld2_read_numeric_vector(data, ["right_tcp_global_z", "right_task_z", "right_tcp_z"], Float32)

    vision_x = jld2_read_numeric_vector(data, ["vision_x"], Float32)
    vision_y = jld2_read_numeric_vector(data, ["vision_y"], Float32)
    vision_z = jld2_read_numeric_vector(data, ["vision_z"], Float32)
    vision_labels = jld2_read_string_vector(data, ["vision_label"])
    vision_colors = jld2_read_string_vector(data, ["vision_color"])

    n_left = minimum([length(left_x), length(left_y), length(left_z)])
    n_right = minimum([length(right_x), length(right_y), length(right_z)])
    n_vis = minimum([length(vision_x), length(vision_y), length(vision_z)])

    left_x = left_x[1:n_left]; left_y = left_y[1:n_left]; left_z = left_z[1:n_left]
    right_x = right_x[1:n_right]; right_y = right_y[1:n_right]; right_z = right_z[1:n_right]
    vision_x = vision_x[1:n_vis]; vision_y = vision_y[1:n_vis]; vision_z = vision_z[1:n_vis]

    if isempty(vision_labels)
        vision_labels = ["" for _ in 1:n_vis]
    else
        vision_labels = vision_labels[1:min(n_vis, length(vision_labels))]
        if length(vision_labels) < n_vis
            append!(vision_labels, ["" for _ in 1:(n_vis - length(vision_labels))])
        end
    end

    if isempty(vision_colors)
        vision_colors = ["unknown" for _ in 1:n_vis]
    else
        vision_colors = vision_colors[1:min(n_vis, length(vision_colors))]
        if length(vision_colors) < n_vis
            append!(vision_colors, ["unknown" for _ in 1:(n_vis - length(vision_colors))])
        end
    end

    return Dict(
        "left_x" => left_x, "left_y" => left_y, "left_z" => left_z,
        "right_x" => right_x, "right_y" => right_y, "right_z" => right_z,
        "vision_x" => vision_x, "vision_y" => vision_y, "vision_z" => vision_z,
        "vision_labels" => vision_labels, "vision_colors" => vision_colors,
    )
end

function color_from_name(name::String)
    txt = lowercase(strip(name))
    if occursin("red", txt)
        return RGBAf(0.95, 0.15, 0.15, 0.95)
    elseif occursin("yellow", txt)
        return RGBAf(0.95, 0.9, 0.1, 0.95)
    elseif occursin("green", txt)
        return RGBAf(0.2, 0.85, 0.25, 0.95)
    elseif occursin("blue", txt)
        return RGBAf(0.2, 0.35, 0.95, 0.95)
    elseif occursin("purple", txt)
        return RGBAf(0.65, 0.2, 0.9, 0.95)
    elseif occursin("tan", txt)
        return RGBAf(0.82, 0.71, 0.55, 0.95)
    end
    return RGBAf(0.8, 0.8, 0.8, 0.8)
end

function main()
    jld2_path = resolve_jld2_path()
    lock_windowed = arg_bool("--lock-windowed", true)
    payload = load_history(jld2_path)

    left_pts = Point3f.(payload["left_x"], payload["left_y"], payload["left_z"])
    right_pts = Point3f.(payload["right_x"], payload["right_y"], payload["right_z"])
    vision_pts = Point3f.(payload["vision_x"], payload["vision_y"], payload["vision_z"])
    vision_cols = [color_from_name(c) for c in payload["vision_colors"]]

    fig = Figure(size = (1800, 1000))
    ax = Axis3(fig[1, 1], xlabel = "X [m]", ylabel = "Y [m]", zlabel = "Z [m]", title = "Offline Waypoint Context: Arm Traces + Historical Camera Data")

    lines!(ax, left_pts, color = RGBAf(0.15, 0.45, 0.95, 0.55), linewidth = 2.0, label = "Left trace")
    lines!(ax, right_pts, color = RGBAf(0.95, 0.45, 0.15, 0.55), linewidth = 2.0, label = "Right trace")
    # Keep full arm waypoint traces visible, but only show the current/present
    # vision-identified object position via the moving marker below.

    left_marker_obs = Observable(Point3f(0, 0, 0))
    right_marker_obs = Observable(Point3f(0, 0, 0))
    vision_marker_obs = Observable(Point3f(0, 0, 0))
    vision_label_obs = Observable("")

    scatter!(ax, left_marker_obs, color = RGBAf(0.05, 0.35, 0.95, 1.0), markersize = 18)
    scatter!(ax, right_marker_obs, color = RGBAf(0.95, 0.35, 0.05, 1.0), markersize = 18)
    scatter!(ax, vision_marker_obs, color = RGBAf(0.95, 0.95, 0.95, 1.0), markersize = 22, marker = :utriangle, label = "Present vision object")

    side = GridLayout(fig[1, 2])
    Label(side[1, 1], "Offline Historical Feed", halign = :left)
    vision_text = Label(side[2, 1], vision_label_obs, halign = :left)
    path_text = Label(side[3, 1], "JLD2: $jld2_path", halign = :left)

    slider_row = GridLayout(fig[2, 1:2])
    max_idx = max(length(left_pts), length(right_pts), length(vision_pts))
    Label(slider_row[1, 1], "Time/Sample", halign = :left)
    sample_slider = Slider(slider_row[1, 2], range = 1:max(1, max_idx), startvalue = 1)
    sample_text = Label(slider_row[1, 3], "Sample: 1", halign = :left)

    # Match live_plot_runner sizing pattern so plot area remains dominant/readable.
    colsize!(fig.layout, 1, Relative(0.86))
    colsize!(fig.layout, 2, Relative(0.14))
    rowsize!(fig.layout, 1, Relative(0.93))
    rowsize!(fig.layout, 2, Relative(0.07))
    rowsize!(side, 1, Relative(0.12))
    rowsize!(side, 2, Relative(0.40))
    rowsize!(side, 3, Relative(0.48))
    colsize!(slider_row, 1, Relative(0.12))
    colsize!(slider_row, 2, Relative(0.74))
    colsize!(slider_row, 3, Relative(0.14))

    on(sample_slider.value) do v
        i = Int(round(v))
        sample_text.text[] = "Sample: $i / $max_idx"

        if !isempty(left_pts)
            left_marker_obs[] = left_pts[clamp(i, 1, length(left_pts))]
        end
        if !isempty(right_pts)
            right_marker_obs[] = right_pts[clamp(i, 1, length(right_pts))]
        end
        if !isempty(vision_pts)
            vi = clamp(i, 1, length(vision_pts))
            vision_marker_obs[] = vision_pts[vi]
            lbl = payload["vision_labels"][vi]
            col = payload["vision_colors"][vi]
            vision_label_obs[] = "Vision sample label=$(lbl) color=$(col)"
        else
            vision_label_obs[] = "No historical camera points"
        end
    end

    axislegend(ax, position = :lt)
    GLMakie.activate!()
    screen = GLMakie.Screen(
        start_renderloop = true,
        fullscreen = false,
        resizable = !lock_windowed,
        title = lock_windowed ? "Offline Waypoint Viewer (Windowed Lock)" : "Offline Waypoint Viewer",
    )
    display(screen, fig)
    if lock_windowed
        enforce_window_lock!(screen)
        println("Windowed lock enabled: fullscreen/maximize disabled to avoid GLFW resize lockups on Windows.")
    end
    println("Loaded offline history from: $jld2_path")
    println("Close plot window to exit.")
    while isopen(screen)
        sleep(0.1)
    end
end

main()
