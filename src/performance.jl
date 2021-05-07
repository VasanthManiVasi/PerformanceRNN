export Performance, PerformanceEvent, encodeindex, decodeindex, perf2notes

struct PerformanceEvent
    event_type::Int
    event_value::Int

    function PerformanceEvent(type::Int, value::Int)
        if (type == NOTE_ON || type == NOTE_OFF)
            if !(MIN_MIDI_PITCH <= value <= MAX_MIDI_PITCH)
                error("The event has an invalid pitch value")
            end
        elseif type == TIME_SHIFT
            if !(value >= 0)
                error("The event has an invalid time shift value")
            end
        elseif type == VELOCITY
            if !(MIN_MIDI_VELOCITY <= value <= MAX_MIDI_VELOCITY)
                error("The event has an invalid velocity value")
            end
        else
            error("Performance event has invalid type")
        end

        new(type, value)
    end
end

function Base.show(io::IO, a::PerformanceEvent)
    if a.event_type == NOTE_ON
        s = "NOTE-ON"
    elseif a.event_type == NOTE_OFF
        s = "NOTE-OFF"
    elseif a.event_type == TIME_SHIFT
        s = "TIME-SHIFT"
    elseif a.event_type == VELOCITY
        s = "VELOCITY"
    else
        error("Performance event has invalid type")
    end

    s *= " $(a.event_value)"
    println(io, s)
end

struct Performance
    # Stores the ranges of each event
    event_ranges::Vector{Tuple{Int, Int ,Int}}
    num_classes::Int

    function Performance(;velocity_bins::Int = 32, max_shift_steps::Int = 100)
        event_ranges = [
            (NOTE_ON, MIN_MIDI_PITCH, MAX_MIDI_PITCH)
            (NOTE_OFF, MIN_MIDI_PITCH, MAX_MIDI_PITCH)
            (TIME_SHIFT, 1, max_shift_steps)
        ]
        velocity_bins > 0 && push!(event_ranges, (VELOCITY, 1, velocity_bins))
        num_classes = sum(map(range -> range[3] - range[2] + 1, event_ranges))
        new(event_ranges, num_classes)
    end
end

function encodeindex(event::PerformanceEvent, perfctx::Performance)
    offset = 0
    for (type, min, max) in perfctx.event_ranges
        if event.event_type == type
            return offset + event.event_value - min
        end
        offset += (max - min + 1)
    end
end

function decodeindex(idx::Int, perfctx::Performance)
    offset = 0
    for (type, min, max) in perfctx.event_ranges
        if idx < offset + (max - min + 1)
            return PerformanceEvent(type, min + idx - offset)
        end
        offset += (max - min + 1)
    end
end

function perf2notes(events::Vector{PerformanceEvent}, perfctx::Performance)
    # TODO:
    #   Declare the defaults in constants.jl
    ppq = 220 # default by magenta
    qpm = 120 # default by magenta
    ticks_per_step = second_to_tick(1, qpm, ppq) ÷ perfctx.steps_per_second # sps = 100
    notes = Notes(tpq = ppq)
    pitchmap = Dict{Int, Vector{Tuple{Int, Int}}}()
    step = 0
    velocity = 64
    for event in events
        if event.event_type == NOTE_ON
            if event.event_value in keys(pitchmap)
                push!(pitchmap[event.event_value], (step, velocity))
            else
                pitchmap[event.event_value] = [(step, velocity)]
            end
        elseif event.event_type == NOTE_OFF
            if event.event_value in keys(pitchmap)
                start_step, vel = popfirst!(pitchmap[event.event_value])

                if isempty(pitchmap[event.event_value])
                    delete!(pitchmap, event.event_value)
                end

                if step == start_step
                    continue
                end

                position = ticks_per_step * start_step
                duration = ticks_per_step * step
                push!(notes, Note(event.event_value, vel, position, duration))
            end
        elseif event.event_type == TIME_SHIFT
            step += event.event_value
        elseif event.event_type == VELOCITY
            if event.event_value != velocity
                velocity = bin2velocity(event.event_value, perfctx.velocity_bins)
            end
        end
    end

    # End the notes which don't have a NOTE_OFF event
    for pitch in keys(pitchmap)
        for (start_step, vel) in pitchmap[pitch]
            position = ticks_per_step * start_step
            duration = ticks_per_step * (start_step + 5 * perfctx.steps_per_second) # End after 5 seconds
            push!(notes, Note(pitch, vel, position, duration))
        end
    end

    notes
end

binsize(velocity_bins) = Int(ceil(
        (MAX_MIDI_VELOCITY - MIN_MIDI_VELOCITY + 1) / velocity_bins))

bin2velocity(bin, velocity_bins) = MIN_MIDI_VELOCITY + (bin - 1) * binsize(velocity_bins)

second_to_tick(second, qpm, tpq) = second ÷ (1e-3 * ms_per_tick(qpm, tpq))
