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
