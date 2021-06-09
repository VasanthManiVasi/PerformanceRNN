export PerformanceEvent, Performance
export encodeindex, decodeindex, perf2notes

"""     PerformanceEvent <: Any
Event-based performance representation from Oore et al.

In the performance representation, midi data is encoded as a 388-length one hot vector.
It corresponds to NOTE_ON, NOTE_OFF events for each of the 128 midi pitches,
32 bins for the 128 midi velocities and 100 TIME_SHIFT events
(it can represent a time shift from 10 ms up to 1 second).

`PerformanceEvent` is the base representation.

## Fields
* `event_type::Int`  :  Type of the event. One of {NOTE_ON, NOTE_OFF, TIME_SHIFT, VELOCITY}.
* `event_value::Int` :  Value of the event corresponding to its type.
"""
mutable struct PerformanceEvent
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
            error("PerformanceEvent has invalid type")
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
        error("PerformanceEvent has invalid type")
    end

    s *= " $(a.event_value)"
    print(io, s)
end

"""     Performance <: Any
`Performance` is a vector of `PerformanceEvents` along with its context variables

## Fields
* `events::Vector{PerformanceEvent}` : The actual performance vector.
* `velocity_bins::Int`    : Number of bins for velocity values.
* `steps_per_second::Int` : Number of steps per second for quantization.
* `num_classes::Int`      : Total number of event classes (`NOTE_ON` events + `NOTE_OFF` events +
                            `TIME_SHIFT` events + `VELOCITY` events)
* `event_ranges::Vector{Tuple{Int, Int, Int}}` : Stores the min and max values of each event type.
"""
mutable struct Performance
    events::Vector{PerformanceEvent}
    velocity_bins::Int
    steps_per_second::Int
    num_classes::Int
    max_shift_steps::Int
    event_ranges::Vector{Tuple{Int, Int ,Int}} # Stores the range of each event type

    function Performance(;velocity_bins::Int = 32, steps_per_second::Int = 100, max_shift_steps::Int = 100)
        event_ranges = [
            (NOTE_ON, MIN_MIDI_PITCH, MAX_MIDI_PITCH)
            (NOTE_OFF, MIN_MIDI_PITCH, MAX_MIDI_PITCH)
            (TIME_SHIFT, 1, max_shift_steps)
        ]
        velocity_bins > 0 && push!(event_ranges, (VELOCITY, 1, velocity_bins))
        num_classes = sum(map(range -> range[3] - range[2] + 1, event_ranges))
        new(Vector{PerformanceEvent}(), velocity_bins, steps_per_second, num_classes, max_shift_steps, event_ranges)
    end
end

function Base.getproperty(performance::Performance, sym::Symbol)
    if sym === :labels
        return 0:(performance.num_classes - 1)
    elseif sym === :numsteps
        steps = 0
        for event in performance
            if event.event_type == TIME_SHIFT
                steps += event.event_value
            end
        end
        return steps
    else
        getfield(performance, sym)
    end
end

Base.push!(performance::Performance, event::PerformanceEvent) = push!(performance.events, event)

Base.length(performance::Performance) = length(performance.events)

Base.getindex(performance::Performance, idx) = performance.events[idx]

Base.iterate(performance::Performance, state=1) = iterate(performance.events, state)

"""     truncate(performance::Performance, numevents)
Truncate the performance to exactly `numevents` events.
"""
function Base.truncate(performance::Performance, numevents)
    performance.events = performance.events[1:numevents]
end

function append_steps(performance::Performance, num_steps)
    if (!isempty(performance) &&
        performance[end].event_type == TIME_SHIFT &&
        performance[end].event_value < performance.max_shift_steps)
        steps = min(num_steps, performance.max_shift_steps - performance[end].event_value)
        performance[end].event_value += steps
    end

    while num_steps >= performance.max_shift_steps
        push!(performance, PerformanceEvent(TIME_SHIFT, performance.max_shift_steps))
        num_steps -= performance.max_shift_steps
    end

    if num_steps > 0
        push!(performance, PerformanceEvent(TIME_SHIFT, num_steps))
    end
end

function trim_steps(performance::Performance, num_steps)
    trimmed = 0
    while !isempty(performance) && trimmed < num_steps
        if performance[end].event_type == TIME_SHIFT
            if trimmed + performance[end].event_value > num_steps
                performance[end].event_value = performance[end].event_value - num_steps + steps_trimmed
                trimmed = num_steps
            else
                trimmed += performance[end].event_value
                pop!(performance)
            end
        else
            pop!(performance)
        end
    end
end
function set_length(performance, steps)
    if performance.num_steps < steps
        append_steps(performance, steps - performance.num_steps)
    elseif self.num_steps > steps
        trim_steps(performance, performance.num_steps - steps)
    end

    @assert performance.num_steps == steps
end

"""     encodeindex(event::PerformanceEvent, performance::Performance)
Encodes a `PerformanceEvent` to its corresponding one hot index.
"""
function encodeindex(event::PerformanceEvent, performance::Performance)
    offset = 0
    for (type, min, max) in performance.event_ranges
        if event.event_type == type
            return offset + event.event_value - min
        end
        offset += (max - min + 1)
    end
end

"""     decodeindex(idx::Int, performance::Performance)
Decodes a one hot index to its corresponding `PerformanceEvent`.
"""
function decodeindex(idx::Int, performance::Performance)
    offset = 0
    for (type, min, max) in performance.event_ranges
        if idx < offset + (max - min + 1)
            return PerformanceEvent(type, min + idx - offset)
        end
        offset += (max - min + 1)
    end
end

"""     perf2notes(perf::Performance}
Converts a sequence of `PerformanceEvent`s to `MIDI.Notes`.
"""
function perf2notes(performance::Performance
                    ;ppq = DEFAULT_PPQ,
                     qpm = DEFAULT_QPM)
    ticks_per_step = second_to_tick(1, qpm, ppq) / performance.steps_per_second # sps = 100
    notes = Notes(tpq = ppq)
    pitchmap = Dict{Int, Vector{Tuple{Int, Int}}}()
    step = 0
    velocity = 64
    for event in performance
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

                # If start step and end step are the same, ignore
                if step == start_step
                    continue
                end

                position = round(ticks_per_step * start_step)
                duration = round(ticks_per_step * (step - start_step))
                push!(notes, Note(event.event_value, vel, position, duration))
            end
        elseif event.event_type == TIME_SHIFT
            step += event.event_value
        elseif event.event_type == VELOCITY
            if event.event_value != velocity
                velocity = bin2velocity(event.event_value, performance.velocity_bins)
            end
        end
    end

    # End all the notes which don't have a NOTE_OFF event
    for pitch in keys(pitchmap)
        for (start_step, vel) in pitchmap[pitch]
            position = round(ticks_per_step * start_step)
            # End after 5 seconds
            duration = round(ticks_per_step * 5 * performance.steps_per_second)
            push!(notes, Note(pitch, vel, position, duration))
        end
    end

    sort!(notes.notes, by=note -> note.position)
    notes
end

"""     binsize(velocity_bins)
Returns the size of a bin given the total number of bins.
"""
binsize(velocity_bins) = Int(ceil(
        (MAX_MIDI_VELOCITY - MIN_MIDI_VELOCITY + 1) / velocity_bins))

"""     bin2velocity(bin, velocity_bins)
Returns a velocity value given a bin and the total number of velocity bins.
"""
bin2velocity(bin, velocity_bins) = MIN_MIDI_VELOCITY + (bin - 1) * binsize(velocity_bins)

"""     second_to_tick(second, qpm, tqp)
Returns a MIDI tick corresponding to the given time in seconds,
quarter notes per minute and the amount of ticks per quarter note.
"""
second_to_tick(second, qpm, tpq) = second / (1e-3 * ms_per_tick(qpm, tpq))
