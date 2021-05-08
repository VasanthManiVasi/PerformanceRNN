export generate, notes2midi

function generate(model, perfctx::Performance; primer=[PerformanceEvent(TIME_SHIFT, 100)], numsteps=1000)
    Flux.reset!(model)
    indices = map(event -> encodeindex(event, perfctx), primer)
    inputs = map(index -> Flux.onehot(index, perfctx.labels), indices)
    events = deepcopy(primer)

    outputs = model.(inputs)
    out = wsample(perfctx.labels, softmax(outputs[end]))
    push!(events, decodeindex(out, perfctx))
    for _ in 1:numsteps
        input = Flux.onehot(out, perfctx.labels)
        out = wsample(perfctx.labels, softmax(model(input)))
        push!(events, decodeindex(out, perfctx))
    end

    events
end

function notes2midi(notes::Notes)
    metatrack = MIDITrack()
    # TODO:
    #   Find a better way to do this
    events = [
        MetaEvent(0, 0x51, UInt8[0x07, 0xa1, 0x20])        # 120 QPM
        MetaEvent(0, 0x58, UInt8[0x04, 0x02, 0x18, 0x08])  # 4/4 Time Signature
    ]
    addevents!(metatrack, [0, 0], events) # Add defaults
    notestrack = MIDITrack()
    addnotes!(notestrack, notes)
    midi = MIDIFile(1, notes.tpq, [metatrack, notestrack])
end
