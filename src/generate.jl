export generate, notes2midi

using StatsBase: wsample
using Flux: gate

# Overloading the Flux LSTM implementation
# with the TensorFlow 1 BasicLSTMCell implementation
# https://github.com/tensorflow/tensorflow/blob/v2.4.1/tensorflow/python/keras/layers/legacy_rnn/rnn_cell_impl.py#L786
function (m::Flux.LSTMCell)((h, c), x) where {A,V,T}
  b, o = m.b, size(h, 1)
  g = m.Wi*x .+ m.Wh*h .+ b
  input = σ.(gate(g, o, 1))
  cell = tanh.(gate(g, o, 2))
  forget = σ.(gate(g, o, 3))
  output = σ.(gate(g, o, 4))
  c = forget .* c .+ input .* cell
  h′ = output .* tanh.(c)
  sz = size(x)
  return (h′, c), reshape(h′, :, sz[2:end]...)
end

"""     generate(model, perfctx::Performance)
Generate `PerformanceEvent`s by sampling from a model.
"""
function generate(model, perfctx::Performance
                    ;primer=[PerformanceEvent(TIME_SHIFT, 100)],
                     numsteps=1000)
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

"""     notes2midi(notes::Notes)
Return a `MIDIFile` from the given `Notes`.
"""
function notes2midi(notes::Notes; qpm = 120)
    metatrack = MIDITrack()
    # TODO:
    #   Find a better way to set time signature
    qpm_bytes = reinterpret(UInt8, [UInt32(6e7 / qpm)])
    events = [
        MetaEvent(0, 0x51, reverse(qpm_bytes[1:3])), 
        MetaEvent(0, 0x58, UInt8[0x04, 0x02, 0x18, 0x08])  # 4/4 Time Signature
    ]
    addevents!(metatrack, [0, 0], events) # Add defaults
    notestrack = MIDITrack()
    addnotes!(notestrack, notes)
    midi = MIDIFile(1, notes.tpq, [metatrack, notestrack])
end
