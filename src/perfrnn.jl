export PerformanceRNN, generate

using StatsBase: wsample
using Flux: gate

struct PerfRNN
    model
    perfctx::PerformanceContext
end

"""     generate(model, perfctx::Performance)
Generate `PerformanceEvent`s by sampling from a model.
"""
function generate(perfrnn::PerfRNN;
        primer::Performance=[PerformanceEvent(TIME_SHIFT, 100)],
        numsteps=1000,
        raw = false)

    model, perfctx = perfrnn.model, perfrnn.perfctx # For readability
    Flux.reset!(model)
    
    # Primer is already numsteps or longer
    if len(primer) >= numsteps
        return primer
    end

    performance = deepcopy(primer)

    indices = map(event -> encodeindex(event, perfctx), performance)
    inputs = map(index -> Flux.onehot(index, perfctx.labels), indices)

    outputs = model.(inputs)
    out = wsample(perfctx.labels, softmax(outputs[end]))
    push!(performance, decodeindex(out, perfctx))

    while len(performance) < numsteps
        input = Flux.onehot(out, perfctx.labels)
        out = wsample(perfctx.labels, softmax(model(input)))
        push!(performance, decodeindex(out, perfctx))
    end

    raw == true && return performance
    notes = perf2notes(performance, perfctx)
    sort!(notes.notes, by=note -> note.position)
    notes
end

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
