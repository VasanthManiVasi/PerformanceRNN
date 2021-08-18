export PerfRNN, generate

using StatsBase
using Flux: gate

struct PerfRNN
    model
    encoder::PerformanceOneHotEncoding
    num_velocitybins::Int
end

"""
    generate(perfrnn::PerfRNN, performance::Performance)

Generate `PerformanceEvent`s by sampling from a PerformanceRNN model.
"""
function generate(perfrnn::PerfRNN;
                  primer::Performance=Performance(100, velocity_bins=perfrnn.num_velocitybins),
                  numsteps=3000,
                  as_notesequence=false)

    model, encoder = perfrnn.model, perfrnn.encoder
    Flux.reset!(model)
    
    performance = deepcopy(primer)

    if isempty(performance)
        push!(performance, encoder.defaultevent)
    end

    # Primer is already numsteps or longer
    if performance.numsteps >= numsteps
        return performance
    end

    indices = map(event -> encodeindex(event, encoder), performance)
    inputs = map(index -> Flux.onehot(index, encoder.labels), indices)

    outputs = model.(inputs)
    out = wsample(encoder.labels, softmax(outputs[end]))
    push!(performance, decodeindex(out, encoder))

    while performance.numsteps < numsteps
        input = Flux.onehot(out, encoder.labels)
        logits = model(input)
        out = wsample(encoder.labels, softmax(logits))
        push!(performance, decodeindex(out, encoder))
    end

    ns = getnotesequence(performance)
    as_notesequence == true && return ns
    notesequence_to_midi(ns)
end

# Replace Flux's LSTMCell with BasicLSTMCell from TensorFlow 1.
# It's implementation is slightly different.
# https://github.com/tensorflow/tensorflow/blob/v2.4.1/tensorflow/python/keras/layers/legacy_rnn/rnn_cell_impl.py#L786-L787
# https://github.com/FluxML/Flux.jl/blob/master/src/layers/recurrent.jl#L141-L142
# The replacement is required because the pre-trained models use BasicLSTMCell.
struct BasicLSTMCell{A,V,S}
  Wi::A
  Wh::A
  b::V
  state0::S
end

function BasicLSTMCell(in::Integer, out::Integer;
                  init = Flux.glorot_uniform,
                  initb = zeros,
                  init_state = zeros)
  cell = BasicLSTMCell(init(out * 4, in), init(out * 4, out), initb(out * 4), (init_state(out,1), init_state(out,1)))
  cell.b[gate(out, 2)] .= 1
  return cell
end

function (m::BasicLSTMCell)((h, c), x) where {A,V,T}
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

Flux.@functor BasicLSTMCell

Base.show(io::IO, l::BasicLSTMCell) =
  print(io, "BasicLSTMCell(", size(l.Wi, 2), ", ", size(l.Wi, 1)÷4, ")")

"""
    BasicLSTM(in::Integer, out::Integer)
Implements the BasicLSTMCell from TensorFlow 1
https://github.com/tensorflow/tensorflow/blob/v2.4.1/tensorflow/python/keras/layers/legacy_rnn/rnn_cell_impl.py#L648-L814
"""
BasicLSTM(a...; ka...) = Flux.Recur(BasicLSTMCell(a...; ka...))
Flux.Recur(m::BasicLSTMCell) = Flux.Recur(m, m.state0)
