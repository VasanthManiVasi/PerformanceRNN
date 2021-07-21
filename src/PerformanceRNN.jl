module PerformanceRNN

using Flux, MIDI, NoteSequences
using NoteSequences.PerformanceRepr

include("perfrnn.jl")
include("pretrain.jl")

@init register_configs(configs)

end
