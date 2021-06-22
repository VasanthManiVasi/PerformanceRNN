module PerformanceRNN

using Flux, MIDI, NoteSequences

include("perfrnn.jl")
include("pretrain.jl")

@init register_configs(configs)

end
