module PerformanceRNN

using Flux, MIDI

include("constants.jl")
include("performance.jl")
include("pretrain.jl")
include("generate.jl")

function __init__()
    register_config(configs)
end

end
