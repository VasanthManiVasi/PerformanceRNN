module PerformanceRNN

using Flux, MIDI

include("constants.jl")
include("performance.jl")
include("perfrnn.jl")
include("pretrain.jl")
include("utils.jl")

function __init__()
    register_configs(configs)
end

end
