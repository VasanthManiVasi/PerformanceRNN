# PerformanceRNN

Implementation of the [PerformanceRNN](https://arxiv.org/abs/1808.03715) in Julia. 

# Installation

```julia
] add https://github.com/VasanthManiVasi/PerformanceRNN.jl
```

# Example

```julia
using FileIO
using PerformanceRNN
ENV["DATADEPS_ALWAYS_ACCEPT"] = true

perfrnn = pretrained"perfrnn_dynamics"

midi = generate(perfrnn)

save("generated.mid", midi)
```


