# PerformanceRNN

Implementation of the [PerformanceRNN](https://arxiv.org/abs/1808.03715) in Julia. 

# Installation

```julia
] add https://github.com/VasanthManiVasi/PerformanceRNN.jl
```

# Example

```julia
using PerformanceRNN

ENV["DATADEPS_ALWAYS_ACCEPT"] = true

perfrnn = pretrain"perfrnn_dynamics"

notes = generate(perfrnn)

save("generated.mid", notes2midi(notes))
```


