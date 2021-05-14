export ckpt_to_jld2, load_pretrain, list_pretrains, @pretrain_str

using JLD2, Requires, Pkg.TOML, DataDeps
using GoogleDrive: google_download
using Flux: loadparams!

const configs = open(TOML.parse, joinpath(@__DIR__, "pretrains.toml"))

"""     load_model(weights, input_dims, lstm_units)
Loads the weights into a Flux model
"""
function load_model(weights, config)
    lstm_units = config["lstm_units"]
    input_dims = config["input_dims"]

    layers = []
    for i in 1:config["num_layers"]
        layer = LSTM(
                     (i == 1) ? input_dims : lstm_units,
                     lstm_units
                )
        push!(layers, layer)
    end
    push!(layers, Dense(lstm_units, input_dims))

    model = Chain(layers...)

    weight_names = keys(weights)
    rnn_weights = filter(name -> occursin("rnn", name), weight_names)
    dense_weights = filter(name -> occursin("fully_connected", name), weight_names)

    for i in 1:length(rnn_weights)
        cell_weights = filter(name -> occursin("cell_$(i-1)", name), rnn_weights)
        for j in cell_weights
            if occursin("kernel", j)
                kernel = weights[j]
                if i == 1
                    Wi = @view kernel[:, 1:input_dims]
                    Wh = @view kernel[:, input_dims+1:end]
                else
                    Wi = @view kernel[:, 1:lstm_units]
                    Wh = @view kernel[:, lstm_units+1:end]
                end
                loadparams!(model[i], [Wi, Wh])
            elseif occursin("bias", j)
                loadparams!(model[i].cell.b, [weights[j]])
            end
        end
    end  

    for j in dense_weights
        if occursin("weights", j)
            loadparams!(model[end].W, [weights[j]])
        elseif occursin("biases", j)
            loadparams!(model[end].b, [weights[j]])
        end
    end
 
    model
end

"""     ckp2_to_jld2(ckptpath, input_dims, lstm_units)
Loads a pre-trained model from a tensorflow checkpoint and saves to jld2
"""
function ckpt_to_jld2(ckptpath::String; ckptname::String="perfrnn.ckpt", savepath::String="./")
    files = readdir(ckptpath)
    ckptname ∉ files && error("The checkpoint file $ckptname is not found")
    ckptname*".meta" ∉ files && error("The checkpoint meta file is not found")
    weights = readckpt(joinpath(ckptpath, ckptname))
    jld2name = normpath(joinpath(savepath, ckptname[1:end-5]*".jld2"))
    @info "Saving the model weights to $jld2name"
    JLD2.@save jld2name weights
end

# From Transformers.jl
"""     readckpt(path)
Load weights from a tensorflow checkpoint file into a Dict.
"""
readckpt(path) = error("readckpt require TensorFlow.jl installed. run `Pkg.add(\"TensorFlow\"); using TensorFlow`")

@init @require TensorFlow="1d978283-2c37-5f34-9a8e-e9c0ece82495" begin
  import .TensorFlow
  #should be changed to use c api once the patch is included
  function readckpt(path)
    weights = Dict{String, Array}()
    TensorFlow.init()
    ckpt = TensorFlow.pywrap_tensorflow.x.NewCheckpointReader(path)
    shapes = ckpt.get_variable_to_shape_map()

    for (name, shape) ∈ shapes
      weight = ckpt.get_tensor(name)
      if length(shape) == 2 && name != "cls/seq_relationship/output_weights"
        weight = collect(weight')
      end
      weights[name] = weight
    end

    weights
  end
end

"""     load_pretrain(path)
Loads a pre-trained performance rnn model.
"""
function load_pretrain(model_name::String)
    if model_name ∉ keys(configs)
        error("""Invalid model. 
               Please try list_pretrains() to check the available pre-trained models""")
    end
    
    model_path = @datadep_str("$model_name/$model_name.jld2")
    if !endswith(model_path, ".jld2")
        error("""Invalid file. A jld2 file is required to load the model.
                 If this is a tensorflow checkpoint file, run ckpt_to_jld2 to convert""")
    end

    JLD2.@load model_path weights
    model = load_model(weights, configs[model_name])
    perfctx = PerformanceContext(velocity_bins = configs[model_name]["velocity_bins"])
    perfrnn = PerfRNN(model, perfctx)
end

# From Transformers.jl
macro pretrain_str(name)
    :(load_pretrain($(esc(name))))
end

function description(description::String, host::String, link::String, cite=nothing)
  """
  $description
  Released by $(host) at $(link).
  $(isnothing(cite) ? "" : "\nCiting:\n$cite")"""
end

"""     list_pretrains()
List all the available pre-trained models.
"""
function list_pretrains()
    println.(keys(configs))
    nothing
end

function register_configs(configs)
    for (model_name, config) in pairs(configs)
        model_desc = description(config["description"], config["host"], config["link"])
        checksum = config["checksum"]
        url = config["url"]
        dep = DataDep(model_name, model_desc, url, checksum;
                      fetch_method=google_download)
        DataDeps.register(dep)
    end
end
