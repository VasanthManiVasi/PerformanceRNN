export ckpt_to_jld2, load_pretrain

using JLD2, Requires
using Flux: loadparams!

readckpt(path) = error("TensorFlow.jl is required to read the checkpoint. Run `Pkg.add(\"TensorFlow\"); using TensorFlow`")

@init @require TensorFlow="1d978283-2c37-5f34-9a8e-e9c0ece82495" begin
  import .TensorFlow
    function readckpt(path::String)
        weights = Dict{String, Array}()
        TensorFlow.init()
        ckpt = TensorFlow.pywrap_tensorflow.x.NewCheckpointReader(path)
        shapes = ckpt.get_variable_to_shape_map()
    
        for (name, shape) ∈ shapes
          weight = ckpt.get_tensor(name)
          if length(shape) == 2
            weight = collect(weight')
          end
          weights[name] = weight
        end
    
        weights
    end
end

"""     load_model(weights, input_dims, lstm_units)
Loads the weights into a Flux model
"""
function load_model(weights, input_dims::Int, lstm_units::Int)
    weight_names = keys(weights)
    rnn_weights = filter(name -> occursin("rnn", name), weight_names)
    dense_weights = filter(name -> occursin("fully_connected", name), weight_names)
    # TODO:
    #   Find input_dims and lstm_units automatically (from fully connected layer)
    #
    # input_dims = 388 # size(weights[dense_weights], 1)
    # lstm_units = 512 # size(rnn_)
    model = Chain(
        LSTM(input_dims, lstm_units),
        LSTM(lstm_units, lstm_units),
        LSTM(lstm_units, lstm_units),
        Dense(lstm_units, input_dims)
    )    

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
function ckpt_to_jld2(ckptpath::String, input_dims::Int, lstm_units::Int, ckptname::String="perfrnn.ckpt", savepath::String="./")
    files = readdir(ckptpath)
    ckptname ∉ files && error("The checkpoint file $ckptname is not found")
    ckptname*".meta" ∉ files && error("The checkpoint meta file is not found")
    weights = readckpt(joinpath(ckptpath, ckptname))
    model = load_model(weights, input_dims, lstm_units)
    jld2name = normpath(joinpath(savepath, ckptname[1:end-5]*".jld2"))
    @info "Saving the model to $jld2name"
    JLD2.@save jld2name model
end

"""     load_pretrain(path)
Loads a pre-trained performance rnn model from the given .jld2 file.
"""
function load_pretrain(path::String)
    if endswith(path, ".jld2")
        JLD2.@load path model
        return model
    else
        error("""Invalid file. Please pass a jld2 file.
                 If this is a tensorflow checkpoint file, please run ckpt_to_jld2""")
    end
end
