export ckpt2bson

using Flux, BSON, Requires
using Flux: loadparams!

@init @require TensorFlow="1d978283-2c37-5f34-9a8e-e9c0ece82495" begin
  import .TensorFlow
    function readckpt(path)
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

function load_model(weights, input_dims, lstm_units)
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

function ckpt2bson(ckptpath, input_dims, lstm_units, ckptname="perfrnn.ckpt", savepath="./")
    files = readdir(ckptpath)
    ckptname ∉ files && error("The checkpoint file $ckptname is not found")
    ckptname*".meta" ∉ files && error("The checkpoint meta file is not found")
    weights = readckpt(joinpath(ckptpath, ckptname))
    model = load_model(weights, input_dims, lstm_units)
    bsonname = normpath(joinpath(savepath, ckptname[1:end-5]*".bson"))
    BSON.@save bsonname model
end
