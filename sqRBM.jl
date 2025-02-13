module qbm

using LinearAlgebra
using Printf
using Random
using CSV, DataFrames

include("distributions.jl")
using .distributions:get_datasetfn

include("information.jl")
using .information:kl_divergence, tvd

include("utils.jl")
using .utils: create_folder

export get_hamiltonian_operators
export train
export get_pv
export get_grad


global TOL=1e-16

function get_connectivity(num_visible::Int64, num_hidden::Int64)
    #define conectivity of the model
    indices=[]
    for v=1:num_visible
        for h=num_visible+1:num_visible+num_hidden
            push!(indices, (v,h))
        end
    end
    return indices
end

function get_hamiltonian_operators(num_visible::Int64, num_hidden::Int64, model::String)
    #get list of obs
    obs=[]
    indices = get_connectivity(num_visible, num_hidden)
    if model=="rbm"
        v_field_op=['Z']
        h_field_op=['Z']
        connection_op = ["ZZ"]
    elseif model=="sqrbm-XZ"
        v_field_op=['Z']
        h_field_op=['Z', 'X']
        connection_op = ["ZZ", "ZX"]
    elseif model=="sqrbm-XYZ"
        v_field_op=['Z']
        h_field_op=['Z', 'X', 'Y']
        connection_op = ["ZZ", "ZX", "ZY"]
    end
        
    for pair in indices
        for op in connection_op
            s = "I"^(num_visible+num_hidden)
            s = s[1:pair[1]-1]*op[1]*s[pair[1]+1:end]
            s = s[1:pair[2]-1]*op[2]*s[pair[2]+1:end]
            push!(obs, s)
        end
    end

    for nv=1:num_visible
        for op in v_field_op
            s = "I"^(num_visible+num_hidden)
            s = s[1:nv-1]*op*s[nv+1:end]
            push!(obs, s)
        end
    end

    for nh=num_visible+1:(num_visible+num_hidden)
        for op in h_field_op
            s = "I"^(num_visible+num_hidden)
            s = s[1:nh-1]*op*s[nh+1:end]
            push!(obs, s)
        end
    end

    return obs
end
   

function get_pv(
    model::String,
    H::Dict,
    num_visible::Int64,
    num_hidden::Int64,
    )

    if model=="rbm"
        ɸ = zeros(Float64, 1, num_hidden, 2^num_visible)
    elseif model=="sqrbm-XZ"
        ɸ = zeros(Float64, 2, num_hidden, 2^num_visible)
    elseif model=="sqrbm-XYZ"
        ɸ = zeros(Float64, 3, num_hidden, 2^num_visible)
    end
    
    paulis = ["Z", "X", "Y"]
    bitstrings = [reverse(digits(x, base=2, pad=num_visible)) for x in 0:(2^num_visible - 1)]
    
    for h=1:num_hidden # loop over hidden units
        for i=1:size(ɸ)[1] # loop over X,Y,Z types
            for j=1:2^num_visible # loop over all visible configurations
                bitstring = bitstrings[j]

                # hidden field 
                s = "I"^(num_visible+num_hidden)
                s = s[1:h+num_visible-1]*paulis[i]*s[h+num_visible+1:end]
                term = H[s]

                # interaction terms
                for v=1:num_visible
                    s = "I"^(num_visible+num_hidden)
                    s = s[1:v-1]*"Z"*s[v+1:end]
                    s = s[1:h+num_visible-1]*paulis[i]*s[h+num_visible+1:end]
                    term += (-1)^(bitstring[v]) * H[s]
                end
                ɸ[i,h,j] = term
            end
        end
    end

    # compute unnormalized probabilities
    p_tilde = ones(Float64, 2^num_visible)
    for j=1:2^num_visible # loop over all visible configurations
        bitstring = bitstrings[j]
        for v=1:num_visible
            s = "I"^(num_visible+num_hidden)
            s = s[1:v-1]*paulis[1]*s[v+1:end]
            term = exp(-(-1)^(bitstring[v])*H[s])
            p_tilde[j] *= term
        end
        for h=1:num_hidden
            p_tilde[j] *= 2*cosh(norm(ɸ[:,h,j],2))
        end

    end

    # normalize to obtain probabilities
    Z = sum(p_tilde)
    p = p_tilde/Z

    return p, p_tilde, Z, ɸ

end

function get_grad(
    model::String,
    H::Dict,
    num_visible::Int64,
    num_hidden::Int64,
    pv::Vector,
    p_data::Vector,
    p_tilde::Vector,
    Z::Float64,
    ɸ::Array
    )

    # Copy parameter dictionary
    grads = copy(H)

    # Set all values to zero
    for k in keys(grads)
        grads[k] = 0
    end
    
    paulis = ["Z", "X", "Y"]
    bitstrings = [reverse(digits(x, base=2, pad=num_visible)) for x in 0:(2^num_visible - 1)]
    v_vec = zeros(num_visible, 2^num_visible) # contains (-1)^v_i
    for v=1:num_visible
        for j=1:2^num_visible
            v_vec[v,j] = (-1)^(bitstrings[j][v])
        end
    end

    # compute gradients of field terms on visible units
    for v=1:num_visible
        s = "I"^(num_visible+num_hidden)
        s = s[1:v-1]*"Z"*s[v+1:end]

        dellogz = sum(v_vec[v,:].*pv)
        grads[s] = sum(p_data.*(v_vec[v,:] .- dellogz))
    end

    # compute gradients of field terms on hidden units
    delphi = zeros(size(ɸ)[1], num_hidden, 2^num_visible)
    for h=1:num_hidden
        for i=1:size(ɸ)[1] # loop over X,Y,Z types
            for j=1:2^num_visible # loop over all visible configurations
                ɸ_norm = norm(ɸ[:,h,j],2)
                delphi[i,h,j] = ɸ[i,h,j]*tanh(ɸ_norm)/ɸ_norm
            end
        end
    end

    for h=1:num_hidden
        for i=1:size(ɸ)[1] # loop over X,Y,Z types
            s = "I"^(num_visible+num_hidden)
            s = s[1:h+num_visible-1]*paulis[i]*s[h+num_visible+1:end]
            dellogz = sum(delphi[i,h,:].*pv)
            grads[s] = -sum(p_data.*(delphi[i,h,:] .- dellogz))
        end
    end

    # compute gradients of interaction terms
    for v=1:num_visible
        for h=1:num_hidden
            for i=1:size(ɸ)[1] # loop over X,Y,Z types
                s = "I"^(num_visible+num_hidden)
                s = s[1:v-1]*"Z"*s[v+1:end]
                s = s[1:h+num_visible-1]*paulis[i]*s[h+num_visible+1:end]
                dellogz = sum(v_vec[v,:].*delphi[i,h,:].*pv)
                grads[s] = -sum(p_data.*(v_vec[v,:].*delphi[i,h,:] .- dellogz))
            end
        end    
    end
    
    return grads
end


function train(
    model::String,
    num_visible::Int64,
    num_hidden::Int64,
    runID::Int64,
    logPath::String,
    dataset::String,
    num_iter::Int64=2000,
    )

    # set the seed
    Random.seed!(runID)

    num_qubits = num_visible+num_hidden
    
    global p_data = get_datasetfn(dataset)(num_visible, runID)
  
    path = logPath * @sprintf("%s/%s/%dv/%dh/", dataset, model, num_visible, num_hidden)   
    loss_path = path * @sprintf("%s_%dv_%dh_%d_loss.csv", model, num_visible, num_hidden, runID)
    grads_path = path * @sprintf("%s_%dv_%dh_%d_grads.csv", model, num_visible, num_hidden, runID)
    coef_path = path * @sprintf("%s_%dv_%dh_%d_coef.csv", model, num_visible, num_hidden, runID)
    pv_path = path * @sprintf("%s_%dv_%dh_%d_pv.csv", model, num_visible, num_hidden, runID)

    create_folder(path)

    # get model Hamiltonian
    op = get_hamiltonian_operators(num_visible, num_hidden, model)
    num_terms = size(op)[1]

    # instantiate variables for logging
    global kl_hist = []
    global tvd_hist = []
    global time_hist = []
    grads_hist = Dict(op .=> [Vector{Float64}() for _ in 1:num_terms])
    coef_hist = Dict(op .=> [Vector{Float64}() for _ in 1:num_terms])

    # initialize coefficients [-1,1]
    coefficients = (rand(Float64, num_terms).-0.5).*2

    # construct Hamiltonian dictionary 
    H = Dict(op .=> coefficients)

    # log initial coefficients
    for op in keys(coef_hist)
        push!(coef_hist[op], H[op])
    end

    # instantiate moments for optimizer
    moment1 = Dict(op .=> zeros(num_terms))
    moment2 = Dict(op .=> zeros(num_terms))
    moment2_hat = Dict(op .=> zeros(num_terms))

    # Optimizer hyper-parameters
    lr = 0.1
    m = 0.9
    n = 0.999

    for i=1:num_iter
        local step_time = @elapsed begin # time iteration step

        # compute pv and perform preprocessing for gradients
        pv, p_tilde, Z, ɸ = get_pv(model, H, num_visible, num_hidden)
        # compute gradients
        grads = get_grad(model, H, num_visible, num_hidden, pv, p_data, p_tilde, Z, ɸ)

        # AMSgrad optimzation step
        lr_hat = lr * sqrt(1 - n^(i + 1))/(1 - m^(i + 1))
        for (op, coef) in H
            moment1[op] = m*moment1[op] + ((1-m)*grads[op])
            moment2[op] = n*moment2[op] + ((1-n)*(grads[op]^2))
            moment2_hat[op] = max(moment2_hat[op], moment2[op])
            update = lr_hat * (moment1[op]/(sqrt(moment2_hat[op]) + TOL))
            H[op] = coef - update
        end

        end # end of timer

        push!(time_hist, step_time)

        # log the gradients
        for op in keys(grads)
            push!(grads_hist[op], grads[op])
        end

        # log updated parameters
        for op in keys(H)
            push!(coef_hist[op], H[op])
        end

        # Compute loss after iteration step
        kl_div = kl_divergence(p_data, pv)
        push!(kl_hist, kl_div)

        tvd_val = tvd(p_data, pv)
        push!(tvd_hist, tvd_val)
        
    end


    # Compute loss one more time with the last parameters at the end of training
    pv, _, _, _ = get_pv(model, H, num_visible, num_hidden)
    kl_div = kl_divergence(p_data, pv)
    tvd_val = tvd(p_data, pv)
    push!(kl_hist, kl_div)
    push!(tvd_hist, tvd_val)
    push!(time_hist, NaN) # last kl computation does not use grad function, so time is NaN

    # save the logs to files after every step
    CSV.write(loss_path, DataFrame(kl=kl_hist, tvd=tvd_hist, elapsed=time_hist))
    CSV.write(grads_path, DataFrame(grads_hist))
    CSV.write(coef_path, DataFrame(coef_hist))
    CSV.write(pv_path, DataFrame(pv=pv))

end


end