
using LinearAlgebra
using Printf
using Random
using CSV, DataFrames

include("distributions.jl")
using .distributions: get_datasetfn

include("sqRBM.jl")
using .qbm: get_grad, get_pv, get_hamiltonian_operators

include("information.jl")
using .information: kl_divergence

include("utils.jl")
using .utils: create_folder


global TOL = 1e-16

function batch_compute_grads_kl(
    model::String,
    num_visible::Int64,
    num_hidden::Int64,
    num_runs::Int64,
    logPath::String,
    dataset::String,
)

    num_qubits = num_visible + num_hidden

    # get target probability distribution
    global p_data = get_datasetfn(dataset)(num_visible)

    op = get_hamiltonian_operators(num_visible, num_hidden, model)
    num_terms = size(op)[1]

    kl_hist = []
    grads_hist = Dict(op .=> [Vector{Float64}() for _ = 1:num_terms])

    for runID = 1:num_runs

        Random.seed!(runID)
        # initialize coefficients [-10,10]
        coefficients = (rand(Float64, num_terms) .- 0.5) .* (20)

        # construct Hamiltonian dictionary 
        H = Dict(op .=> coefficients)

        num_qubits = num_visible + num_hidden

        # compute pv and perform preprocessing for gradients
        pv, p_tilde, Z, ɸ = get_pv(model, H, num_visible, num_hidden)
        # compute gradients
        grads = get_grad(model, H, num_visible, num_hidden, pv, p_data, p_tilde, Z, ɸ)


        # Compute loss after iteration step
        kl_div = kl_divergence(p_data, pv)

        #log gradients and kl
        for op in keys(grads)
            push!(grads_hist[op], grads[op])
        end

        push!(kl_hist, kl_div)
    end

    path = logPath * @sprintf("%s/", dataset)
    create_folder(path)
    file_path = path * @sprintf("%s_%dv_%dh", model, num_visible, num_hidden)

    CSV.write(file_path * "_kl.csv", DataFrame(kl = kl_hist))
    CSV.write(file_path * "_grads.csv", DataFrame(grads_hist))

end


dataset = "parity" #  "bas", "parity", "cardinality", "on2"
model = "rbm" # "rbm", "sqrbm-XZ", "sqtbm-XYZ"
num_visible = 4
num_hidden = 2
num_runs = 10000
num_iter = 2000


logPath = "./logs/gradients/"


batch_compute_grads_kl(model, num_visible, num_hidden, num_runs, logPath, dataset)
