module distributions

using LinearAlgebra
using Random

export get_datasetfn

function get_datasetfn(dataset::String)
    return dataset_dict[dataset]
end

function even_parity_probability_distribution(n::Int, seed::Int = 0)
    bitstrings = [digits(x, base = 2, pad = n) for x = 0:(2^n-1)]
    entries = [1 - (sum(bits) % 2) for bits in bitstrings]
    probability = entries ./ sum(entries)
    return Vector{Float64}(probability)
end

function cardinality_distribution(n::Int, seed::Int = 0)
    bitstrings = [digits(x, base = 2, pad = n) for x = 0:(2^n-1)]
    entries = [Int(sum(bits) == n / 2) for bits in bitstrings]
    probability = entries ./ sum(entries)
    return Vector{Float64}(probability)
end

function on2_distribution(n::Int, seed::Int = 0)
    Random.seed!(seed)
    probs = zeros(Float64, 2^n)
    # Randomly choose n unique indices
    chosen_indices = randperm(2^n)[1:n^2]
    # Assign the value 1/n to the chosen indices
    probs[chosen_indices] .= 1.0 / n^2
    return probs
end

function bas_distribution(n::Int, seed::Int = 0)
    # BAS distribution with shape 2xk=n
    # All horizontal and vertical lines are encoded with same probability
    # Only single lines exist in the dataset
    # hardcoding the vertical dimension m=2 
    # m x k = n, where m is vertical and k is horizontal directions
    m = 2
    k = Int(n / m)
    bitstrings = [digits(x, base = 2, pad = n) for x = 0:(2^n-1)]
    probability = zeros(2^n)

    # assign horizontal lines
    for line_index = 1:m
        # Initialize the grid with zeros
        grid = zeros(Int, m, k)
        # Set the values for the chosen line index (horizontal)
        grid[line_index, :] .= 1
        # Find the index of the matching vector
        index = findfirst(v -> v == vec(grid), bitstrings)
        probability[index] = 1.0
    end
    # assign vertical lines
    for line_index = 1:k
        # Initialize the grid with zeros
        grid = zeros(Int, m, k)
        # Set the values for the chosen line index (horizontal)
        grid[:, line_index] .= 1
        # Find the index of the matching vector
        index = findfirst(v -> v == vec(grid), bitstrings)
        probability[index] = 1.0
    end

    return Vector{Float64}(probability ./ sum(probability))
end

global dataset_dict = Dict(
    "parity" => even_parity_probability_distribution,
    "cardinality" => cardinality_distribution,
    "on2" => on2_distribution,
    "bas" => bas_distribution,
)


end
