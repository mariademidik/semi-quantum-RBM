module information

using LinearAlgebra

export kl_divergence
export tvd

global TOL=1e-16

function kl_divergence(p_data::Vector, p_model::Vector)
    return sum(p_data.*(log.(p_data .+ TOL) .- log.(p_model .+ TOL)))
end

function tvd(p_data::Vector, p_model::Vector)
    return sum(abs.(p_data .- p_model))/2
end


end