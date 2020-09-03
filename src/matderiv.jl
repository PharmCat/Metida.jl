

"""

"""
function vmat(θ::AbstractVector{T})::AbstractMatrix where T <: Real
    θ*θ'+Diagonal(θ)
end

"""

"""
#cfg = ForwardDiff.JacobianConfig(covmat, [1.0,2.0,3.0]);
function covmat_grad(f, θ, cfg)
    ForwardDiff.jacobian(f, θ, cfg);
end

function covmat_hessian(f, θ)
    ForwardDiff.jacobian(x -> ForwardDiff.jacobian(f, x), θ)
end
