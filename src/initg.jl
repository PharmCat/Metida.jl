#=
function initgstep(f, θ)
    g = ForwardDiff.gradient(f, θ)
    he = ForwardDiff.hessian(f, θ)
    θ .-= inv(he)*g ./2
end
=#
