
"""

"""
#cfg = ForwardDiff.JacobianConfig(covmat, [1.0,2.0,3.0]);
#function covmat_grad(f, θ, cfg)
#    ForwardDiff.jacobian(f, θ, cfg);
#end
function covmat_grad(f, Z, θ)
    m = ForwardDiff.jacobian(x -> f(gmat(x[3:5]), rmat(x[1:2], Z), Z), θ);
    reshape(m, size(Z, 1), size(Z, 1), length(θ))
end

function covmat_hessian(f, θ)
    ForwardDiff.jacobian(x -> ForwardDiff.jacobian(f, x), θ)
end
