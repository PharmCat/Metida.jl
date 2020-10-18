#=
function gmat(θ::Vector{T}, zn::Int, ::CovarianceType, ::Val{:SI}) where T
    Matrix{T}(I(zn)*(θ[1] ^ 2))
    #I(zn)*(θ[1] ^ 2)
end
function gmat(θ::Vector{T}, ::Int, ::CovarianceType, ::Val{:VC}) where T
    Matrix{T}(Diagonal(θ .^ 2))
    #Diagonal(θ .^ 2)
end
function gmat(θ::Vector{T}, zn::Int, ::CovarianceType, ::Val{:AR}) where T
    mx  = Matrix{T}(undef, zn, zn)
    mx .= θ[1] ^ 2
    if zn > 1
        for m = 1:zn - 1
            for n = m + 1:zn
                @inbounds mx[m, n] = mx[m, m] * θ[2] ^ (n - m)
            end
        end
    end
    Matrix{T}(Symmetric(mx))
    #Symmetric(mx)
end
function gmat(θ::Vector{T}, zn::Int, ::CovarianceType, ::Val{:ARH}) where T
    mx  = Matrix{T}(undef, zn, zn)
    for m = 1:zn
        @inbounds mx[m, m] = θ[m]
    end
    if zn > 1
        for m = 1:zn - 1
            for n = m + 1:zn
                @inbounds mx[m, n] = mx[m, m] * mx[n, n] * θ[end] ^ (n - m)
            end
        end
    end
    for m = 1:zn
        @inbounds mx[m, m] = mx[m, m] * mx[m, m]
    end
    Matrix{T}(Symmetric(mx))
    #Symmetric(mx)
end
function gmat(θ::Vector{T}, zn::Int, ::CovarianceType, ::Val{:CSH}) where T
    mx = Matrix{T}(undef, zn, zn)
    for m = 1:zn
        @inbounds mx[m, m] = θ[m]
    end
    if zn > 1
        for m = 1:zn - 1
            for n = m + 1:zn
                @inbounds mx[m, n] = mx[m, m] * mx[n, n] * θ[end]
            end
        end
    end
    for m = 1:zn
        @inbounds mx[m, m] = mx[m, m] * mx[m, m]
    end
    Matrix{T}(Symmetric(mx))
    #Symmetric(mx)
end
=#
#=
function gmat_blockdiag(θ::Vector{T}, covstr) where T
    vm = Vector{AbstractMatrix{T}}(undef, length(covstr.ves) - 1)
    for i = 1:length(covstr.random)
        vm[i] = gmat(θ[covstr.tr[i]], covstr.q[i], covstr.random[i].covtype, Val{covstr.random[i].covtype.s}()) #covstr.random[i])
    end
    BlockDiagonal(vm)
end
=#
#=
@inline function rmat(θ::Vector{T}, rz, rn, ct, ::Val{:SI}) where T
    I(rn) * (θ[1] ^ 2)
end
@inline function rmat(θ::Vector{T}, rz, rn, ct, ::Val{:VC}) where T
    Diagonal(rz * (θ .^ 2))
end
@inline function rmat(θ::Vector{T}, rz, rn, ct, ::Val{:AR}) where T
    mx  = Matrix{T}(undef, rn, rn)
    mx .= θ[1] ^ 2
    if rn > 1
        for m = 1:rn - 1
            for n = m + 1:rn
                @inbounds mx[m, n] = mx[m, m] * θ[2] ^ (n - m)
            end
        end
    end
    Symmetric(mx)
end
@inline function rmat(θ::Vector{T}, rz, rn, ct, ::Val{:ARH}) where T
    mx   = Matrix(Diagonal(rz * (θ[1:end-1])))
    if rn > 1
        for m = 1:rn - 1
            for n = m + 1:rn
                @inbounds mx[m, n] = mx[m, m] * mx[n, n] * θ[end] ^ (n - m)
            end
        end
    end
    for m = 1:rn
        @inbounds mx[m, m] = mx[m, m] * mx[m, m]
    end
    Symmetric(mx)
end
@inline function rmat(θ::Vector{T}, rz, rn,  ct, ::Val{:CSH}) where T #???
    mx   = Matrix(Diagonal(rz * (θ[1:end-1])))
    if rn > 1
        for m = 1:rn - 1
            for n = m + 1:rn
                @inbounds mx[m, n] = mx[m, m] * mx[n, n] * θ[end]
            end
        end
    end
    for m = 1:rn
        @inbounds mx[m, m] = mx[m, m] * mx[m, m]
    end
    Symmetric(mx)
end
=#
#=
function get_z_matrix(data, covstr::CovStructure{Vector{VarEffect}})
    rschema = apply_schema(covstr.random[1].model, schema(data, covstr.random[1].coding))
    Z       = modelcols(rschema, data)
    if length(covstr.random) > 1
        for i = 1:length(covstr.random)
            rschema = apply_schema(covstr.random[i].model, schema(data, covstr.random[i].coding))
            Z       = hcat(modelcols(rschema, data))
        end
    end
    Z
end

function get_z_matrix(data, covstr::CovStructure)
    rschema = apply_schema(covstr.random.model, schema(data, covstr.random.coding))
    Z       = modelcols(rschema, data)
end

function get_term_vec(covstr::CovStructure)
    covstr.random.model
end
=#
#=
"""

"""
#cfg = ForwardDiff.JacobianConfig(covmat, [1.0,2.0,3.0]);
#function covmat_grad(f, θ, cfg)
#    ForwardDiff.jacobian(f, θ, cfg);
#end
function covmat_grad(f, Z, θ)
    #fx = x -> f(gmat(x[3:5]), rmat(x[1:2], Z), Z)
    #cfg   = ForwardDiff.JacobianConfig(fx, θ)
    m     = ForwardDiff.jacobian(x -> f(gmat(x[3:5]), rmat(x[1:2], Z), Z), θ);
    reshape(m, size(Z, 1), size(Z, 1), length(θ))
end
function covmat_hessian(f, θ)
    ForwardDiff.jacobian(x -> ForwardDiff.jacobian(f, x), θ)
end
=#
################################################################################
#=
function covmat_grad2(f, Z, θ)
    #fx = x -> f(gmat(x[3:5]), rmat(x[1:2], Z), Z)
    #cfg   = ForwardDiff.JacobianConfig(fx, θ)

    m     = ForwardDiff.jacobian(x -> f(gmat(x[3:5]), Z, x), θ);
end

function fgh!(F, G, H, x; remlβcalc::Function, remlcalc::Function)

  reml, β, c = remlβcalc(x)
  remlcalcd = x -> remlcalc(β, x)
  if G != nothing
      G .= ForwardDiff.gradient(remlcalcd, x)
  end
  if H != nothing
      H .= ForwardDiff.hessian(remlcalcd, x)
  end
  if F != nothing
    return reml
  end
  nothing
end
=#
