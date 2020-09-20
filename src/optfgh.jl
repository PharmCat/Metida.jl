# optfgh.jl


function fgh!(F,G,H,x)

  val = func2x(x)

  if G != nothing
      G .= ForwardDiff.gradient(funcx, x)
  end
  if H != nothing
      H .= ForwardDiff.hessian(funcx, x)
  end
  if F != nothing
    return val
  end
  nothing
end
