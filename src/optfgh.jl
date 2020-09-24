# optfgh.jl


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
