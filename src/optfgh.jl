# optfgh.jl


function fgh!(F, G, H, x; remlβcalc::Function, remlcalc::Function)


  reml, β, c = remlβcalc(x)
  #remlcalc2  = x -> remlβcalc(x)[1]
  remlcalcd = x -> remlcalc(β, x)

  if G != nothing
      G .= ForwardDiff.gradient(remlcalcd, x)
      #G .= ForwardDiff.gradient(remlcalc2, x)
  end
  if H != nothing
      H .= ForwardDiff.hessian(remlcalcd, x)
      #H .= ForwardDiff.hessian(remlcalc2, x)
  end
  if F != nothing
    return reml
  end
  nothing
end
