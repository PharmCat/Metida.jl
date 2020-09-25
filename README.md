# Metida

Experimental package for variance-component calculation.

*Alfa version*

Install:

```
import Pkg; Pkg.add("Metida")
```

Using:

```
lmm = LMM(@formula(var~sequence+period+formulation), df;
random = VarEffect(@covstr(formulation), CSH),
repeated = VarEffect(@covstr(formulation), VC),
subject = :subject)

Metida.fit!(lmm)
```

Version 0.1.1

© 2020 Metida
