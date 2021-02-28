

* Q1: Why it working so slow?

Metida.jl work with small and medium datasets. Model fitting is based on variance-covariance matrix inversion at each iteration. That's why if you have big blocks it will work slow. If you have big blocks you can try to use MetidaCu.jl for optimization on CUDA GPU. You can use MetidaNLopt.jl for better performance, but you will not get Hessian matrix at the end of optimization. Also if you don't need specify repeated-measure (R) covariance part and use SI, DIAG, CS, CSH covariance types for random-effect part (G) you can use MixedModels.jl - it work much faster.

* Q2: What blocks is?

Blocks depend on subjects. If you have only one random effect block is equivalent to subject. If you have more than one random effect blocks will be made as non-crossing combination for all subject variables.     
