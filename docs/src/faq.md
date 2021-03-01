

* Q1: Why it working so slow?

Metida.jl work with small and medium datasets. Model fitting is based on variance-covariance matrix inversion at each iteration. That's why if you have big blocks it will work slow. If you have big blocks you can try to use MetidaCu.jl for optimization on CUDA GPU. You can use MetidaNLopt.jl for better performance, but you will not get Hessian matrix at the end of optimization. Also if you don't need to specify repeated-measures (R) covariance part and use SI, DIAG, CS, CSH covariance types for random-effect part (G) you can use MixedModels.jl - it work much faster.

* Q2: What blocks is?

Blocks depend on subjects. If you have only one random effect block is equivalent to subject. If you have more than one random effect blocks will be made as non-crossing combination for all subject variables.     

* Q3: Why model does not converge?

Optimization of REML function can depend on many factors. In some cases covariance parameters can be correlated (ill-conditioned/singular covariance matrix). So hypersurface in the maximum area can be very flat, that why the result can be different for different starting values (or for different software even REML is near equal). Also, some models can not be fitted for specific data at all. If the model not fitted try to check how meaningful and reasonable is the model or try to guess more robust initial conditions.

* Q2: Is model fitting is correct?

Use 'lmm.log' to see warnings and errors. If you have warnings and error maybe this model is overspecified. Try to change variance structure or guess  more robust initial conditions. 
