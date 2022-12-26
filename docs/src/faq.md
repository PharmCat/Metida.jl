

* Q1: Why it working so slow?

Metida.jl work with small and medium datasets. Model fitting is based on variance-covariance matrix inversion at each iteration. That's why if you have big blocks it will work slow. If you have big blocks you can try to use MetidaCu.jl for optimization on CUDA GPU. You can use MetidaNLopt.jl for better performance, but you will not get Hessian matrix at the end of optimization. Also if you don't need to specify repeated-measures (R) covariance part and use SI, DIAG, CS, CSH covariance types for random-effect part (G) you can use MixedModels.jl - it work much faster.

* Q2: What blocks is?

Blocks depend on subjects. If you have only one random effect block is equivalent to subject. If you have more than one random effect blocks will be made as non-crossing combination for all subject variables.     

* Q3: Why model does not converge?

Optimization of REML function can depend on many factors. In some cases covariance parameters can be correlated (ill-conditioned/singular covariance matrix). So hypersurface in the maximum area can be very flat, that why the result can be different for different starting values (or for different software even REML is near equal). Also, some models can not be fitted for specific data at all. If the model not fitted try to check how meaningful and reasonable is the model or try to guess more robust initial conditions.

* Q4: Is model fitting is correct?

Use 'lmm.log' to see warnings and errors. If you have warnings and error maybe this model is overspecified. Try to change variance structure or guess  more robust initial conditions.

* Q5: How to choose best variance-covariance structure?

SAS Manual, Mixed Models Analyses Using the SAS System Course Notes:

> Unfortunately, our attempt to share a very RECENT perspective by a relatively small number of statistics and statistics related research has somewhat sidetracked the focus of lesson 1. Would like to attempt to provide some clarity to some of the discussion on the discussion forum about the bar chart vs. interval charts.
>You can use information criteria produced by the MIXED procedure as a tool to help you select the model with the most appropriate covariance structure. The smaller the information criteria value is, the better the model is. Theoretically, the smaller the -2 Res Log Likelihood is, the better the model is. However, you can always make this value smaller by adding parameters to the model. Information criteria attached penalties to the negative -2 Res Log Likelihood value; that is, the more the parameters, the bigger the penalties.
>Two commonly used information criteria are Akaike's (1974) and Schwartz's (1978). Generally speaking, BIC tends to choose less complex models than AIC. Because choosing a model that is too simple inflates Type I error rate, when Type I error control is the highest priority, you may want to use AIC. On the other hand, if loss of power is more of a concern, BIC might be preferable (Guerin and Stroup 2000).
>Starting in the Release 8.1, the MIXED procedure produces another information criteria, AICC. AICC is a finite-sample corrected Akaike Information Criterion. For small samples, it reduces the bias produced by AIC; for large samples, AICC converges to AIC. In general, AICC is preferred to AIC. For more information on information criteria, especially AICC, refer to Burnham, K. P. and Anderson, D. R. (1998).
>The basic idea for repeated measures analysis is that, among plausible within-subject covariance models given a particular study, the model that minimizes AICC or BIC (your choice) is preferable. When AICC or BIC are close, the simpler model is generally preferred.

* Q6: I have a slightly different results for DF calculation, what's wrong?

Check logs. If final hessian matrix for REML function is ill-conditioned results on differet OS can be slightly different. If possible, try to use more stable covariance structure. Or make an issue on github.

* Q7: I changed `maxthreads` settings but nothing happend. 

Check the number of execution threads: `Threads.nthreads()`, maybe you should change julia startup settings. See also [julia threads docs](https://docs.julialang.org/en/v1/manual/multi-threading/).


See also:

* Barnett, A.G., Koper, N., Dobson, A.J., Schmiegelow, F. and Manseau, M. (2010), Using information criteria to select the correct variance–covariance structure for longitudinal data in ecology. Methods in Ecology and Evolution, 1: 15-24. https://doi.org/10.1111/j.2041-210X.2009.00009.x

* [Guidelines for Selecting the Covariance Structure in Mixed Model Analysis](https://support.sas.com/resources/papers/proceedings/proceedings/sugi30/198-30.pdf)

* H. J. Keselman,James Algina,Rhonda K. Kowalchuk &Russell D. Wolfinger. A comparison of two approaches for selecting covariance structures in the analysis of repeated measurements. https://doi.org/10.1080/03610919808813497
