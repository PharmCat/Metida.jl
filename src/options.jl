#options
#=
mutable struct Options
    method
    ulim
    llim
    vlink
    x_tol
    f_tol
    g_tol
    x_abstol::Real
    x_reltol::Real
    f_abstol::Real
    f_reltol::Real
    g_abstol::Real
    g_reltol::Real
    outer_x_tol
    outer_f_tol
    outer_g_tol
    outer_x_abstol::Real
    outer_x_reltol::Real
    outer_f_abstol::Real
    outer_f_reltol::Real
    outer_g_abstol::Real
    outer_g_reltol::Real
    f_calls_limit::Int
    g_calls_limit::Int
    h_calls_limit::Int
    allow_f_increases::Bool
    allow_outer_f_increases::Bool
    successive_f_tol::Int
    iterations::Int
    outer_iterations::Int
    store_trace::Bool
    trace_simplex::Bool
    show_trace::Bool
    extended_trace::Bool
    show_every::Int
    callback
    time_limit
end

function Options(;
method = Optim.Newton(),
ulim = nothing,
llim = nothing,
vlink = nothing,
x_tol = nothing,
f_tol = nothing,
g_tol = nothing,
x_abstol::Real = 0.0,
x_reltol::Real = 0.0,
f_abstol::Real = 0.0,
f_reltol::Real = 0.0,
g_abstol::Real = 1e-8,
g_reltol::Real = 1e-8,
outer_x_tol = 0.0,
outer_f_tol = 0.0,
outer_g_tol = nothing,
outer_x_abstol::Real = 0.0,
outer_x_reltol::Real = 0.0,
outer_f_abstol::Real = 0.0,
outer_f_reltol::Real = 0.0,
outer_g_abstol::Real = 1e-8,
outer_g_reltol::Real = 1e-8,
f_calls_limit::Int = 0,
g_calls_limit::Int = 0,
h_calls_limit::Int = 0,
allow_f_increases::Bool = true,
allow_outer_f_increases::Bool = true,
successive_f_tol::Int = 1,
iterations::Int = 1_000,
outer_iterations::Int = 1000,
store_trace::Bool = false,
trace_simplex::Bool = false,
show_trace::Bool = false,
extended_trace::Bool = false,
show_every::Int = 1,
callback = nothing,
time_limit = NaN)
    Options(
        method,
        ulim,
        llim,
        vlink,
        x_tol,
        f_tol,
        g_tol,
        x_abstol,
        x_reltol,
        f_abstol,
        f_reltol,
        g_abstol,
        g_reltol,
        outer_x_tol,
        outer_f_tol,
        outer_g_tol,
        outer_x_abstol,
        outer_x_reltol,
        outer_f_abstol,
        outer_f_reltol,
        outer_g_abstol,
        outer_g_reltol,
        f_calls_limit,
        g_calls_limit,
        h_calls_limit,
        allow_f_increases,
        allow_outer_f_increases,
        successive_f_tol,
        iterations,
        outer_iterations,
        store_trace,
        trace_simplex,
        show_trace,
        extended_trace,
        show_every,
        callback,
        time_limit)
end
=#
