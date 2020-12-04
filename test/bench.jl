using DataFrames, Plots, ForwardDiff, StatsBase, StatsModels, TimerOutputs

function gendata(subjn, sobsn, f, f2)
    slist = collect(1:subjn)
    sdat  = Vector{Int}(undef, 0)
    vdat = Vector{eltype(f)}(undef, 0)
    for i = 1:length(slist)
        v = Vector{Int}(undef, sobsn)
        v .= slist[i]
        append!(sdat, v)
        vf = Vector{eltype(f)}(undef, sobsn)
        vf .= rand(f)
        append!(vdat, vf)
    end
    rf = rand(f2, subjn*sobsn)
    rv = rand(subjn*sobsn)
    DataFrame(subject = sdat, fac = vdat, rfac = rf, var = rv)
end

function bench1(s, e)
    ne = e
    ns = s
    num = ne - ns + 1
    mem = Vector{Int}(undef, num)
    mem2 = Vector{Int}(undef, num)
    for i = ns:ne
        df = gendata(10, i, ["a", "b"], ["c", "d"])
        categorical!(df, :subject);
        categorical!(df, :fac);
        categorical!(df, :rfac);
        lmm = Metida.LMM(@formula(var~fac), df;
        random = Metida.VarEffect(Metida.@covstr(rfac), Metida.VC; subj = :subject),
        repeated = Metida.VarEffect(Metida.@covstr(rfac), Metida.VC; subj = :subject),
        subject = :subject)
        theta = rand(lmm.covstr.tl)
        mem[i-ns+1]  = @allocated Metida.reml_sweep_β2(lmm, theta)
        mem2[i-ns+1] = @allocated ForwardDiff.hessian(x -> Metida.reml_sweep_β2(lmm, x)[1], theta)
    end
    mem, mem2
end

function bench2(s, e)
    ne = e
    ns = s
    num = ne - ns + 1
    mem = Vector{Int}(undef, num)
    mem2 = Vector{Int}(undef, num)
    for i = ns:ne
        df = gendata(10, i, ["a", "b"], ["c", "d"])
        categorical!(df, :subject);
        categorical!(df, :fac);
        categorical!(df, :rfac);
        lmm = Metida.LMM(@formula(var~fac), df;
        random = Metida.VarEffect(Metida.@covstr(rfac), Metida.VC; subj = :subject),
        repeated = Metida.VarEffect(Metida.@covstr(rfac), Metida.VC; subj = :subject),
        subject = :subject)
        theta = rand(lmm.covstr.tl)
        mem[i-ns+1]  = @allocated Metida.reml_sweep_β(lmm, theta)
        mem2[i-ns+1] = @allocated ForwardDiff.hessian(x -> Metida.reml_sweep_β(lmm, x)[1], theta)
    end
    mem, mem2
end


n = 120
y01, y02 = bench1(3, n)
y11, y12 = bench2(3, n)
x = collect(3:n)

plot(x, y01)
plot!(x, y11)
plot(x, y02)
p = plot!(x, y12)
png("plot120-VC-VC.png", p)
maxmem = y02[end]/2^20

function bench3(s, e)
    ne = e
    ns = s
    num = ne - ns + 1
    mem = Vector{Int}(undef, num)
    mem2 = Vector{Int}(undef, num)
    for i = ns:ne
        df = gendata(10, i, ["a", "b"], ["c", "d"])
        categorical!(df, :subject);
        categorical!(df, :fac);
        categorical!(df, :rfac);
        lmm = Metida.LMM(@formula(var~fac), df;
        random = Metida.VarEffect(Metida.@covstr(rfac), Metida.VC; subj = :subject),
        repeated = Metida.VarEffect(Metida.@covstr(rfac), Metida.VC; subj = :subject),
        subject = :subject)
        theta = rand(lmm.covstr.tl)
        t1 = @elapsed mem[i-ns+1]   = @allocated Metida.fit!(lmm)
        t2 = @elapsed mem2[i-ns+1]  = @allocated Metida.fit2!(lmm)
        println("T1: ", t1, " | T2: ", t2 )
    end
    mem, mem2
end

n = 100
y01, y02 = bench3(4, n)
x = collect(4:n)
plot(x, y01)
p = plot!(x, y02)

png("plot100-VC-VC-NLopt-Optim.png")