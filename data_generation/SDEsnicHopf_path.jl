using HDF5
using HomotopyContinuation
using Random
using LinearAlgebra
using DifferentialEquations
using ProgressBars
using Base.Threads

rng = MersenneTwister(1234);


function SNICHopfParameters(k::Float64, alpha_p::Float64, alpha_m::Float64)
    a = zeros(9)
    b = zeros(9)
    a[6] = -1.0
    a[8] = -1.0
    b[7] = -1.0
    b[9] = -1.0
    a[1] = k
    a[2] = alpha_p - alpha_m
    b[1] = alpha_p + alpha_m
    b[2] = k
    return a,b, 1.0, 1.0, 1.0
end

function make_sys(ap,bp, X, Y)
    #dy = zeros(2, Float32)
    a = ap .* [1, Y/X, X, Y, Y*Y/X, X^2, X*Y, Y*Y, Y^3/X]
    b = bp .* [X/Y, 1, X*X/Y, X, Y, X^3/Y, X*X, X*Y, Y^2]
    function func!(du, u, p, t)
        du[1] = a[1]*u[1] + a[2]*u[2] + a[3]*u[1]^2 + a[4]*u[1]*u[2] + a[5]*u[2]^2 + 
            a[6]*u[1]^3 + a[7]* u[1]^2 * u[2] + a[8]*u[1]*u[2]^2 + a[9]*u[2]^3
        du[2] = b[1]*u[1] + b[2]*u[2] + b[3]*u[1]^2 + b[4]*u[1]*u[2] + b[5]*u[2]^2 + 
            b[6]*u[1]^3 + b[7]* u[1]^2 * u[2] + b[8]*u[1]*u[2]^2 + b[9]*u[2]^3
    end
    function jac!(J, u, p, t)
        # dfx/dx
        J[1,1] = a[1]+2*a[3]*u[1]+a[4]*u[2]+3*a[6]*u[1]^2+2*a[7]*u[1]*u[2]+a[8]*u[2]^2
        #dfx/dy
        J[1,2] =  a[2]+a[4]*u[1]+2*a[5]*u[2]+a[7]*u[1]^2+2*a[8]*u[1]*u[2]+3*a[9]*u[2]^2
        #dfy/dx
        J[2,1] = b[1]+2*b[3]*u[1]+b[4]*u[2]+3*b[6]*u[1]^2+2*b[7]*u[1]*u[2]+b[8]*u[2]^2
        #dfy/dy
        J[2,2] = b[2]+b[4]*u[1]+2*b[5]*u[2]+b[7]*u[1]^2+2*b[8]*u[1]*u[2]+3*b[9]*u[2]^2
        nothing
    end
    return func!, jac!
end


function make_sys_SDE(ap,bp, D, X, Y, tau)
    #dy = zeros(2, Float32)
    a = ap .* [1, Y/X, X, Y, Y*Y/X, X^2, X*Y, Y*Y, Y^3/X]
    b = bp .* [X/Y, 1, X*X/Y, X, Y, X^3/Y, X*X, X*Y, Y^2]
    function func!(du, u, p, t)
        du[1] = a[1]*u[1] + a[2]*u[2] + a[3]*u[1]^2 + a[4]*u[1]*u[2] + a[5]*u[2]^2 + 
            a[6]*u[1]^3 + a[7]* u[1]^2 * u[2] + a[8]*u[1]*u[2]^2 + a[9]*u[2]^3
        du[2] = b[1]*u[1] + b[2]*u[2] + b[3]*u[1]^2 + b[4]*u[1]*u[2] + b[5]*u[2]^2 + 
            b[6]*u[1]^3 + b[7]* u[1]^2 * u[2] + b[8]*u[1]*u[2]^2 + b[9]*u[2]^3
    end
    D_reduced = sqrt(2*D/tau)
    function g!(du, u, p, t)
        du[1] = D_reduced
        du[2] = D_reduced
    end

    function jac!(J, u, p, t)
        # dfx/dx
        J[1,1] = a[1]+2*a[3]*u[1]+a[4]*u[2]+3*a[6]*u[1]^2+2*a[7]*u[1]*u[2]+a[8]*u[2]^2
        #dfx/dy
        J[1,2] =  a[2]+a[4]*u[1]+2*a[5]*u[2]+a[7]*u[1]^2+2*a[8]*u[1]*u[2]+3*a[9]*u[2]^2
        #dfy/dx
        J[2,1] = b[1]+2*b[3]*u[1]+b[4]*u[2]+3*b[6]*u[1]^2+2*b[7]*u[1]*u[2]+b[8]*u[2]^2
        #dfy/dy
        J[2,2] = b[2]+b[4]*u[1]+2*b[5]*u[2]+b[7]*u[1]^2+2*b[8]*u[1]*u[2]+3*b[9]*u[2]^2
        nothing
    end
    return func!, g!, jac!
end


function homotopy_sys(a,b)
        @var x y
        f₁ = a[1] * x + a[2] * y + a[3] * x^2 + a[4]* x * y + a[5]* y^2 + 
            a[6] * x^3 + a[7]* x^2 * y + a[8]* x * y^2 + a[9]* y^3
        f₂ = b[1]*x + b[2]*y + b[3]*x^2 + b[4]*x*y + b[5]*y^2 + 
            b[6]*x^3 + b[7]* x^2 * y + b[8]*x*y^2 + b[9]*y^3
        F = System([f₁, f₂])
    return F
end


function main(n_sys::Int, num_IC::Int, D::Float64, t_max::Float64, N_times::Int, fname::String, Ks, Ams, alpha_p::Float64)

    dico = Dict{Int, Vector{Vector{Float64}}}()

    alpha_p = 1.0
    fid = h5open(fname, "w")
    for n = ProgressBar(1:length(Ams), printing_delay=1.0)

        a, b, X, Y, tau = SNICHopfParameters(Ks[n], alpha_p, Ams[n])
        fun!, g!, jac! = make_sys_SDE(a,b, D, X, Y, tau)
        times = LinRange(0,t_max, N_times)
        f! = ODEFunction(fun!, jac = jac!)
        trajectories = Array{Float64, 3}(undef, num_IC, length(times), 2)
        F = homotopy_sys(a,b)
        EOptions = HomotopyContinuation.EndgameOptions(max_endgame_steps = 2000)
        TOptions = HomotopyContinuation.TrackerOptions(automatic_differentiation = 3, max_steps = 10_000)
        fixed_points = real_solutions(HomotopyContinuation.solve(F; show_progress= false, start_system = :polyhedral, endgame_options=EOptions, tracker_options=TOptions))
        for i = 1:num_IC
            u0 = zeros(Float64, 2)
            prob = SDEProblem(fun!, g!, zeros(Float64, 2), (0.0, times[end]))
            sol = DifferentialEquations.solve(prob, SKenCarp(), alg_hints=[:stiff], saveat=times, maxiters=1e7)
            trajectories[i,:,1] = X*getindex.(sol.u, 1)
            trajectories[i,:,2] = Y*getindex.(sol.u, 2)
        end
        fid["$(n)"] = trajectories
        attributes(fid["$(n)"])["params"] = hcat(a,b)
        attributes(fid["$(n)"])["SNICHopf_param"] = [Ks[n], alpha_p, Ams[n]]

        # Fixed point processing: compute stability
        stabilities = zeros(length(fixed_points))
        for i = 1:length(fixed_points)
            fp = fixed_points[i]
            J = zeros(2,2)
            jac!(J, fp, 0., 0.)
            eigs = eigvals(J)
            rparts = real(eigs)
            if any(rparts .>= 0)
                stabilities[i] = 1
            else
                stabilities[i] = -1
            end
        end
        # error handling in case of HC fail
        if length(fixed_points)>0
            attributes(fid["$(n)"])["fps"] = reduce(hcat,fixed_points)
            attributes(fid["$(n)"])["stabilities"] = stabilities
        else
            attributes(fid["$(n)"])["fps"] = 0
            attributes(fid["$(n)"])["stabilities"] = 0
        end
    
    end
    close(fid)
    println("~success~")
end

## with fixed times SDE
n_sys = 32
Kmin = -2.0
Kmax = 2.0
Am_min = -2.0
Am_max = 2.0

A0 = 1.5
theta = LinRange(0, 2*pi, 500)

Ams = [A0*sin(t) for t = theta]
Ks = [A0*cos(t) for t = theta]

Ks[Ams .< 0.] .= LinRange(-A0, A0, 250)
Ams[Ams .< 0.] .= 0.





alpha_p = 1.0
num_IC = 30 # 1200 # 30 # 1200 # 30 
D = 0.005 #diffusion constant
t_max = 50. # 0.1 #50 # 0.1 
N_times = 400 # 10 # 400 # 10 
fname = "SDEpathSnicHopf2_hemi_tmax50_n30_D0d005.h5"

for i = 1:20
    fname = "./snichopf_data/SDEpathSnicHopf2_hemi_tmax50_n30_D0d005_$(i).h5"
    main(n_sys, num_IC, D, t_max, N_times, fname, Ks, Ams, alpha_p)
end

