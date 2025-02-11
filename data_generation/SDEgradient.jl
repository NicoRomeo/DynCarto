using HDF5
using HomotopyContinuation
using Random
using LinearAlgebra
using DifferentialEquations
using ProgressBars
using Base.Threads

rng = MersenneTwister(4567);


function sample_system_conservative()
    V = randn(rng, Float64, 12)
    a = zeros(9)
    b = zeros(9)
    
    a[1] = -2.0 *V[1]
    a[2] = -V[2]
    a[3] = -3.0 *V[4]
    a[4] = -2.0 *V[5]
    a[5] = - V[6]
    a[6] = - 4.0 *abs(V[8])
    a[7] = 0.
    a[8] = -2.0 *abs(V[10])
    a[9] = 0.
    
    b[1] = -V[2]
    b[2] = -2.0 *V[3]
    b[3] = -V[5]
    b[4] = -2.0 *V[6]
    b[5] = - 3.0 *V[7]
    b[6] = 0. #- 4*abs(V[8])
    b[7] = -2.0 *abs(V[10])
    b[8] = 0.
    b[9] = -4.0 *abs(V[12])
    
    
    
    b /= abs(a[1])
    tau = abs(a[1])
    a /= abs(a[1])
    a[6] = - abs(a[6])
    a[8] = - abs(a[8])
    b[6] = - a[7]
    b[7] = - abs(b[7])
    b[8] = - a[9]
    b[9] = - abs(b[9])
    X = sqrt(abs(1/a[6]))
    Y = sqrt(abs(1/b[9]))
    return a,b, X, Y, tau
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
    DX = D_reduced/X
    DY = D_reduced/Y
    function g!(du, u, p, t)
        du[1] = DX
        du[2] = DY
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


function main(n_sys::Int, num_IC::Int, D::Float64, t_max::Float64, N_times::Int, fname::String,)

    dico = Dict{Int, Vector{Vector{Float64}}}()

    fid = h5open(fname, "w")
    for n = ProgressBar(1:n_sys, printing_delay=1.0)

        a, b, X, Y, tau = sample_system_conservative()
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
            sol = DifferentialEquations.solve(prob, SKenCarp(), alg_hints=[:stiff], saveat=times, maxiters=1e9)
            trajectories[i,:,1] = X*getindex.(sol.u, 1)
            trajectories[i,:,2] = Y*getindex.(sol.u, 2)
        end
        fid["$(n)"] = trajectories
        attributes(fid["$(n)"])["params"] = hcat(a,b)
        attributes(fid["$(n)"])["scales"] = [X, Y, tau, D]

        stabilities = zeros(length(fixed_points))
        for i = 1:length(fixed_points)
            fp = fixed_points[i]
            J = zeros(2,2)
            jac!(J, fp, 0., 0.)
            eigs = eigvals(J)
            rparts = real(eigs)
            if sum(rparts .> 0) > 0
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
n_sys = 20000
num_IC = 30 # 1200 # 30 # 1200 # 30 
D = 0.01 #diffusion constant
t_max = 50. # 0.1 #50 # 0.1 
N_times = 400 # 10 # 400 # 10 
fname = "conservativeSDEtau20000_tmax50_n30_D0d01.h5"

main(n_sys, num_IC, D, t_max, N_times, fname)