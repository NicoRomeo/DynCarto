using HDF5
using HomotopyContinuation
using Random
using LinearAlgebra
using DifferentialEquations
using ProgressBars
using Base.Threads

rng = MersenneTwister(12345);


function sample_system()
    a = zeros(9)
    b = randn(rng, Float64, 9)
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

function create_fhn_fastslow(nu, epsilon)
    a = zeros(9)
    b = zeros(9)
    a[1] =  -1.0
    a[2] =  -1.0
    a[3] = nu
    b[1] = epsilon
    b[2] = - epsilon
    a[6] = - 1.0

    X = 1.0
    Y = 1.0
    tau = 1.0
    return a, b, X, Y, tau
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


function make_sys_SDE_fs(ap,bp, D, X, Y, tau, epsilon)
    #dy = zeros(2, Float32)
    a = ap .* [1, Y/X, X, Y, Y*Y/X, X^2, X*Y, Y*Y, Y^3/X]
    b = bp .* [X/Y, 1, X*X/Y, X, Y, X^3/Y, X*X, X*Y, Y^2]
    function func!(du, u, p, t)
        du[1] = a[1]*u[1] + a[2]*u[2] + a[3]*u[1]^2 + a[4]*u[1]*u[2] + a[5]*u[2]^2 + 
            a[6]*u[1]^3 + a[7]* u[1]^2 * u[2] + a[8]*u[1]*u[2]^2 + a[9]*u[2]^3
        du[2] = b[1]*u[1] + b[2]*u[2] + b[3]*u[1]^2 + b[4]*u[1]*u[2] + b[5]*u[2]^2 + 
            b[6]*u[1]^3 + b[7]* u[1]^2 * u[2] + b[8]*u[1]*u[2]^2 + b[9]*u[2]^3
    end
    D_reduced = sqrt(2*D)#/tau) #/tau)
    D1 = D_reduced/X
    D2 = D_reduced*sqrt(epsilon)
    function g!(du, u, p, t)
        du[1] = D1
        du[2] = D2
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


function homotopy_norm2(a,b)
        @var x y
        f₁ = 2*(a[1] * x + a[2] * y + a[3] * x^2 + a[4]* x * y + a[5]* y^2 + 
            a[6] * x^3 + a[7]* x^2 * y + a[8]* x * y^2 + a[9]* y^3) * 
                    (a[1]+2*a[3]*x + a[4]*y + 3*a[6]*x^2 + 2*a[7]*x*y + a[8]*y^2) +
            2*(b[1]*x + b[2]*y + b[3]*x^2 + b[4]*x*y + b[5]*y^2 + 
            b[6]*x^3 + b[7]* x^2 * y + b[8]*x*y^2 + b[9]*y^3) *
                (b[1]+2*b[3]*x + b[4]*y + 3*b[6]*x^2 + 2*b[7]*x*y + b[8]*y^2)
        f₂ = 2*(a[1] * x + a[2] * y + a[3] * x^2 + a[4]* x * y + a[5]* y^2 + 
            a[6] * x^3 + a[7]* x^2 * y + a[8]* x * y^2 + a[9]* y^3) * 
                    (a[2]+ a[4]*x + 2*a[5]*y + a[7]*x^2 + 2*a[8]*x*y + 3*a[9]*y^2) +
            2*(b[1]*x + b[2]*y + b[3]*x^2 + b[4]*x*y + b[5]*y^2 + 
            b[6]*x^3 + b[7]* x^2 * y + b[8]*x*y^2 + b[9]*y^3) *
                (b[2]+ b[4]*x + 2*b[5]*y + b[7]*x^2 + 2*b[8]*x*y + 3*b[9]*y^2)
        F = System([f₁, f₂])
    return F
end



function main(num_IC::Int, D::Float64, t_max::Float64, N_times::Int, fname::String, nus, epsilon::Float64)

    dico = Dict{Int, Vector{Vector{Float64}}}()

    fid = h5open(fname, "w")
    for n = ProgressBar(1:length(nus), printing_delay=1.0)
        nu = nus[n]
        a, b, X, Y, tau = create_fhn_fastslow(nu, epsilon)
        fun!, g!, jac! = make_sys_SDE_fs(a,b, D, X, Y, tau, epsilon)
        times = LinRange(0,t_max, N_times)
        f! = ODEFunction(fun!, jac = jac!)
        trajectories = Array{Float64, 3}(undef, num_IC, length(times), 2)
        
        # Homotopy Continuation  computations
        F = homotopy_sys(a,b)
        norm2_F = homotopy_norm2(a,b)
        
        EOptions = HomotopyContinuation.EndgameOptions(max_endgame_steps = 2000)
        TOptions = HomotopyContinuation.TrackerOptions(automatic_differentiation = 3, max_steps = 10_000)
        
        fixed_points = real_solutions(HomotopyContinuation.solve(F; show_progress= false, start_system = :polyhedral, endgame_options=EOptions, tracker_options=TOptions))
        local_minima = real_solutions(HomotopyContinuation.solve(norm2_F; show_progress= false, start_system = :polyhedral, endgame_options=EOptions, tracker_options=TOptions))
        for i = 1:num_IC
            u0 = zeros(Float64, 2)
            prob = SDEProblem(fun!, g!, u0, (0.0, times[end]))
            sol = DifferentialEquations.solve(prob, SKenCarp(), alg_hints=[:stiff], saveat=times, maxiters=5e8)
            trajectories[i,:,1] = X*getindex.(sol.u, 1)
            trajectories[i,:,2] = Y*getindex.(sol.u, 2)
        end
        fid["$(n)"] = trajectories
        attributes(fid["$(n)"])["params"] = hcat(a,b)
        attributes(fid["$(n)"])["Dscales"] = [X, Y, tau, D]

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
        
        # Norm of minima: compute 'ghostliness'
        
        speeds = zeros(length(local_minima))
        for i = 1:length(local_minima)
            lm = local_minima[i]
            lm[1] = lm[1]/X
            lm[2] = lm[2]/Y
            du = zeros(2)
            fun!(du, lm, 0., 0.)
            speeds[i] = norm(du)
        end
        
        # error handling in case of HC fail
        if length(fixed_points)>0
            attributes(fid["$(n)"])["fps"] = reduce(hcat,fixed_points)
            attributes(fid["$(n)"])["stabilities"] = stabilities
        else
            attributes(fid["$(n)"])["fps"] = 0
            attributes(fid["$(n)"])["stabilities"] = 0
        end
        
        if length(local_minima)>0
            attributes(fid["$(n)"])["lms"] = reduce(hcat,local_minima)
            attributes(fid["$(n)"])["speeds"] = speeds
        else
            attributes(fid["$(n)"])["lms"] = 0
            attributes(fid["$(n)"])["speeds"] = 0
        end
    
    end
    close(fid)
    println("~success~")
end

## with fixed times SDE

rng = MersenneTwister(1234550980);


num_IC = 30 # 1200 # 30 # 1200 # 30 
D = 0.0001 #diffusion constant
t_max = 200. # 0.1 #50 # 0.1 
N_times = 400 # 10 # 400 # 10 
fname = "dataSDE_fastslow_linear_tmax200_n30_D0d0001.h5"
epsilon = 0.1
nus = collect(2.01:0.05:5)

main(num_IC, D, t_max, N_times, fname, nus, epsilon)

