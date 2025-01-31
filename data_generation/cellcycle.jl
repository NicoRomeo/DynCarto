""" Integrates the cell cycle model from Tyson, PNAS 1991 with additive noise """
using HDF5
using Random
using LinearAlgebra
using DifferentialEquations
using ProgressBars
using Base.Threads


rng = MersenneTwister(1234567);

function cellcycle!(du, u, p, t)
    """
    Dynamics of a cell cycle model from Tyson, PNAS 1991 
    """
    C2 = u[1]
    CP = u[2]
    pM = u[3]
    M = u[4]
    Y = u[5]
    YP = u[6]
    
    CT = C2 + CP + pM + M
    
    # dC2
    du[1]  = p.k6 * M - p.k8tP*C2 + p.k9 * CP
    # dCP
    du[2] = - p.k3ct * CP / CT * Y + p.k8tP*C2 - p.k9*CP
    # dpM
    du[3] = p.k3ct * CP / CT * Y - pM * (p.k4p + p.k4*(M/CT)^2) + p.k5tP*M
    # dM
    du[4] = pM * (p.k4p + p.k4*(M/CT)^2) - p.k5tP*M - p.k6*M
    # dY
    du[5] = p.k1aact*CT - p.k2*Y - p.k3ct*CP/CT*Y
    # dYP
    du[6] = p.k6*M - p.k7*YP
    nothing
end


function noise!(du, u, p, t)
    """ Not a chemically accurate noise """
    du .= p.noise .* u 
    nothing
end



function main(n_sys::Int, num_IC::Int, t_max::Float64, N_times::Int, fname::String, k4_min::Float64, k4_max::Float64, k6_min::Float64, k6_max::Float64, noiseD::Float64, c0::Float64)

    
    k4_range = 10 .^ (LinRange(log10(k4_min), log10(k4_max), n_sys))
    k6_range = 10 .^ (LinRange(log10(k6_min), log10(k6_max), n_sys))
    k4s = [k4 for k4 in k4_range, k6 in k6_range]
    k6s = [k6 for k4 in k4_range, k6 in k6_range]
    

    fid = h5open(fname, "w")
    for n = ProgressBar(1:length(k4s), printing_delay=1.0)
        

        p = (k1aact=0.015,
            k2 = 0.,
            k3ct = 200.,
            k4 = k4s[n],  #ranges from 10-1000 min^-1
            k4p = 0.018,
            k5tP = 0.,
            k6= k6s[n],
            k7 = 0.6,
            k8tP = 1000., # >> k9
            k9 = 100, # >> k6
            noise = noiseD
        )

        t_max = maximum([200., 200. * 0.015/p.k1aact, 20. / p.k6 ])
        times = LinRange(0,t_max, N_times)

        #trajectories = Array{Float64, 3}(0., num_IC, length(times), 6)
        trajectories = zeros(num_IC, length(times), 6)
        converged = true
        for i = 1:num_IC

            u0 = [10., 1., 1., 1., 1., 1.] .* c0
            
            condition(u,t,integrator) = true
            function affect!(integrator)

                if any(x->x<0, integrator.u) < 0
                    # set to zero
                    integrator.u .= max.(integrator.u, 0.)
                end
            end
            cb = DiscreteCallback(condition,affect!;save_positions=(false,false))
            
            
            prob = SDEProblem{true}(cellcycle!, noise!, u0, (0.0, t_max), p)

            sol = DifferentialEquations.solve(prob, SKenCarp(), alg_hints=[:stiff], saveat=times, maxiters=1e7,  callback = cb)
            if sol.retcode != ReturnCode.Success
                    converged = false
                    break
                end
            #C2 = getindex.(sol.u, 1)
            #CP = getindex.(sol.u, 2)
            #pM = getindex.(sol.u, 3)
            #M = getindex.(sol.u, 4)
            #Y = getindex.(sol.u, 5)
            #YP = getindex.(sol.u, 6)
            trajectories[i,:,1] = getindex.(sol.u, 1)
            trajectories[i,:,2] = getindex.(sol.u, 2)
            trajectories[i,:,3] = getindex.(sol.u, 3)
            trajectories[i,:,4] = getindex.(sol.u, 4)
            trajectories[i,:,5] = getindex.(sol.u, 5)
            trajectories[i,:,6] = getindex.(sol.u, 6)
        end
        fid["$(n)"] = trajectories
        attributes(fid["$(n)"])["param_k4k6D"] = [k4s[n], k6s[n], p.noise, Int(converged)]


    
    end
    close(fid)
    println("~success~")
end

## Integration of SDE

n_sys = 64 # a bit of a misnomer: we sample parameters on a (n_sys x n_sys) log-grid in k4, k6 
k4_min = 10. #p.k6/400
k4_max = 1000. #p.k6/10
k6_min =  0.1 #0.005 # 0.5*min(p.k6*sqrt(p.k6/p.k4), p.k6*sqrt(k4_min/p.k4))#0.1 #currently used for k1
k6_max = 10. #2*p.k6*sqrt(p.k6/p.k4) #10.
noiseD = sqrt(2*0.001)
c0 = 50.
num_IC = 30
t_max = 200. 
N_times = 450 
fname = "tysonSDE4096_k4k6_tmax200_n30_noise001all_c50_moretimes.h5"

main(n_sys, num_IC, t_max, N_times, fname, k4_min, k4_max, k6_min, k6_max, noiseD, c0)
