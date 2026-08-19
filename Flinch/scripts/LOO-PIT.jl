using Pkg
Pkg.activate(".")
Pkg.instantiate()
Pkg.resolve()

using Distributions
using LinearAlgebra
using StatsFuns
using StatsBase

using LogDensityProblems
using LogDensityProblemsAD
using AbstractDifferentiation
using MCMCDiagnosticTools
using AdvancedHMC
using MicroCanonicalHMC
using Pathfinder
using Transducers

using Healpix
using HealpixMPI
using MPI

using Plots
using StatsPlots
using LaTeXStrings
using DataFrames

using Random
using ProgressMeter
using BenchmarkTools
using NPZ
using CSV
using Test

using Zygote: @adjoint
using Zygote
using ChainRules.ChainRulesCore

include("AD_parallSHT.jl")
include("gen_datatest.jl")
include("reparam.jl")
include("utils.jl")
include("inference_model.jl")
include("neglogproblem.jl")

# using Turing
# using Capse
using NPZ
# # using PlanckLite
# using BenchmarkTools
# using Plots
# using Optim
# using ForwardDiff
# using Pathfinder
# #using MicroCanonicalHMC
# #using Transducers
# using StatsPlots
# #using PairPlots
# #using LaTeXStrings
# #using CairoMakie
# using MCMCChains
using PSIS
# include("utils.jl");
using ArviZ
#using ArviZPythonPlots
using LinearAlgebra
using IrrationalConstants
using DimensionalData
using Statistics

Threads.nthreads()

# files = filter(
#     f -> endswith(f, "_mask_NUTS_nside_16.npy"),
#     readdir("/Users/ethansmith/Desktop/Waterloo/Masters/Andrea_flinch_scripts/scripts/MPI_chains"; join=true)
# )

# for file in files
#     println(file)
# end

# sort!(files)

# sampled_params = hcat((npzread(f) for f in files)...)

files = ["/Users/ethansmith/Desktop/Waterloo/Masters/LOO_PIT_Manual/Flinch/scripts/MPI_Chains/nside_16/aNW8N_mask_NUTS_nside_16.npy"]
sampled_params = npzread("/Users/ethansmith/Desktop/Waterloo/Masters/LOO_PIT_Manual/Flinch/scripts/MPI_Chains/nside_16/aNW8N_mask_NUTS_nside_16.npy")

samples = size(sampled_params, 2)

ids = [
    replace(splitext(basename(f))[1],
            "_mask_NUTS_nside_16" => "")
    for f in files
]

runname = join(ids, "_")

seed = 1123
Random.seed!(seed)

#   RESOLUTION PARAMETERS
nside = 16
lmax = 2*nside - 1

MPI.Init()

comm = MPI.COMM_WORLD
crank = MPI.Comm_rank(comm)
csize = MPI.Comm_size(comm)
root = 0
ncore = 32

#   REALIZATION MAP
realiz_Cl, realiz_HAlm, realiz_HMap = Realization("Capse_fiducial_Dl.csv", nside, lmax, seed)
realiz_θ = vcat(x_vecmat2vec(from_healpix_alm_to_alm([realiz_HAlm], lmax, 1, comm, root=root), lmax, 1, comm, root=root), Cl2Kl(realiz_Cl))

d = length(realiz_θ)

#   SURVEY MASK
mask_512 = readMapFromFITS("wmap_temperature_kq85_analysis_mask_r9_9yr_v5.fits",2,Float64)
mask_nside = udgrade(nest2ring(mask_512), nside)
for i in 1:length(mask_nside.pixels)
    if mask_nside.pixels[i]<=0.5
        mask_nside.pixels[i]=1
    else
        mask_nside.pixels[i]=0
    end
end

#   GENERATED DATA MEASUREMENTS
#   Noise
ϵ=150
N = ϵ*ones(nside2npix(nside))
N[mask_nside.==1] .= 5*10^4
#   Gaussian beam and pixel window function
Bl = ones(length(realiz_Cl))#gaussbeam(0.001, lmax, pol=false)
Pl = ones(length(realiz_Cl))#pixwin(nside, pol=false)[1:lmax+1]
BP_l = Bl.*Pl

#   Data Map
gen_Cl, gen_HAlm, gen_HMap = Measurement(realiz_Cl, Bl, Pl, mask_nside, N, nside, lmax, seed)
gen_θ = vcat(x_vecmat2vec(from_healpix_alm_to_alm([gen_HAlm], lmax, 1, comm, root=root), lmax, 1, comm, root=root), Cl2Kl(gen_Cl))
invN_HMap = HealpixMap{Float64,RingOrder}(1 ./ N)

#   STARTING POINT
start_Cl, start_HAlm, start_HMap = StartingPoint(gen_Cl, nside)
start_θ = vcat(x_vecmat2vec(from_healpix_alm_to_alm([start_HAlm], lmax, 1, comm, root=root), lmax, 1, comm, root=root), Cl2Kl(start_Cl))

#   PROMOTE HEALPIX.MAP TO HEALIPIXMPI.DMAP
gen_DMap = DMap{RR}(comm)
invN_DMap = DMap{RR}(comm)
HealpixMPI.Scatter!(gen_HMap, gen_DMap, comm, clear=true)
HealpixMPI.Scatter!(invN_HMap, invN_DMap, comm, clear=true)

helper_DMap = deepcopy(gen_DMap)

θ = sampled_params[:,1500]
alm = θ[1:length(θ)-lmax-1]
sample_alm = x_vec2vecmat(alm, lmax, 1, comm, root=root)
sample_HAlm = from_alm_to_healpix_alm(sample_alm, lmax, 1)[1] # comm, root=root)[1]
sample_DAlm = HAlm2DAlm(sample_HAlm, comm; clear=true, root=root)

sample_HMap = Healpix.alm2map(sample_HAlm, nside)
plot(HealpixMap{Float64, RingOrder}(sample_HMap.pixels[:, 1]))

pixels = length(gen_DMap.pixels) 
NLLs = Matrix{Any}(undef, pixels, samples)
NLLs_HMap = [HealpixMap{Float64, RingOrder}(nside) for i in 1:samples]
samples_HMap = [HealpixMap{Float64, RingOrder}(nside) for i in 1:samples]

for n in 1:samples
    θ = sampled_params[:,n]
    alm = θ[1:length(θ)-lmax-1]
    sample_alm = x_vec2vecmat(alm, lmax, 1, comm, root=root)
    sample_HAlm = from_alm_to_healpix_alm(sample_alm, lmax, 1, comm, root=root)[1]
    sample_DAlm = HAlm2DAlm(sample_HAlm, comm; clear=true, root=root)
    
    Threads.@threads for i in 1:pixels
        NLL = NLL_Pixel(sample_DAlm, helper_DMap, ncore, gen_DMap, invN_DMap, BP_l, i)
        NLLs[i,n] = NLL
    end

    NLLs_pixels_n = map(x -> x === nothing ? NaN : x, NLLs[:, n])
    helper_DMap.pixels = reshape(NLLs_pixels_n, :, 1)
    MPI.Gather!(helper_DMap, NLLs_HMap[n])

    sample_HMap = Healpix.alm2map(sample_HAlm, nside)
    samples_HMap[n] = sample_HMap
end


NLLs_pixels = reduce(hcat, [NLLs_HMap[i].pixels for i in 1:samples])

plot(NLLs_HMap[1000])

plot(samples_HMap[1000])

psis_result = Vector{PSISResult{Float64, Vector{Float64}, Int64, Int64, PSIS.GeneralizedPareto{Float64}}}(undef, pixels);
for i in 1:pixels
    psis_result[i] = psis(NLLs_pixels[i,:])
end

pareto_shapes = Vector{Float64}()
for i in 1:pixels
    append!(pareto_shapes, psis_result[i].pareto_shape)
end
println("Pareto Shapes: ", pareto_shapes)

shape_plot = Plots.scatter(pareto_shapes; marker =:+, markersize=6, legend = false, linewidth=2)
hline!([0.5,0.7], linestyle=:dot, color =:black)
hline!([1.0], linestyle=:solid, color=:black)

psis_shapes_hist = histogram(pareto_shapes)
vline!([0.7])

failure_count = 0                       #counts the number of pixels who have a pareto shape value exceeding the threshhold of 0.7
for i in 1:pixels
    if pareto_shapes[i] > 0.7
        global failure_count += 1
    end
end

unmasked_pixels = 0                     #counts the total number of pixels that are not covered by the survey mask
for i in 1:pixels
    if !isnan(pareto_shapes[i])
        global unmasked_pixels += 1
    end
end

failure_rate = failure_count / unmasked_pixels
println(failure_count)
println(failure_rate)

mkpath("PSIS_results/$(samples)_$(nside)/$runname")
npzwrite("PSIS_results/$(samples)_$(nside)/$runname/pareto_shapes.npy", pareto_shapes)
npzwrite("PSIS_results/$(samples)_$(nside)/$runname/failure_rate.npy", [failure_count, unmasked_pixels, failure_rate])

mkpath("NLLs/$(samples)_$(nside)/$runname")
npzwrite("NLLs/$(samples)_$(nside)/$runname/NLLs.npy", NLLs_pixels)

Plots.savefig(shape_plot, "PSIS_results/$(samples)_$(nside)/$runname/pareto_shapes.png")
Plots.savefig(psis_shapes_hist, "PSIS_results/$(samples)_$(nside)/$runname/pareto_shapes_histogram.png")

plot(mask_nside)

shape_mapping = HealpixMap{Float64, RingOrder, Vector{Float64}}(pareto_shapes)
shape_map = plot(shape_mapping)

failure_vector = Vector{Float64}(undef, pixels)

for i in 1:pixels
    if isnan(pareto_shapes[i])
        failure_vector[i] = NaN
    elseif pareto_shapes[i] > 0.7
        failure_vector[i] = 1.0
    else
        failure_vector[i] = 0.0  
    end
end

failure_mapping = HealpixMap{Float64, RingOrder, Vector{Float64}}(failure_vector)
failure_map = plot(failure_mapping)

Plots.savefig(failure_map, "PSIS_results/$(samples)_$(nside)/$runname/failure_map.png")
Plots.savefig(shape_map, "PSIS_results/$(samples)_$(nside)/$runname/shape_map.png")

mask = .!isnan.(gen_HMap.pixels)
y_obs = gen_HMap.pixels;

#sampled_observed_maps = Vector{HealpixMap{Float64,RingOrder}}(undef, samples)
preds = Matrix{Float64}(undef, samples, pixels)

for s in 1:samples

    # Convert sampled parameters θ to Healpix alm
    θ = sampled_params[:,s]
    alm = θ[1:end-(lmax+1)]
    sample_alm = x_vec2vecmat(alm, lmax, 1, comm, root=root)
    sample_HAlm = from_alm_to_healpix_alm(sample_alm, lmax, 1, comm, root=root)[1]

    # Apply beam and pixel window
    pred_HAlm = almxfl(sample_HAlm, BP_l)

    # Convert to map
    sampled_observed_map = Healpix.alm2map(pred_HAlm, nside)


    # Independent observational noise
    noise = rand( MvNormal(zeros(length(N)), Diagonal(N)) )

    # Posterior predictive map
    pred_pixels = sampled_observed_map.pixels .+ noise

    # Keep observed pixels only
    preds[s,:] .= pred_pixels[mask]

end

loo_pit_values = Vector{Float64}(undef, pixels)

for j in 1:pixels

    # Retrieve PSIS log weights
    logw = copy(psis_result[j].log_weights)

    # Normalize the log weights
    logw .-= maximum(logw)

    w = exp.(logw)

    w ./= sum(w)

    # Weighted empirical CDF
    loo_pit_values[j] = sum(w .* (preds[:,j] .<= y_obs[j]))

end

loo_pit_plot = histogram(loo_pit_values, bins=20, normalize=:pdf, xlabel="LOO-PIT", ylabel="Density", title="PSIS LOO-PIT", legend = false)

mkpath("LOO-PIT_Plots/$(samples)_$(nside)/$runname")

Plots.savefig(loo_pit_plot, "LOO-PIT_Plots/$(samples)_$(nside)/$runname/loo_pit_plot.png")
