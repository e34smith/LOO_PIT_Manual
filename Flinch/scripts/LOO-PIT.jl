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
using ArviZPythonPlots
using LinearAlgebra
using IrrationalConstants
using DimensionalData
using Statistics
using Plots

Threads.nthreads()

files = filter(
    f -> endswith(f, "_mask_NUTS_nside_16.npy"),
    readdir("/Users/ethansmith/Desktop/Waterloo/Masters/Andrea_flinch_scripts/scripts/MPI_chains"; join=true)
)

# for file in files
#     println(file)
# end

# sort!(files)

#sampled_params = hcat((npzread(f) for f in files)...)

sampled_params = npzread("/Users/ethansmith/Desktop/Waterloo/Masters/Andrea_flinch_scripts/scripts/MPI_chains/2yjXc_mask_NUTS_nside_16.npy")
samples = size(sampled_params, 2)

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

# θ = sampled_params[:,1]
# alm = θ[1:length(θ)-lmax-1]
# sample_alm = x_vec2vecmat(alm, lmax, 1, comm, root=root)
# sample_HAlm = from_alm_to_healpix_alm(sample_alm, lmax, 1, comm, root=root)[1]
# sample_DAlm = HAlm2DAlm(sample_HAlm, comm; clear=true, root=root)

# NLL_Pixel(sample_DAlm, helper_DMap, ncore, gen_DMap, invN_DMap, BP_l, 1354)

pixels = length(gen_DMap.pixels)

NLLs = Matrix{Any}(undef, pixels, samples)

for n in 1:samples
    Threads.@threads for i in 1:pixels
        θ = sampled_params[:,n]
        alm = θ[1:length(θ)-lmax-1]
        sample_alm = x_vec2vecmat(alm, lmax, 1, comm, root=root)
        sample_HAlm = from_alm_to_healpix_alm(sample_alm, lmax, 1, comm, root=root)[1]
        sample_DAlm = HAlm2DAlm(sample_HAlm, comm; clear=true, root=root)
        NLL = NLL_Pixel(sample_DAlm, helper_DMap, ncore, gen_DMap, invN_DMap, BP_l, i)
        NLLs[i,n] = NLL
    end
end

NLLs

function count_nothings(NLLs, pixels, samples)
    nothing_count = 0
    something_count = 0

    for i in 1:pixels
        for j in 1:samples
            if NLLs[i,j] === nothing
                nothing_count += 1
            else
                something_count += 1
            end
        end
    end

    return nothing_count, something_count
end

nothing_count, something_count = count_nothings(NLLs, pixels, samples)

println(nothing_count)
println(something_count)

switch_vector = Vector{Int64}()
for i in 1:(pixels-1)
    if (NLLs[i,1] == nothing && NLLs[i+1, 1] != nothing) || (NLLs[i,1] != nothing && NLLs[i+1, 1] == nothing)
        append!(switch_vector, i)
    end
end
println(length(switch_vector))
println(switch_vector)

unmasked_pixels = 0
for i in 1:pixels
    if NLLs[i, 1] != nothing
        global unmasked_pixels += 1
    end
end

unmasked_NLLs = Matrix{Float64}(undef, unmasked_pixels, samples)
unmasked_pixels = 0
for i in 1:pixels
    if NLLs[i, 1] != nothing
        global unmasked_pixels += 1
        unmasked_NLLs[unmasked_pixels,:] = NLLs[i,:]
    end
end

unmasked_pixels

unmasked_NLLs

histogram(unmasked_NLLs[:,1])

psis_result = Vector{PSISResult{Float64, Vector{Float64}, Int64, Int64, PSIS.GeneralizedPareto{Float64}}}(undef, unmasked_pixels);
for i in 1:unmasked_pixels
    psis_result[i] = psis(unmasked_NLLs[i,:])
end

pareto_shapes = Vector{Float64}()
for i in 1:unmasked_pixels
    append!(pareto_shapes, psis_result[i].pareto_shape)
end
println("Pareto Shapes: ", pareto_shapes)

shape_plot = Plots.scatter(pareto_shapes; marker =:+, markersize=6, legend = false, linewidth=2)
hline!([0.5,0.7], linestyle=:dot, color =:black)
hline!([1.0], linestyle=:solid, color=:black)

psis_shapes_hist = histogram(pareto_shapes)
vline!([0.7])

failure_count= 0 
for i in 1:unmasked_pixels
    if pareto_shapes[i] > 0.7
        global failure_count += 1
    end
end
failure_rate = failure_count / unmasked_pixels
println(failure_count)
println(failure_rate)

mkpath("PSIS_results")
npzwrite("PSIS_results/pareto_shapes_$(samples)_$(nside).npy", pareto_shapes)
npzwrite("PSIS_results/failure_rate_$(samples)_$(nside).npy", [failure_count, unmasked_pixels, failure_rate])
Plots.savefig(shape_plot, "PSIS_results/pareto_shapes.png")
Plots.savefig(psis_shapes_hist, "PSIS_results/pareto_shapes_histogram.png")


