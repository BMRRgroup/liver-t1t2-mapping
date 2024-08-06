module ReconBMRR

using MRIOperators
using RegularizedLeastSquares
using MRICoilSensitivities
using Statistics
using MultivariateStats
using LinearAlgebra
using SparsityOperators
using FourierTools
using Distances
using Clustering
using NFFTTools
using ImageTransformations
using Interpolations
using CUDA
using FLoops
using CuNFFT
using StatsBase
using MAT
using DataFrames
using CSV
using HDF5
using ProgressMeter
using DataStructures
using NFFT
using Plots
using NaNStatistics
using StaticArrays
using NPZ
# using MATLAB
using Suppressor
using Mmap
using ImageFiltering
using PyCall

include("Helper.jl")
include("Trajectories.jl")
include("ReconParams.jl")
# include("_Hidden.jl")
include("Operators/StackOfStarsOp.jl")
include("Operators/NFFTOp.jl")
include("Operators/DiagOp.jl")
include("Preprocessing.jl")
include("CoilSensitvities.jl")
include("Motion.jl")
include("Reconstruction.jl")
include("Postprocessing.jl")
include("Export.jl")

export perform

function perform(r::ReconParams)
    if r.reconParameters[:profileCorrectionFINO] && r.scanParameters[:isFINO]
        profileCorrectionFINO!(r)
    end
    r = sortData(r)

    # KdataPreprocessed
    if r.reconParameters[:motionGating] == :motionStatesSelfGating!
        motionStatesSelfGating!(r)
    end
    if r.reconParameters[:artificalUndersampling] > 0
        artificalUndersampling!(r)
    end
    if r.reconParameters[:useDoublePrecision]
        r = changePrecision(r)
    end
    coilSensitivities!(r)

    r = iterativeRecon_perEchoDyn(r)

    # ImgData
    if r.reconParameters[:upsampleRecVoxelSize]
        upsampleRecVoxelSize!(r)
    end
    if r.reconParameters[:removeOversampling]
        removeOversampling!(r)
    end
    return r
end

end # module