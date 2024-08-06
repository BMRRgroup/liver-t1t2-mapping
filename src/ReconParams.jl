export ReconParams

"""
kspace & reconstructed data types
"""
abstract type AbstractReconData end
abstract type AbstractKdata <: AbstractReconData end

struct KdataRaw{T<:AbstractFloat} <: AbstractKdata
    accImagData::Matrix{Complex{T}}
    rejImagData::Matrix{Complex{T}}
    phaseCorrData::Matrix{Complex{T}}
    freqCorrData::Matrix{Complex{T}}
    noiseData::Matrix{Complex{T}}
    labels::Dict{Symbol,Any}
end

struct KdataPreprocessed{T<:AbstractFloat} <: AbstractKdata
    kdata::Array{Complex{T},7} # kx, ky, kz, echoes, dynamics, channels, interleaves
end

struct ImgData{T<:AbstractFloat} <: AbstractReconData
    imgData::Array{Complex{T}}
end

"""
Struct describing MRI acquisition

And another environment variable if figures shouldn't be opened in the GUI
    -> GKSwstype="nul"
"""
mutable struct ReconParams{T<:AbstractReconData, V<:AbstractTrajectory}
    filename::String
    pathProc::String 
    scanParameters::Dict{Symbol,Any}
    reconParameters::Dict{Symbol,Any}
    data::T
    traj::V
    performedMethods::Vector{Symbol}
end

function setDefaultReconParameters!(r::ReconParams)
    r.reconParameters = Dict(
        :cuda => has_cuda_gpu(),
        :useDoublePrecision => false,
        :artificalUndersampling => 0,
        :export => Dict{Symbol,Any}(),
        :iterativeReconParams => setIterativeReconParams(),
        :removeKxOversampling => r.scanParameters[:AcqMode] == "Radial" ? false : true,
        :profileCorrectionFINO => r.scanParameters[:isFINO],
        :ringingFilter => false,
        :changeResolution => false,
        :coilSensitivities => :ESPIRiT, #:MRecon,
        :motionGating => r.scanParameters[:AcqMode] == "Radial" ? :motionStatesSelfGating! : false,
        :motionGatingParams => setMotionGatingParams(r.scanParameters[:AcqMode] == "Radial"),
        :motionStatesRecon => r.scanParameters[:AcqMode] == "Radial" ? "2" : nothing,
        :noisePreWhitening => false,
        :radialPhaseCorrection => false,
        :deltaB0Estimation => false,
        :deltaB0Correction => true,
        :deltaB0Params => setDeltaB0Params(r.scanParameters[:AcqMode] == "Radial"),
        :intensityCorrection => false,
        :upsampleRecVoxelSize => r.scanParameters[:AcqMode] == "Radial" ? true : nothing,
        :removeOversampling => true
    )
end

function setIterativeReconParams()
    return Dict(
        :Regularization => Dict(
            :L1Wavelet_spatial => 0.0,
            :LLR => 0.0,
            :TV_spatialTemporal => 0.0,
            :TV_spatial => 0.03,
            :TV_temporal => 0.3,
        ),
        :solver => "admm",
        :normalizeReg => true,
        :iterations => 10,
        :iterationsInner => 5,
        :vary_ρ => :balance,
        :verboseIteration => false
    )
end

function setMotionGatingParams(isRadial::Bool)
    return isRadial ? Dict(
        :numClusters => 5,
        :doPlotBcurve => true,
        :method => :relDisplacement,
        :eddyCurrentCorrection => true
    ) : Dict{Symbol,Any}()
end

function setDeltaB0Params(isRadial::Bool)
    return isRadial ? Dict(
        :doCorrection => true,
        :doPlotMaps => true,
        :corr2D => true
    ) : Dict{Symbol,Any}()
end

function accImagDataLabels(r::ReconParams{KdataRaw{T}, <:AbstractTrajectory}) where T<:AbstractFloat
    return [(Int.(r.data.labels[:LabelLookupTable][1]))...]
end

function numKx(r::ReconParams{KdataRaw{T}, <:AbstractTrajectory}) where T<:AbstractFloat
    return size(r.data.accImagData, 1)
end

function numKx(r::ReconParams{KdataPreprocessed{T}, <:AbstractTrajectory}) where T<:AbstractFloat
    return size(r.data.kdata, 1)
end

function numKy(r::ReconParams{KdataRaw{T}, <:AbstractCartesian}) where T<:AbstractFloat
    indices = accImagDataLabels(r)
    return maximum(r.data.labels[:ky][indices]) - minimum(r.data.labels[:ky][indices]) + 1
end

function numKy(r::ReconParams{KdataRaw{T}, <:AbstractNonCartesian}) where T<:AbstractFloat
    indices = accImagDataLabels(r)
    return  length(unique(r.data.labels[:ky][indices]))
end

function numKy(r::ReconParams{KdataPreprocessed{T}, <:AbstractTrajectory}) where T<:AbstractFloat
    return size(r.data.kdata, 2)
end

function numKz(r::ReconParams{KdataRaw{T}, <:AbstractTrajectory}) where T<:AbstractFloat
    indices = accImagDataLabels(r)
    return maximum(r.data.labels[:kz][indices]) - minimum(r.data.labels[:kz][indices]) + 1
end

function numKz(r::ReconParams{KdataPreprocessed{T}, <:AbstractTrajectory}) where T<:AbstractFloat
    return size(r.data.kdata, 3)
end

function numEchoes(r::ReconParams{KdataRaw{T}, <:AbstractTrajectory}) where T<:AbstractFloat
    indices = accImagDataLabels(r)
    return length(unique(r.data.labels[:echo][indices]))
end

function numEchoes(r::ReconParams{KdataPreprocessed{T}, <:AbstractTrajectory}) where T<:AbstractFloat
    return size(r.data.kdata, 4)
end

function numDyn(r::ReconParams{KdataRaw{T}, <:AbstractTrajectory}) where T<:AbstractFloat 
    indices = accImagDataLabels(r)
    return length(unique(r.data.labels[:dyn][indices]))
end

function numDyn(r::ReconParams{KdataPreprocessed{T}, <:AbstractTrajectory}) where T<:AbstractFloat
    return size(r.data.kdata, 5)
end

function numChan(r::ReconParams{KdataRaw{T}, <:AbstractTrajectory}) where T<:AbstractFloat 
    indices = accImagDataLabels(r)
    return length(unique(r.data.labels[:chan][indices]))
end

function numChan(r::ReconParams{KdataPreprocessed{T}, <:AbstractTrajectory}) where T<:AbstractFloat
    return size(r.data.kdata, 6)
end

function numInterleaves(r::ReconParams{KdataRaw{T}, <:AbstractTrajectory}) where T<:AbstractFloat 
    indices = accImagDataLabels(r)
    return length(unique(r.data.labels[:extr1][indices]))
end

function numInterleaves(r::ReconParams{KdataPreprocessed{T}, <:AbstractTrajectory}) where T<:AbstractFloat
    return size(r.data.kdata, 7)
end

function getACregion(r::ReconParams; eddyCurrentCorrection::Bool=true, removeOversampling::Bool=true)
    # Get self-gating data
    centerKx = Int(size(r.data.kdata,1)/2) + 1
    data = r.data.kdata[centerKx,:,:,:,:,:]

    # if r.reconParameters[:coilSensitivities] == :MRecon
    data = ifftshift(ifft(ifftshift(data, (2)), 2), (2))
    # end

    if removeOversampling
        # Remove oversampling
        center = div(size(data,2), 2)
        encodingSize = r.scanParameters[:encodingSize]
        data = data[:,center-div(encodingSize[3], 2)+1:center-div(encodingSize[3], 2)+encodingSize[3],:,:,:]
    end
    numSlices = size(data,2)

    if eddyCurrentCorrection
        # Change to profile order
        ky = r.reconParameters[:uniqueKy]
        data[ky.+1,:,:,:,:] = data
        data = permutedims(data, [1,2,3,5,4])
        data = reshape(data, numKy(r), numChan(r)*numDyn(r)*numEchoes(r)*numSlices)

        # Do correction according to Rosenzweig et al.
        nH = 5
        phi = diff(collect(2*pi.-2*pi.*(0:numKy(r)-1)/numKy(r)))[1]
        nCorr = ones(ComplexF32, numKy(r), 2*nH)
        for i in 1:2*nH
            for j in 1:numKy(r)
                nCorr[j, i] = exp(1im*(-1)^i*ceil(i/2)*(j-1)*phi)
            end
        end
        data = reshape(transpose(data - nCorr*(pinv(nCorr)*data)), numChan(r)*numEchoes(r)*numSlices, numDyn(r),
                        numKy(r))
        data = permutedims(data, [1, 3, 2])

        # Change to temporal order
        data = data[:,ky.+1,:]
    else
        data = permutedims(data, [1,2,3,5,4])
        data = reshape(data, numKy(r), numChan(r)*numDyn(r)*numEchoes(r)*numSlices)
        data = reshape(transpose(data), numChan(r)*numEchoes(r)*numSlices, numDyn(r),
                        numKy(r))
        data = permutedims(data, [1, 3, 2])
    end
    return data
end
