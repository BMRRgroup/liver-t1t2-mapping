export removeOversampling!, sortData, profileCorrectionFINO!, ringingFilter!, 
    plotSamplingMask, changeResolution!, artificalUndersampling!, changePrecision

function profileCorrectionFINO!(r::ReconParams{KdataRaw{T}, <:AbstractTrajectory}) where T<:AbstractFloat
    # TODO: check for correct dynamics
    @debug("Correct for FINO preparation profiles (Preperation order not defined).")
    kzCorr = T.(npzread(string(@__DIR__, "/Files/correction_factors_finot1t2.npy")))
    labels = r.data.labels
    indices = accImagDataLabels(r)
    kzSort = unique(labels[:kz][indices])
    # kzCorrSorted[kzSort] = kzCorr[:, :]
    for j in eachindex(axes(kzCorr,2))
        for (i, kz) in enumerate(kzSort)
            mask = (labels[:kz][indices] .== kz .&& labels[:dyn][indices] .== j-1)
            r.data.accImagData[:,mask] .*= kzCorr[i,j]
        end
    end
    append!(r.performedMethods, [nameof(var"#self#")])
end

function sortData(r::ReconParams{KdataRaw{T}, <:NonCartesian3D}; traj3d::Bool=false) where T<:AbstractFloat
    @debug("Sort radial Stack of Stars data. (Assume kz inner loop!)")
    labels = r.data.labels
    indices = accImagDataLabels(r)
    
    if r.traj.name == :StackOfStars
        echo_labels = labels[:echo][indices]
        kz_labels = labels[:kz][indices]
        chan_labels = labels[:chan][indices]
        num_kx = numKx(r)
        num_chan = numChan(r)
        num_echoes = numEchoes(r)
        num_kz = numKz(r)
        num_ky = numKy(r)
        num_dyn = numDyn(r)

        data = reshape(r.data.accImagData, num_kx, num_chan, num_echoes, num_kz, num_ky, num_dyn, 1)
        traj = zeros(T, traj3d ? 3 : 2, num_kx, num_echoes, num_ky, num_dyn)
        for i in 1:num_echoes
            mask = (echo_labels .== i-1) .& (kz_labels .== 0) 
            masked_chan = mask[chan_labels .== chan_labels[1]]
            
            if traj3d == false
                traj[:,:,i,:,:] = 
                    reshape(r.traj.kdataNodes[1:2, :, masked_chan], 
                    2, num_kx, num_ky, num_dyn)
            else
                traj[:,:,i,:,:] = 
                    reshape(r.traj.kdataNodes[:, :, masked_chan], 
                    3, num_kx, num_ky, num_dyn)
            end
        end

        data = permutedims(data, [1, 5, 4, 3, 6, 2, 7])
        if traj3d == false
            data = sortDataKzStackofStars(r, data)
        end

        r.reconParameters[:uniqueKy] = unique(labels[:ky][indices])
        traj = NonCartesian3D(permutedims(traj, [1, 2, 4, 3, 5]), r.traj.name)
        data = KdataPreprocessed(data)
        append!(r.performedMethods, [nameof(var"#self#")])
        return ReconParams(r.filename, r.pathProc, r.scanParameters, r.reconParameters, 
            data, traj, r.performedMethods)
    end
end

function sortDataKzStackofStars(r::ReconParams, data::AbstractArray{Complex{T}}) where T<:AbstractFloat
    labels = r.data.labels
    indices = accImagDataLabels(r)
    for i in 1:numDyn(r)
        mask = (labels[:dyn][indices] .== i-1)
        sortKz = unique(labels[:kz][indices][mask] .- minimum(labels[:kz][indices]) .+ 1)
        data[:,:,sortKz,:,i,:,:] = data[:,:,:,:,i,:,:]
    end

    # SENSE zeroFill
    if r.scanParameters[:SENSEFactor][3] == 2.0
        dataTemp = zeros(Complex{T}, size(data,1), size(data,2), 2*size(data,3), 
            size(data,4), size(data,5), size(data,6), size(data,7))
        dataTemp[:,:,2:2:end,:,:,:,:] = data
        data = dataTemp
    elseif r.scanParameters[:SENSEFactor][3] > 1
        error("SENSE factor > 1 not yet implemented!")
    end

    if r.scanParameters[:HalfScanFactors][2] < 1
        kzFull = Int(minimum(abs.(r.scanParameters[:KzRange])))
        numSlices = 2*numKz(r)-(2*kzFull+1)
        dataTemp = zeros(Complex{T}, size(data,1), size(data,2), numSlices, 
            size(data,4), size(data,5), size(data,6), size(data,7))
        dataTemp[:,:,end-numKz(r)+1:end,:,:,:,:] = data
        data = dataTemp
    end
    return data
end

function ringingFilter!(r::ReconParams{KdataPreprocessed{T}, <:AbstractTrajectory}) where T<:AbstractFloat
    @debug("Apply ringing filter")
    dims = (r.traj.name == :StackOfStars) ? [1,3] : 1:3
    r.data = KdataPreprocessed(hamming(r.data.kdata, dims=collect(dims)))
    append!(r.performedMethods, [nameof(var"#self#")])
end

function artificalUndersampling!(r::ReconParams{KdataPreprocessed{T}, <:NonCartesian3D}) where T<:AbstractFloat
    @debug("Apply artifical undersampling.")
    removedKy = round(Int, numKy(r)*r.reconParameters[:artificalUndersampling])
    traj = r.traj.kdataNodes[:,:,1:end-removedKy,:,:]
    data = r.data.kdata[:,1:end-removedKy,:,:,:,:,:]
    if :motionStates in keys(r.reconParameters)
        bCurve = reshape(r.reconParameters[:motionStates], numKy(r), numDyn(r))
        bCurve = bCurve[1:end-removedKy, :]
        r.reconParameters[:motionStates] = reshape(bCurve,:)
    end
    r.data = KdataPreprocessed(data)
    r.traj = NonCartesian3D(traj, r.traj.name)
    append!(r.performedMethods, [nameof(var"#self#")])
end

function changePrecision(r::ReconParams{KdataPreprocessed{T}, J}) where {T<:AbstractFloat, J<:AbstractTrajectory}
    @debug("Change to double precision.")
    data = KdataPreprocessed(ComplexF64.(r.data.kdata))
    traj = J(Float64.(r.traj.kdataNodes), r.traj.name)
    r = ReconParams(r.filename, r.pathProc, deepcopy(r.scanParameters), deepcopy(r.reconParameters), 
        data, traj, deepcopy(r.performedMethods))
    r.reconParameters[:sensMaps] = ComplexF64.(r.reconParameters[:sensMaps])
    append!(r.performedMethods, [nameof(var"#self#")])
    return r
end