export coilSensitivities!

function coilSensitivities!(r::ReconParams{KdataPreprocessed{T}, <:NonCartesian3D}) where T<:AbstractFloat
    if r.reconParameters[:coilSensitivities] == :ESPIRiT
        coilSensitivitiesESPIRiT!(r)
    elseif r.reconParameters[:coilSensitivities] == :ESPIRiT_slicewise
        coilSensitivitiesESPIRiT_slicewise!(r)
    end
end

function initCoilSensitivitiesESPIRiT!(r::ReconParams{KdataPreprocessed{T}, <:NonCartesian3D}) where T<:AbstractFloat
    @debug("Compute ESPIRiT coil sensitivities.")
    encodingSize = deepcopy(r.scanParameters[:encodingSize])
    encodingSize[1] = round(Int, encodingSize[1] * r.scanParameters[:KxOversampling][1])
    encodingSize[2] = round(Int, encodingSize[2] * r.scanParameters[:KxOversampling][1])
    @assert r.traj.name == :StackOfStars
    @assert size(r.traj.kdataNodes,1) == 2
    traj = reshape(r.traj.kdataNodes[:,:,:,1,1], 2, :)
    data = r.data.kdata[:,:,:,1,1,:]
    return data, traj, encodingSize
end
 
function coilSensitivitiesESPIRiT_slicewise!(r::ReconParams{KdataPreprocessed{T}, <:NonCartesian3D}; ncalib::Int=16, verbose::Bool=true) where T<:AbstractFloat
    data, traj, encodingSize = initCoilSensitivitiesESPIRiT!(r)
    p = verboseProgress(numKz(r), "Slice-wise ESPIRiT sensitivities... ", verbose)
    # Choose first dyn (max. contrast for our sequence) and first echo
    data = ifftshift(ifft(ifftshift(data, (3)), 3), (3))

    # Perform slice-wise ESPIRiT
    NFFT = MRIOperators.NFFTOp(Tuple(encodingSize[1:2]), traj, cuda=false)
    weights = sqrt.(sdc(NFFT.plan, iters=10))
    coilsensEspirit = zeros(Complex{T}, encodingSize[1], encodingSize[2], numKz(r), numChan(r), 1)
    let coilsensEspirit = coilsensEspirit, data = data, traj = traj
        #@floop @inbounds 
        for i = eachindex(axes(data, 3))
            local NFFT = NFFTOp(Tuple(encodingSize[1:2]), traj, cuda=r.reconParameters[:cuda])
            local FFT = FFTOp(Complex{T}, Tuple(encodingSize[1:2]))
            local W = WeightingOp(weights)
            kdata = reshape(data[:,:,i,:], :, numChan(r))
            kdataGrid = Array{Complex{T}}(undef, encodingSize[1], encodingSize[2], size(data,4))
            cgnr_iter = 2
            solver = RegularizedLeastSquares.CGNR(W*NFFT, iterations=cgnr_iter)
            for j in 1:numChan(r)
                # kdataGrid[:,:,j] = FFT * (adjoint(NFFT) * (kdata[:,j] .* weights.^2))
                kdata[:,j] .= kdata[:,j] .* weights
                img = solve(solver, kdata[:,j])
                kdataGrid[:,:,j] = FFT * img
            end
            cropKdata = MRICoilSensitivities.crop(kdataGrid, (ncalib, ncalib, size(data,4)))
            coilsensEspirit[:,:,i,:,:] = espirit(cropKdata, Tuple(encodingSize[1:2]))
            verboseNext!(p)
        end
    end
    coilsensEspirit = coilsensEspirit[:,:,:,:,1]

    # Simple correction for phase shifts between slices
    for i in 1:numChan(r)
        for j in eachindex(axes(data, 3))
            if j > 1
                diff1 = sum(abs.(coilsensEspirit[:,:,j,i]-coilsensEspirit[:,:,j-1,i]))
                diff2 = sum(abs.(coilsensEspirit[:,:,j,i]+coilsensEspirit[:,:,j-1,i]))
                if diff1 > diff2 coilsensEspirit[:,:,j,i] ./= -1 end
            end
        end
    end 
    r.reconParameters[:sensMaps] = Complex{T}.(coilsensEspirit)
    append!(r.performedMethods, [nameof(var"#self#")])
end

function coilSensitivitiesESPIRiT!(r::ReconParams{KdataPreprocessed{T}, <:NonCartesian3D}; ncalib::Int=24) where T<:AbstractFloat
    data, traj, encodingSize = initCoilSensitivitiesESPIRiT!(r)
    shape = (encodingSize[1], encodingSize[2], numKz(r))

    # Perform slice-wise ESPIRiT
    NFFT = MRIOperators.NFFTOp(Tuple(encodingSize[1:2]), traj, cuda=false)
    weights = Complex{T}.(sqrt.(sdc(NFFT.plan, iters=10)))
    weights .= weights .* T.(1/sqrt(numKz(r)))
    weights = repeat(weights[:], numKz(r))

    cgnr_iter = 2
    NFFT = StackOfStarsOp(shape, traj, cuda=r.reconParameters[:cuda])
    FFT = FFTOp(ComplexF32, shape)
    W = WeightingOp(weights)
    solver = RegularizedLeastSquares.CGNR(W*NFFT, iterations=cgnr_iter)

    kdata = Complex{T}.(reshape(data, :, numChan(r)))
    kdataGrid = Array{ComplexF32}(undef, encodingSize[1], encodingSize[2], numKz(r), numChan(r))
    for j in 1:numChan(r)
        kdata[:,j] .= kdata[:,j] .* weights
        img = solve(solver, kdata[:,j])
        kdataGrid[:,:,:,j] = FFT * img
    end
    cropKdata = MRICoilSensitivities.crop(kdataGrid, (ncalib, ncalib, ncalib, size(data,4)))

    coilsensEspirit = espirit(cropKdata, shape, eigThresh_2=0.0)

    # coilsensEspirit[abs.(r.reconParameters[:sensMaps]) .== 0] .= 0
    r.reconParameters[:sensMaps] = Complex{T}.(coilsensEspirit[:,:,:,:,1]) #_highRes
    append!(r.performedMethods, [nameof(var"#self#")])
end