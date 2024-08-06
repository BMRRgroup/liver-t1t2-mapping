export iterativeRecon_perEchoDyn, sumOfSquaresRecon!

function getBcurve(r::ReconParams{KdataPreprocessed{T}, <:AbstractTrajectory}) where T<:AbstractFloat
    if r.reconParameters[:motionGating] != false
        if r.reconParameters[:motionStatesRecon] == "all"
            numMotionStates = r.reconParameters[:motionGatingParams][:numClusters]
        else
            numMotionStates = parse(Int64, r.reconParameters[:motionStatesRecon])
        end
        bCurve = r.reconParameters[:motionStates]
    else
        numMotionStates = 1
        bCurve = ones(numKy(r)*numDyn(r))
    end
    return numMotionStates, bCurve
end

function getRegularization(r::ReconParams{KdataPreprocessed{T}, <:AbstractTrajectory}, reconSize::Tuple, 
        numEchoesNumDyn::Int, numMotionStates::Int, numCoils::Int) where T<:AbstractFloat
    reg = Vector{Regularization}()
    if r.reconParameters[:iterativeReconParams][:Regularization][:L1Wavelet_spatial] > 0
        push!(reg, Regularization("L1", T.(r.reconParameters[:iterativeReconParams][:Regularization][:L1Wavelet_spatial]))) #params[:Regularization][:L1Wavelet_spatial]; params))
        sparseTrafo = MRIOperators.SparseOp(Complex{T}, "Wavelet", (reconSize[1], reconSize[2], reconSize[3]))
        reg[1].params[:sparseTrafo] = DiagOp( repeat([sparseTrafo],Int.(numEchoesNumDyn*numMotionStates*numCoils))... )
    end
    if r.reconParameters[:iterativeReconParams][:Regularization][:TV_spatial] > 0
        push!(reg, Regularization("TV", T.(r.reconParameters[:iterativeReconParams][:Regularization][:TV_spatial]), 
            shape=(reconSize[1], reconSize[2], reconSize[3], numEchoesNumDyn*numMotionStates*numCoils), 
            dims=[1,2,3]))
    end
    if r.reconParameters[:iterativeReconParams][:Regularization][:TV_spatialTemporal] > 0
        push!(reg, Regularization("TV", T.(r.reconParameters[:iterativeReconParams][:Regularization][:TV_spatialTemporal]), 
            shape=(reconSize[1], reconSize[2], reconSize[3], numEchoesNumDyn, numMotionStates, numCoils), 
            dims=[1,2,3,5]))
    end
    if r.reconParameters[:iterativeReconParams][:Regularization][:TV_temporal] > 0 && numMotionStates > 1
        push!(reg, Regularization("TV", T.(r.reconParameters[:iterativeReconParams][:Regularization][:TV_temporal]), 
            shape=(reconSize[1], reconSize[2], reconSize[3], numEchoesNumDyn, numMotionStates, numCoils), 
            dims=5))
    end
    if r.reconParameters[:iterativeReconParams][:Regularization][:LLR] > 0
        push!(reg, Regularization("LLR", T.(r.reconParameters[:iterativeReconParams][:Regularization][:LLR]), 
            shape=(reconSize[1], reconSize[2], reconSize[3], numEchoesNumDyn*numMotionStates*numCoils), 
            blockSize=(2,2,2,2)))
    end
    return reg
end

function set_regTrafo(r::ReconParams{KdataPreprocessed{T}, <:AbstractTrajectory}, numMotionStates::Int, numCoils::Int, numReg::Int, coilWise::Bool) where T<:AbstractFloat
    if !coilWise
        trafo = Diagonal(repeat(reshape(r.reconParameters[:sensMaps][:,:,:,1].!=0, :), numMotionStates))
    else
        trafo = Diagonal(repeat(reshape(r.reconParameters[:sensMaps][:,:,:,1].!=0, :), numMotionStates*numCoils))
    end
    return fill(trafo, numReg)
end

function initializeVariables(r::ReconParams{KdataPreprocessed{T}, <:AbstractTrajectory}; coilWise::Bool=false) where T<:AbstractFloat
    data = r.data.kdata
    if typeof(r.traj) == Cartesian3D
        traj = nothing
        reconSize = size(r.data.kdata)[1:3]
    else
        traj = r.traj.kdataNodes
        reconSize = deepcopy(r.scanParameters[:encodingSize])
    end

    num_kx, num_chan, num_echoes, num_kz, num_ky, num_dyn = numKx(r), numChan(r), numEchoes(r), numKz(r), numKy(r), numDyn(r)

    if r.traj.name == :FLORET
        data = reshape(data, num_kx, :, 1, 1, 1, num_chan, 1)
        traj = reshape(traj, 3, num_kx, :, 1, 1)
        num_ky *= num_kz
        num_kz = 1
    elseif r.traj.name == :StackOfStars
        reconSize[1] = round(Int, reconSize[1] * r.scanParameters[:KxOversampling][1])
        reconSize[2] = round(Int, reconSize[2] * r.scanParameters[:KxOversampling][1])
        reconSize[3] = num_kz
    end

    return data, traj, reconSize, num_kx, num_chan, num_echoes, num_kz, num_ky, num_dyn
end

function computeWeightsForSamplingDensity(r::ReconParams{KdataPreprocessed{T}, <:AbstractTrajectory}, maskMotion::Array, numContr::Int, trajTemp::Array, reconSize::Tuple, 
        num_kx::Int, num_ky::Int, num_kz::Int, num_chan::Int) where T<:AbstractFloat
    # Compute weights
    weights = similar(r.data.kdata, (num_kx*num_ky, numContr))
    for i = 1:numContr
        if :sdc in keys(r.reconParameters)
            weights[maskMotion[:,1,i,1],i] = sqrt.(reshape(r.reconParameters[:sdc], :))
        else
            if r.traj.name == :StackOfStars
                weights[maskMotion[:,1,i,1],i] = sqrt.(sdc(ReconBMRR.NFFTOp(Tuple(reconSize[1:2]), trajTemp[i], 
                    cuda=false).plan, iters=10)) # hardcoded for the first dynamic!
            else
                weights[maskMotion[:,1,i,1],i] = sqrt.(sdc(ReconBMRR.NFFTOp(reconSize, trajTemp[i], 
                    cuda=false).plan, iters=10)) # hardcoded for the first dynamic!
            end
        end
    end
    if r.traj.name == :StackOfStars
        weights .= weights .* Float32.(1/sqrt(num_kz))
    end
    weights = repeat(weights[:], num_chan*num_kz)
    weights = permutedims(reshape(weights, num_kx*num_ky, numContr, num_chan, num_kz), [1,4,2,3])[:]
    weightsMasked = weights[maskMotion[:]]
    return weightsMasked
end

function constructOperators(r::ReconParams{KdataPreprocessed{T}, <:AbstractTrajectory}, 
    weightsMasked::Array, numContr::Int, reconSize::Tuple, coilWise::Bool, num_chan::Int; trajTemp::Array=[]) where T<:AbstractFloat
    csm = rescaleSensMaps(r.reconParameters[:sensMaps])

    # Contruct operators
    if typeof(r.traj) == Cartesian3D
        ft = [ReconBMRR.FFTOp(Complex{T}, reconSize; unitary=false) for j=1:numContr]
    else
        if r.traj.name == :StackOfStars
            ft = [StackOfStarsOp(reconSize, trajTemp[j], cuda=r.reconParameters[:cuda]) for j=1:numContr]
        else
            ft = [ReconBMRR.NFFTOp(reconSize, trajTemp[j], cuda=r.reconParameters[:cuda]) for j=1:numContr]
        end
    end
    E = MRIOperators.CompositeOp(WeightingOp(weightsMasked,1), 
        DiagOp(repeat(ft, outer=num_chan)...), isWeighting=true)
    if coilWise
        Efull = E
    else
        Efull = MRIOperators.CompositeOp(E, 
            MRIOperators.SensitivityOp(reshape(csm, prod(reconSize), num_chan), 
            numContr))
        
        # # intensityCorrection
        # W = Complex{T}.(1 ./ (reshape((sum(abs.(csm), dims=4)[:,:,:,1]).^2,:) .+ eps()))
        # W[abs.(reshape(csm[:,:,:,1],:)) .== 0] .= 0
        # Efull = MRIOperators.CompositeOp(Efull, WeightingOp(W,numContr), isWeighting=false)
    end
    return Efull
end

function rescaleSensMaps(csm::Array{Complex{T}, 4}) where T<:AbstractFloat
    # Compute the sum of squares of magnitudes across coils
    sos = sqrt.(sum(abs.(csm).^2, dims=4))

    # Magnitude normalization
    normalized_array = similar(csm)
    normalized_array .= csm ./ (sos .+ eps())
    normalized_array[abs.(csm) .== 0] .= 0
    return normalized_array
end

function iterativeRecon_perEchoDyn(r::ReconParams{KdataPreprocessed{T}, <:AbstractTrajectory}; coilWise::Bool=false, verbose::Bool=true) where T<:AbstractFloat
    @info("Iterative reconstruction.")
    data, traj, reconSize, num_kx, num_chan, num_echoes, num_kz, num_ky, num_dyn = initializeVariables(r, coilWise=coilWise)
    
    numMotionStates, bCurve = getBcurve(r)
    numContr = num_echoes * num_dyn * numMotionStates 
    params = r.reconParameters[:iterativeReconParams] # Recon paramters
    reconSize = Tuple(reconSize)

    if coilWise
        reg = getRegularization(r, reconSize, 1, numMotionStates, num_chan)
        img = zeros(Complex{T}, reconSize[1], reconSize[2], reconSize[3], num_echoes, num_dyn, numMotionStates, num_chan)
    else
        reg = getRegularization(r, reconSize, 1, numMotionStates, 1)
        img = zeros(Complex{T}, reconSize[1], reconSize[2], reconSize[3], num_echoes, num_dyn, numMotionStates, 1)
    end

    bCurve = reshape(bCurve, :, num_dyn)
    p = verboseProgress(num_echoes*num_dyn, "Perform reconstruction per echo and dynamic... ", verbose)

    # combinations = [(i, j) for i in 1:1 for j in [4]]
    combinations = [(i, j) for i in 1:num_echoes for j in 1:num_dyn]

    #Threads.@threads 
    for index in 1:length(combinations)
        let reg = reg, r = r
            i, j = combinations[index]
            dataTemp = zeros(Complex{T}, num_kx, num_ky, num_kz, numMotionStates, num_chan)
            trajTemp = zeros(T, size(r.traj.kdataNodes,1), num_kx, num_ky, numMotionStates)
            maskMotion = zeros(Bool, num_kx, num_ky)
            for k in 1:numMotionStates
                idx = bCurve[:,j] .== k
                dataTemp[:,idx,:,k,:] .= data[:,idx,:,i,j,:,:]
                trajTemp[:,:,idx,k] .= traj[:,:,idx,i,j]
                maskMotion[:,idx] .= 1
            end
            maskMotion = repeat(maskMotion[:], num_kz*numMotionStates*num_chan)
            dataTemp = reshape(dataTemp, num_kx*num_ky, num_kz, numMotionStates, num_chan)
            trajTemp = reshape(trajTemp, size(r.traj.kdataNodes,1), num_kx*num_ky, numMotionStates)
            maskMotion = reshape(maskMotion, num_kx*num_ky, num_kz, numMotionStates, num_chan)
            
            dataTemp = dataTemp[maskMotion] 
            trajTemp = [trajTemp[:,maskMotion[:,1,k,1] .== 1,k] for k=1:numMotionStates]
            samplingMask = (dataTemp .== 0)

            weightsMasked = computeWeightsForSamplingDensity(r, maskMotion, numMotionStates, trajTemp, 
                reconSize, num_kx, num_ky, num_kz, num_chan)
            weightsMasked[samplingMask] .= 0
            dataTemp .= dataTemp .* weightsMasked
            @show size(weightsMasked)
            
            Efull = constructOperators(r, weightsMasked, numMotionStates, reconSize, coilWise, num_chan, trajTemp=trajTemp)
            # Efull = constructOperators(r, weightsMasked, numMotionStates, reconSize, coilWise, num_chan, trajTemp)
            if params[:solver] == "admm"
                params[:ρ] = 1e-1*ones(length(reg))
                params[:regTrafo] = set_regTrafo(r, numMotionStates, num_chan, length(reg), coilWise)
            else
                if haskey(params, :regTrafo)
                    delete!(params, :regTrafo)
                end
            end

            solver = createLinearSolver(params[:solver], Efull; reg=reg, params...)
            if params[:verboseIteration]
                solver.verbose = true
            end
            
            imgTmp = solve(solver, dataTemp) 
            img[:,:,:,i,j,:,:] = reshape(imgTmp, reconSize[1], reconSize[2], reconSize[3], numMotionStates, :)
            verboseNext!(p)
        end
    end
    append!(r.performedMethods, [nameof(var"#self#")])
    return ReconParams(r.filename, r.pathProc, r.scanParameters, r.reconParameters, 
        ImgData(img), r.traj, r.performedMethods)
end

function sumOfSquaresRecon!(r::ReconParams{ImgData{T}, <:AbstractTrajectory}) where T<:AbstractFloat
    @debug("Coil combination using sum of squares reconstruction.")
    img = r.data.imgData
    numContr = size(img, 7)
    ssq = zeros(eltype(img), size(img, 1), size(img, 2), size(img, 3), size(img, 4), 
        size(img, 5), size(img, 6))
    for i in 1:numContr
        ssq += abs2.(img[:, :, :, :, :, :, i])
    end
    ssq = sqrt.(ssq)
    r.data = ImgData(ssq)
end