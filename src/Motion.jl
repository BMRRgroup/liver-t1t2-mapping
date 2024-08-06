export motionStatesCamera!, motionStatesSelfGating!

function motionStatesCamera!(r::ReconParams{KdataRaw{T}, <:AbstractTrajectory}) where T<:AbstractFloat
    params = r.reconParameters[:motionGatingParams]
    motionStatesSelfGating!(r, params[:maskLabel], params[:numClusters], 
        params[:doPlotBcurve], params[:method])
end

# needs to be performed after SortData
function motionStatesSelfGating!(r::ReconParams{KdataPreprocessed{T}, <:NonCartesian3D}) where T<:AbstractFloat
    params = r.reconParameters[:motionGatingParams]
    motionStatesSelfGating!(r, params[:eddyCurrentCorrection], params[:numClusters], 
        params[:doPlotBcurve], params[:method])
end

function motionStatesSelfGating!(r::ReconParams{KdataPreprocessed{T}, <:NonCartesian3D}, 
        eddyCurrentCorrection::Bool, numClusters::Int, doPlotBcurve::Bool, method::Symbol) where T<:AbstractFloat
    @debug("Estimate motion states from data.")
    data = getACregion(r, eddyCurrentCorrection=eddyCurrentCorrection)

    data = reshape(data, :, numKy(r), numDyn(r))

    # Scale the k-space data 
    data = vcat(abs.(data), angle.(data))
    for i in eachindex(axes(data, 3))
        for j in eachindex(axes(data, 1))
            #data[i,:] = data[i,:] .- mean(data[i,:])
            #data[i,:] = (data[i,:] .- minimum(data[i,:])) ./ (maximum(data[i,:])-minimum(data[i,:]))
            data[j,:,i] = data[j,:,i] ./ (maximum(data[j,:,i])-minimum(data[j,:,i]))
            #data[j,:,i] = data[j,:,i] ./ (percentile(data[j,:,i], 95)-percentile(data[j,:,i], 5))
            data[j,:,i] = data[j,:,i] .- median(data[j,:,i])
        end
    end

    data = reshape(data, :, numKy(r)*numDyn(r))

    M = fit(PCA, transpose(data), maxoutdim=2)

    # Choose principal component with main frequency in the range 0.1 Hz - 0.5 Hz
    timePerProfile = r.scanParameters[:shotDuration_ms]
    # timeAcq = collect((1:numKy(r)*numDyn(r)) .* timePerProfile)
    freqHz = fftfreq(numKy(r)*numDyn(r), 1/timePerProfile)
    components = projection(M::PCA)
    fftComponents = abs.(fft(components, 1))[(abs.(freqHz).>0.1) .& (abs.(freqHz).<0.5),:]
    bCurveKz0 = components[:, argmax(fftComponents)[2]]

    # Rescale per dynamic
    bCurveKz0dyn = reshape(bCurveKz0, numKy(r), numDyn(r))
    for i in 1:numDyn(r)
        bCurveKz0dyn[:,i] = (bCurveKz0dyn[:,i] .- quantile(bCurveKz0dyn[:,i], 0.05)) ./ (quantile(bCurveKz0dyn[:,i], 0.95) - quantile(bCurveKz0dyn[:,i], 0.05))
        # if mean(bCurveKz0dyn[:,i]) > 0.5 # Assume that first dynamic is the most robust 
        #     @info("Flipping bCurve dyn " * string(i))
        #     bCurveKz0dyn[:,i] = -bCurveKz0dyn[:,i] .+ 1
        # end
        sum1 = sum(bCurveKz0dyn[:,i] .> 0.5 .&& bCurveKz0dyn[:,i] .< 1.1)
        sum2 = sum(bCurveKz0dyn[:,i] .< 0.5 .&& bCurveKz0dyn[:,i] .> -0.1)
        if sum1 > sum2
            @info("Flipping bCurve dyn " * string(i) * ": " * string(sum1) * " > " * string(sum2))
            bCurveKz0dyn[:,i] = -bCurveKz0dyn[:,i] .+ 1
            # bCurveKz0 = reshape(bCurveKz0dyn, numKy(r)*numDyn(r))
        end
    end
    bCurveKz0 = reshape(bCurveKz0dyn, numKy(r)*numDyn(r))

    # Assume subject is more frequently in expiration state
    # if mean(bCurveKz0[:,1]) > 0.5 # Assume that first dynamic is the most robust 
    #     bCurveKz0 = reshape(bCurveKz0, numKy(r)*numDyn(r))
    #     bCurveKz0 = -bCurveKz0 .+ 1
    # end
    # bCurve = Vector(reshape(transpose(repeat(bCurveKz0, 1, numKz(r))), length(bCurveKz0)*numKz(r)))

    assignments, medoidsCenter = motionStatesClusteringStackOfStars(bCurveKz0, bCurveKz0, 
        numClusters=numClusters, method=method, numKz=numKz(r))
    r.reconParameters[:motionCurve] = bCurveKz0
    r.reconParameters[:motionStates] = assignments
    r.reconParameters[:motionStatesCenter] = medoidsCenter
    r.reconParameters[:motionMethod] = "SelfGating_"*String(method)
    if doPlotBcurve
        plotBcurve(bCurveKz0, joinpath(r.pathProc, basename(r.filename)), assignments=assignments, numClusters=numClusters)
    end
    append!(r.performedMethods, [nameof(var"#self#")])
end

function motionStatesClusteringStackOfStars(bCurveKz0::Array{T, 1}, bCurve::Array{T, 1}; 
        numClusters::Int=5, method::Symbol=:relDisplacement, 
        numKz::Int) where T
    dist = pairwise(Euclidean(), bCurveKz0)
    if method == :kmedoids
        # Perform clustering
        k = kmedoids(dist, numClusters)
        medoidsCenter = k.medoids[sortperm(bCurveKz0[k.medoids])]
        # Assign intra-shot profiles
        dist2 = pairwise(Euclidean(), bCurve, bCurveKz0[medoidsCenter])
        assignments = argmin.(eachrow(dist2))
    elseif method == :relDisplacement
        cluster_range = 1/numClusters # motion curve should be within this interval
        medoidsCenter = collect((cluster_range/2):cluster_range:(1-cluster_range/2)) # motion curve should be sorted along these interval centers
        dist2 = pairwise(Euclidean(), bCurve, medoidsCenter)
        assignments = argmin.(eachrow(dist2))
    elseif method == :equal_spokes_no
        no_spokes = length(bCurveKz0) # spoke groups per Kz0 
        no_spokes_per_cluster = round(Int,(no_spokes / numClusters))
        bcurveKz0_sorted_idx = sortperm(bCurveKz0)
        # bcurve_sorted_idx = Vector(reshape(transpose(repeat(bcurveKz0_sorted_idx, 1, numKz)), length(bcurveKz0_sorted_idx)*numKz))
        assignments_kz0 = Array{Int16}(undef,no_spokes,1)
        medoidsCenter = Array{Float64}(undef,numClusters,1)

        if numClusters == 1
            assignments_kz0[:] .= 1
        else
            for (class, idx) in enumerate(collect(1:no_spokes_per_cluster:no_spokes)[1:end-1])
                idx_end = min((idx + no_spokes_per_cluster), no_spokes)
                for j in collect(bcurveKz0_sorted_idx[idx:idx_end])
                    assignments_kz0[j,1] = class
                end
                medoidsCenter[class, 1] = mean(hcat([bCurve[i] for i in collect(bcurveKz0_sorted_idx[idx:idx_end])]))
            end
        end
        # for (class, idx) in enumerate(collect(1:no_spokes_per_cluster:no_spokes)[1:end-1])
        #     idx_end = min((idx + no_spokes_per_cluster), no_spokes)
        #     for j in collect(bcurveKz0_sorted_idx[idx:idx_end])
        #         assignments_kz0[j,1] = class
        #     end
        #     medoidsCenter[class, 1] = mean(hcat([bCurve[i] for i in collect(bcurveKz0_sorted_idx[idx:idx_end])]))
        # end

        ## unfold kz0 spokes to all spokes
        assignments = Vector(reshape(transpose(repeat(assignments_kz0, 1, numKz)), length(assignments_kz0)*numKz))
        
        ## test:
        # using StatsBase
        # countmap(assignments)
    end
    return assignments, medoidsCenter
end

function plotBcurve(bCurve::Vector{T}, filename::String="."; assignments::Vector{Int}=Vector{Int}(), 
        numClusters::Int=0) where T
    plot(bCurve, label="motion curve")
    if numClusters > 0
        for i in 1:numClusters
            plot!((0:length(bCurve)-1)[assignments.==i], 
                bCurve[assignments.==i], seriestype=:scatter, label="state "*string(i))
        end
    end
    # plot!(legend=:outerbottom, legendcolumns=3)
    xlabel!("shot number")
    ylabel!("relative displacement (a.u.)")
    # ylims!(minimum(bCurve),maximum(bCurve)) 
    # xlims!(0,100)
    splittedFilename = split(filename, ".")
    savefig(join(splittedFilename[1:end-1], ".") * "_motionCurve.png")
end