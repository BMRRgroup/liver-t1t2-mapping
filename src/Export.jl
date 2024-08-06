export exportRawData, saveasImDataParams

function get_file(r::ReconParams, path, name, tag)
    filename = r.filename
    fsplit = split(basename(filename), '_')
    datestr = string(fsplit[2][5:8], fsplit[2][3:4], fsplit[2][1:2])
    timestr = fsplit[3][1:end-1]
    SeriesNumber = string(fsplit[4], '0', fsplit[5])
    fileID = string(datestr, '_', timestr, '_', SeriesNumber)
    if path == ""
        path = r.pathProc
    end
    # Create hdf5 fileID
    if name == ""
        newFilename = joinpath(path, string(fileID, "_", tag, ".h5"))
    else
        newFilename = joinpath(path, string(fileID, "_", tag, "_", name,".h5"))
    end
    fid = h5open(newFilename, "w")
    return fileID, fid, newFilename
end

function saveasImDataParams(r::ReconParams{ImgData{T}, <:AbstractTrajectory}; path::String="", name::String="", saveMotionState::Int=0) where T
    @debug("Save as ImDataParams.")
    fileID, fid, newFilename = get_file(r, path, name, "ImDataParamsBMRR")

    img = r.data.imgData
    if haskey(r.reconParameters, :iSlice)
        iSlice = r.reconParameters[:iSlice]
        img = img[:,:,iSlice:iSlice,:,:]
    end
    if r.scanParameters[:AcqMode] == "Cartesian"
        # only if Cartesian sampling 
        img = reverse(img, dims=(2,3))
    elseif r.traj.name == :StackOfStars
        img = reverse(img, dims=(3))
    end

    g = create_group(fid, "ImDataParams")
    nEchoes = length(r.scanParameters[:TE]) #parse(Int, searchGoalCparams(r, "EX_ACQ_echoes"))
    nInter = 1 #r.scanParameters[:numInterleaves] TODO
    HDF5.attributes(g)["voxelSize_mm"] = vec(r.scanParameters[:FOV]' ./ [size(img,i) for i=1:3])
    HDF5.attributes(g)["fieldStrength_T"] = r.scanParameters[:FieldStrength] #parse(Float32, searchGoalCparams(r, "VW_main_magnetic_field"))
    HDF5.attributes(g)["centerFreq_Hz"] = r.scanParameters[:centerFreq_Hz] #parse(Float32, searchGoalCparams(r, "HW_resonance_freq"))
    HDF5.attributes(g)["TE_s"] = r.scanParameters[:TE_s]
    HDF5.attributes(g)["fileID"] = fileID  

    # Change dimension such that they match python toolbox
    s = size(img)
    if saveMotionState == 0
        img = reshape(img, s[1], s[2], s[3], nEchoes*nInter, :)
    else
        if r.reconParameters[:motionStatesRecon] == "all"
            numMotionStates = r.reconParameters[:motionGatingParams][:numClusters]
        else
            numMotionStates = parse(Int64, r.reconParameters[:motionStatesRecon])
        end
        img = reshape(img, s[1], s[2], s[3], nEchoes*nInter, :, numMotionStates)
        img = img[:,:,:,:,:,saveMotionState]
    end
    img = permutedims(img, [4,5,3,2,1])
    g["signal", deflate=3] = img
    
    process_relax_params!(r, fid)
    process_motion_params!(r, fid)

    close(fid)
    return newFilename
end

function process_motion_params!(r, fid)
    if haskey(r.reconParameters, :motionStatesCenter)
        g = create_group(fid, "MotionParams")
        g["motionCurve", deflate=3] = r.reconParameters[:motionCurve] 
        g["motionStates", deflate=3] = r.reconParameters[:motionStates] 
        g["motionStatesCenter"] = r.reconParameters[:motionStatesCenter] 
        HDF5.attributes(g)["method"] = r.reconParameters[:motionMethod]
    end
end

function process_relax_params!(r, fid)
    if r.scanParameters[:isFINO]
        g = create_group(fid, "RelaxParams")
        HDF5.attributes(g)["TE_s"] = r.scanParameters[:FINO_TE_s]
        HDF5.attributes(g)["delay_s"] = r.scanParameters[:FINO_delay_s]
        HDF5.attributes(g)["angle_deg"] = r.scanParameters[:FINO_angle_deg]
        HDF5.attributes(g)["startup_delay_s"] = r.scanParameters[:FINO_startup_delay_s]
    end
end