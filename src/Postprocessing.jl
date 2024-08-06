export upsampleRecVoxelSize!, removeOversampling! 

function removeOversampling!(r::ReconParams{ImgData{T}, <:AbstractTrajectory}) where T<:AbstractFloat
    @debug("Remove oversampling.")
    img = removeOversampling(r.data.imgData, r.scanParameters[:encodingSize])
    r.data = ImgData(img)
    append!(r.performedMethods, [nameof(var"#self#")])
end

function removeOversampling(img, encodingSize) 
    # Crop oversampling
    center = div.(size(img)[1:3], 2)
    # cropInd_x = floor(Int64, 1/2*(size(img, 1)-encodingSize[1]))
    # cropInd_y = floor(Int64, 1/2*(size(img, 2)-encodingSize[2]))
    # cropInd_z = floor(Int64, 1/2*(size(img, 3)-encodingSize[3]))
    # return Array(selectdim(selectdim(selectdim(img, 1, cropInd_x+1:size(img, 1)-cropInd_x), 
    #                            2, cropInd_y+1:size(img, 2)-cropInd_y), 
    #                 3, cropInd_z+1:size(img, 3)-cropInd_z))
    return Array(selectdim(selectdim(selectdim(img, 1, center[1]-div(encodingSize[1], 2)+1:center[1]-div(encodingSize[1],2)+encodingSize[1]), 
                                     2, center[2]-div(encodingSize[2], 2)+1:center[2]-div(encodingSize[2], 2)+encodingSize[2]), 
                           3, center[3]-div(encodingSize[3], 2)+1:center[3]-div(encodingSize[3], 2)+encodingSize[3]))
end

function upsampleRecVoxelSize!(r::ReconParams{ImgData{T}, <:AbstractTrajectory}) where T<:AbstractFloat
    ## TODO: do this step before removeOversampling
    @debug("Upsample to reconstruction voxel size.")
    img = r.data.imgData
    newsize = T.(collect(size(img)))
    # acqVoxelSize = vec(r.scanParameters[:FOV]' ./ [size(img,i) for i=1:3])
    acqVoxelSize = r.scanParameters[:AcqVoxelSize]
    recVoxelSize = r.scanParameters[:RecVoxelSize]
    fac = acqVoxelSize' ./ recVoxelSize
    newsize[1] *= fac[1]
    newsize[2] *= fac[2]
    newsize[3] *= fac[3]
    encodingSize = round.(Int, r.scanParameters[:FOV] ./ r.scanParameters[:RecVoxelSize])
    r.scanParameters[:encodingSize] = encodingSize
    newsize = round.(Int, newsize)
    img = imresize_dim(img, Tuple(newsize), dims=[1,2,3]) 
    # img = FourierTools.resample(img, Tuple(newsize))
    r.data = ImgData(img)
    append!(r.performedMethods, [nameof(var"#self#")])
end
