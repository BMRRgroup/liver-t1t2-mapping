function verboseProgress(n::Integer, text::String, active::Bool)
    if active
        return Progress(n, dt=1, desc=text, color=:white, barglyphs=BarGlyphs("[=> ]"), barlen=50)
    end
end

function verboseNext!(p::Progress)
    next!(p)
end

function verboseNext!(p::Nothing)
end

function strDict2symDict(dict::Dict{String,Any})
    d = Dict{Symbol, Any}()
    for (k,v) in dict
        d[Symbol(k)] = v
    end
    return d
end

function hamming(arr::AbstractArray{Complex{T}}; strength::Float64=0.25, dims::AbstractVector{Int}=collect(1:3)) where {T<:AbstractFloat}
    # Hamming filter
    a = (1-strength)*0.46 + 0.54
    
    function apply_hamming(slice::AbstractArray{Complex{T}}) where {T<:AbstractFloat}
        @assert length(size(slice)) == 1
        x = Vector(LinRange(-0.5, 0.5, length(slice)))
        return slice .* T.(a .+ (1-a)*cos.(2*pi*x))
    end
    
    for dim in dims
        arr = mapslices(apply_hamming, arr, dims=dim)
    end
    return arr
end

function zeroFill(arr::AbstractArray{Complex{T}}, newShape::AbstractVector{Int}; dims::Vector{Int}=[1,2,3]) where {T<:AbstractFloat}
# function zeroFill(arr::AbstractArray, newShape::AbstractVector{Int}; dims::Vector{Int}=[1,2,3]) where {T<:AbstractFloat, N<:Integer}
    padsize1 = zeros(Int, length(size(arr)))
    padsize2 = zeros(Int, length(size(arr)))
    for i in dims
        padsize1[i] = floor(Int, (newShape[i] - size(arr,i)) / 2)
        padsize2[i] = newShape[i] - padsize1[i] - size(arr,i)
    end
    arr = padarray(arr, Fill(0, padsize1, padsize2))
    return parent(arr)
end

function imresize_dim(arr::AbstractArray, newShape::Tuple; method=Lanczos(4), dims::Vector{Int}=[1,2,3])
    return mapslices(x -> imresize(x, newShape[dims], method=method), arr, dims=dims)
end