export StackOfStarsOp
import Base.adjoint

mutable struct StackOfStarsOp{T} <: AbstractLinearOperator{T}
  nrow :: Int
  ncol :: Int
  symmetric :: Bool
  hermitian :: Bool
  prod! :: Function
  tprod! :: Nothing
  ctprod! :: Function
  nprod :: Int
  ntprod :: Int
  nctprod :: Int
  args5 :: Bool
  use_prod5! :: Bool
  allocated5 :: Bool
  Mv5 :: Vector{T}
  Mtu5 :: Vector{T}
  plan
  plan_z
  iplan_z
end

LinearOperators.storage_type(op::StackOfStarsOp) = typeof(op.Mv5)

"""
    NFFTOp(shape::Tuple, tr::Trajectory; kargs...)
    NFFTOp(shape::Tuple, tr::AbstractMatrix; kargs...)

generates a `NFFTOp` which evaluates the MRI Fourier signal encoding operator using the NFFT.

# Arguments:
* `shape::NTuple{D,Int64}`  - size of image to encode/reconstruct
* `tr`                      - Either a `Trajectory` object, or a `ND x Nsamples` matrix for an ND-dimenensional (e.g. 2D or 3D) NFFT with `Nsamples` k-space samples
* (`nodes=nothing`)         - Array containg the trajectory nodes (redundant)
* (`kargs`)                 - additional keyword arguments
"""
function StackOfStarsOp(shape::Tuple, tr::AbstractMatrix{T}; cuda::Bool=true, oversamplingFactor=2, kernelSize=3, kargs...) where {T}
  if cuda
    plan = plan_nfft(CuArray, tr, shape[1:2], m=kernelSize, σ=oversamplingFactor)
    tmpVec = zeros(Complex{T}, size(tr,2),shape[3]) |> CuArray
    plan_z = plan_fft!(tmpVec, 2)
    iplan_z = plan_bfft!(tmpVec, 2)
    # Pre-allocate GPU arrays outside the function calls
    x_gpu = zeros(Complex{T}, prod(shape)) |> CuArray
    y_gpu = zeros(Complex{T}, size(tr,2)*shape[3]) |> CuArray
    return StackOfStarsOp{Complex{T}}(size(tr,2)*shape[3], prod(shape), false, false
                , (res,x) -> cuprodu!(res,plan,plan_z,x,shape,tmpVec,x_gpu,y_gpu)
                , nothing
                , (res,y) -> cuctprodu!(res,plan,iplan_z,y,shape,tmpVec,x_gpu,y_gpu)
                , 0, 0, 0, false, false, false, Complex{T}[], Complex{T}[]
                , plan, plan_z, iplan_z)
  else
    plan = plan_nfft(tr, shape[1:2], m=kernelSize, σ=oversamplingFactor, precompute=NFFT.TENSOR,
                    fftflags=FFTW.ESTIMATE, blocking=true)
    tmpVec = Array{Complex{real(T)}}(undef, (size(tr,2),shape[3]))
    plan_z = plan_fft!(tmpVec, 2; flags=FFTW.MEASURE)
    iplan_z = plan_bfft!(tmpVec, 2; flags=FFTW.MEASURE)
    return StackOfStarsOp{Complex{T}}(size(tr,2)*shape[3], prod(shape), false, false
                , (res,x) -> produ!(res,plan,plan_z,x,shape,tmpVec)
                , nothing
                , (res,y) -> ctprodu!(res,plan,iplan_z,y,shape,tmpVec)
                , 0, 0, 0, false, false, false, Complex{T}[], Complex{T}[]
                , plan, plan_z, iplan_z)
  end
end

function cuprodu!(y::AbstractVector, plan::AbstractNFFTs.AbstractNFFTPlan, plan_z::AbstractFFTs.Plan, 
    x::AbstractVector, shape::Tuple, tmpVec::AbstractArray, x_gpu::CuArray, y_gpu::CuArray) 
  copyto!(x_gpu, Array(x))
  fill!(y_gpu, zero(eltype(y)))
  produ!(y_gpu, plan, plan_z, x_gpu, shape, tmpVec)
  copyto!(y, Array(y_gpu))
end

function cuctprodu!(x::AbstractVector, plan::AbstractNFFTs.AbstractNFFTPlan, plan_z::AbstractFFTs.Plan, 
    y::AbstractVector, shape::Tuple, tmpVec::AbstractArray, x_gpu::CuArray, y_gpu::CuArray)
  copyto!(y_gpu, Array(y))
  fill!(x_gpu, zero(eltype(x)))
  ctprodu!(x_gpu, plan, plan_z, y_gpu, shape, tmpVec)
  copyto!(x, Array(x_gpu))
end

function produ!(y::AbstractVector, plan::AbstractNFFTs.AbstractNFFTPlan, plan_z::AbstractFFTs.Plan, x::AbstractVector, shape::Tuple, tmpVec::AbstractArray) 
  x = reshape(x, shape)  
  y = reshape(y, :, shape[3])  
  ## NFFT
  for i=1:shape[3]
    mul!(view(y,:,i), plan, (view(x,:,:,i)))
  end
  fft_multiply_shift!(plan_z, y, tmpVec) # FFT
end

function ctprodu!(x::AbstractVector, plan::AbstractNFFTs.AbstractNFFTPlan, plan_z::AbstractFFTs.Plan, y::AbstractVector, shape::Tuple, tmpVec::AbstractArray)
  x = reshape(x, (shape))  
  y = reshape(y, :, shape[3])  
  fft_multiply_shift!(plan_z, y, tmpVec) # FFT
  ## NFFT
  for i=1:shape[3]
    mul!(view(x,:,:,i), adjoint(plan), (view(y,:,i)))
  end
end

function fft_multiply_shift!(plan::AbstractFFTs.Plan, y::AbstractArray, tmpVec::AbstractArray)
  ifftshift!(tmpVec, y)
  plan * tmpVec
  fftshift!(y, tmpVec)
  y *= 1/sqrt(size(tmpVec,2))
end

# function cuprodu!(y::AbstractVector, plan::AbstractNFFTs.AbstractNFFTPlan, plan_z::AbstractFFTs.Plan, x::AbstractVector, shape::Tuple, tmpVec::AbstractArray) 
#   x_gpu = CuArray(x)
#   y_gpu = CuArray(y)
#   x_gpu = reshape(x_gpu, shape)  
#   y_gpu = reshape(y_gpu, :, shape[3])  
#   cuprodu_kernel!(y_gpu, plan, x_gpu)
#   fft_multiply_shift!(y_gpu, plan_z, y_gpu, tmpVec) # FFT

#   copyto!(y, Array(reshape(y_gpu,:)))
# end

# function cuprodu_kernel!(y::CuArray, plan::AbstractNFFTs.AbstractNFFTPlan, x::CuArray) 
#   for i=1:size(x,3)
#     mul!(view(y,:,i), plan, view(x,:,:,i))
#   end
# end

# function cuctprodu!(x::AbstractVector, plan::AbstractNFFTs.AbstractNFFTPlan, plan_z::AbstractFFTs.Plan, y::AbstractVector, shape::Tuple, tmpVec::AbstractArray)
#   x_gpu = CuArray(x)
#   y_gpu = CuArray(y)
#   x_gpu = reshape(x_gpu, (shape))  
#   y_gpu = reshape(y_gpu, :, shape[3])  
#   fft_multiply_shift!(y_gpu, plan_z, y_gpu, tmpVec) # FFT
#   cuctprodu_kernel!(x_gpu, plan, y_gpu)

#   copyto!(x, Array(reshape(x_gpu,:)))
# end

# function cuctprodu_kernel!(x::CuArray, plan::AbstractNFFTs.AbstractNFFTPlan, y::CuArray)
#   for i=1:size(x,3)
#     mul!(view(x,:,:,i), adjoint(plan), view(y,:,i))
#   end
# end