export NFFTOp

function NFFTOp(shape::Tuple, tr::AbstractMatrix{T}; cuda::Bool=true, toeplitz=false, oversamplingFactor=2, kernelSize=3, kargs...) where {T}
  if cuda
    plan = plan_nfft(CuArray, tr, shape, m=kernelSize, σ=oversamplingFactor, precompute=NFFT.TENSOR, 
      fftflags=FFTW.ESTIMATE, blocking=true)
    return MRIOperators.NFFTOp{Complex{T}}(size(tr,2), prod(shape), false, false
    , (res,x) -> produ!(res,plan,x)
    , nothing
    , (res,y) -> ctprodu!(res,plan,y)
    , 0, 0, 0, false, false, false, Complex{T}[], Complex{T}[]
    , plan, toeplitz)
  else
    return MRIOperators.NFFTOp(shape, tr; toeplitz=toeplitz, oversamplingFactor=oversamplingFactor, kernelSize=kernelSize, kargs...)
  end
end

function produ!(y::AbstractVector, plan::CuNFFT.CuNFFTPlan, x::AbstractVector) 
  y_gpu = zeros(eltype(y), size(y)...) |> CuArray
  mul!(y_gpu, plan, CuArray(reshape(x,plan.N)))
  copyto!(y, Array(y_gpu))
end

function ctprodu!(x::AbstractVector, plan::CuNFFT.CuNFFTPlan, y::AbstractVector)
  x_gpu = zeros(eltype(x), size(x)...) |> CuArray
  mul!(reshape(x_gpu, plan.N), adjoint(plan), CuArray(y))
  copyto!(x, Array(x_gpu))
end