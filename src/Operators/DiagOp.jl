function DiagOp(ops :: AbstractLinearOperator...)
    nrow = 0
    ncol = 0
    S = eltype(ops[1])
    for i = 1:length(ops)
        nrow += ops[i].nrow
        ncol += ops[i].ncol
        S = promote_type(S, eltype(ops[i]))
    end
  
    xIdx = cumsum(vcat(1,[ops[i].ncol for i=1:length(ops)]))
    yIdx = cumsum(vcat(1,[ops[i].nrow for i=1:length(ops)]))
  
    Op = MRIOperators.DiagOp{S}( nrow, ncol, false, false,
        (res,x) -> (diagOpProd(res,x,nrow,xIdx,yIdx,ops...)),
        (res,y) -> (diagOpTProd(res,y,ncol,yIdx,xIdx,ops...)),
        (res,y) -> (diagOpCTProd(res,y,ncol,yIdx,xIdx,ops...)),
        0, 0, 0, false, false, false, S[], S[],
        [ops...], false, xIdx, yIdx)
    return Op
end
  
function DiagOp(op::AbstractLinearOperator, N=1)
    nrow = N*op.nrow
    ncol = N*op.ncol
    S = eltype(op)
    ops = [copy(op) for n=1:N]
  
    xIdx = cumsum(vcat(1,[ops[i].ncol for i=1:length(ops)]))
    yIdx = cumsum(vcat(1,[ops[i].nrow for i=1:length(ops)]))
  
    Op = MRIOperators.DiagOp{S}( nrow, ncol, false, false,
        (res,x) -> (diagOpProd(res,x,nrow,xIdx,yIdx,ops...)),
        (res,y) -> (diagOpTProd(res,y,ncol,yIdx,xIdx,ops...)),
        (res,y) -> (diagOpCTProd(res,y,ncol,yIdx,xIdx,ops...)),
        0, 0, 0, false, false, false, S[], S[],
        ops, true, xIdx, yIdx )
    return Op
end
 
function diagOpProd(y::AbstractVector{T}, x::AbstractVector{T}, nrow::Int, xIdx, yIdx, ops :: AbstractLinearOperator...) where T
    #@floop for i=1:length(ops)
    for i=1:length(ops)
        mul!(view(y,yIdx[i]:yIdx[i+1]-1), ops[i], view(x,xIdx[i]:xIdx[i+1]-1))
    end
    return y
end

function diagOpTProd(y::AbstractVector{T}, x::AbstractVector{T}, ncol::Int, xIdx, yIdx, ops :: AbstractLinearOperator...) where T
    #@floop for i=1:length(ops)
    for i=1:length(ops)
        mul!(view(y,yIdx[i]:yIdx[i+1]-1), transpose(ops[i]), view(x,xIdx[i]:xIdx[i+1]-1))
    end
    return y
end

function diagOpCTProd(y::AbstractVector{T}, x::AbstractVector{T}, ncol::Int, xIdx, yIdx, ops :: AbstractLinearOperator...) where T
    #@floop for i=1:length(ops)
    for i=1:length(ops)
        mul!(view(y,yIdx[i]:yIdx[i+1]-1), adjoint(ops[i]), view(x,xIdx[i]:xIdx[i+1]-1))
    end
    return y
end