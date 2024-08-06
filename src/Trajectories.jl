abstract type AbstractTrajectory end
abstract type AbstractNonCartesian <: AbstractTrajectory end
abstract type AbstractCartesian <: AbstractTrajectory end

struct NonCartesian3D <: AbstractNonCartesian
   kdataNodes::Array{<:AbstractFloat}
   name::Symbol
end

struct Cartesian3D <: AbstractCartesian
    profileOrder::Array{Int, 2}
    name::Symbol
end