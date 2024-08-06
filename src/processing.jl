using ReconBMRR
using CUDA
using JLD2, CodecZlib

# CUDA.device!(2) # Set GPU device
Threads.@threads for i in 1:Threads.nthreads()
    a = CuArray([1, 2, 3])  # Trigger CUDA initialization
    @show a
end

function processFINO(filename, withMotion=true)
    r = jldopen(filename)["r"]
    r.pathProc = dirname(replace(filename, "exp_raw" => "exp_pro"))
    if !isdir(r.pathProc)
        mkpath(r.pathProc)  # This will also create any necessary parent directories
    end
    if !withMotion
        r.reconParameters[:motionGating] = false
        r.reconParameters[:ringingFilter] = true 
    end
    r.reconParameters[:iterativeReconParams][:verboseIteration] = true
    time_taken = @elapsed begin
        r = perform(r)
        if withMotion 
            filename2 = saveasImDataParams(r, saveMotionState=1)
        else
            filename2 = saveasImDataParams(r)
        end
        postprocessing.postprocessing_fino_save(filename2)
    end

    # Write the elapsed time to a text file
    output_filename = joinpath(filename2[1:end-3] * "_time_taken_.txt")
    open(output_filename, "w") do file
        write(file, "$time_taken")
    end
    println("The function took $time_taken seconds.")
end