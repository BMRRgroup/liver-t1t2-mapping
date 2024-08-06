using Pkg
Pkg.activate(".")
# ENV["HDF5_DISABLE_VERSION_CHECK"] = 1
# ENV["LD_LIBRARY_PATH"] = ""
using Revise # useful package if something needs to be changed in the Julia code without restarting the Julia session
using PyCall # call python from Julia
# Python import
directory = joinpath(@__DIR__, "../src")
py"""
import sys
sys.path.append($directory)
"""
postprocessing = pyimport("postprocessing")
# FINO processing function
include("../src/processing.jl")

# Reconstruction and post-processing
# T1 phantom
processFINO("data/exp_raw/t1phantom.jld2", false)

# Only post-processing
# postprocessing.postprocessing_fino_save("data/exp_pro/20230107_FINOMappingPhantom/20230107_093945_602_ImDataParamsBMRR.h5")