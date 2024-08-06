import numpy as np
import copy
import sys
import os
from python_postprocessing.interface import ImDataParamsRelax

def postprocessing_fino(obj):
    obj.Masks["tissueMask"] = obj.get_tissueMaskFilled(5, iDyn=0)
    obj.set_FatModel()
    obj.run_fieldmapping()
    obj.perform_dictionary_matching(fieldmap=obj.WFIparams["fieldmap_Hz"][:,:,:,0], complex_signal=True)
    return obj

def postprocessing_fino_save(filename):
    obj = ImDataParamsRelax(filename)
    obj.load_dictionary("data/dictionary/20230730_pf_si_500ms_t1t2mapping_dict_highres_withB0_kprofile_complex.h5")
    obj = postprocessing_fino(obj)
    obj.save_WFIparams()
    obj.save_RelaxParams()