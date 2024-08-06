import numpy as np
import os
import h5py as h5
import copy
import matplotlib.pyplot as plt
from scipy.io import loadmat
from python_postprocessing.dictionary_matching import dictionary_matching, prepare_dictionary
from hmrGC.dixon_imaging.helper import calculate_pdff_percent
from hmrGC_dualEcho.dual_echo import DualEcho
from scipy.ndimage import binary_fill_holes
import pickle
import h5py
import hdf5storage

class ImDataParamsRelax():
    def __init__(self, filename, dicom=False):
        self.ImDataParams = {}
        self.AlgoParams = {}
        self.WFIparams = {}
        self.RelaxParams = {}
        self.Masks = {}
        if dicom:
            self._load_dicom(filename)
        else:
            if filename[-3:] == '.h5':
                self._load_h5(filename)
        self.Masks['airSignalThreshold_percent'] = 5

    def _load_h5(self, filename):
        """load a "*ImDataParamsBMRR.h5" file
        and write into ImDataParamsBMRR object
        :param filename:
        """
        print(f"Load {filename} ...", end="")
        with h5.File(filename, 'r') as f:
            nest_dict = recursively_load_attrs(f, load_data=True)

        attrs = nest_dict.keys()
        for attr in attrs:
            try:
                params = getattr(self, attr)
            except AttributeError:
                setattr(self, attr, {})
                params = getattr(self, attr)
            params.update(nest_dict[attr])
        if "filename" not in self.ImDataParams.keys():
            self.ImDataParams["filename"] = filename
        if isinstance(self.ImDataParams["fileID"], bytes):
            self.ImDataParams["fileID"] = self.ImDataParams["fileID"].decode('utf-8')
        print("Done!")

    def load_WFIparams(self, filename):
        """load a "*WFIparams.mat" file save from MATLAB
        and return it as a python dict
        :param filename:
        :returns:
        :rtype:
        """
        _, file_extension = os.path.splitext(filename)
        if file_extension == '.mat':
            print(f'Load {filename}... ', end='')
            with h5.File(filename, 'r') as f:
                attrs_dict = recursively_load_attrs(f)
                data_dict = recursively_load_data(f, attrs_dict)

            nested_dict = nest_dict(data_dict)

            self.WFIparams = nested_dict['WFIparams']
            self.set_T2s_ms()
            self.set_fatFraction_percent()

            print('Done.')
        elif file_extension == '.pickle':
            with open(filename, 'rb') as f:
                self.WFIparams = pickle.load(f)

    def save_WFIparams(self, savename=None, mat_file=True):
        self._save_params(params2save='WFIparams', savename=savename, mat_file=mat_file)

    def load_RelaxParams(self, filename):
        """load a "*RelaxParams.mat" file save from MATLAB
        and return it as a python dict
        :param filename:
        :returns:
        :rtype:
        """
        _, file_extension = os.path.splitext(filename)
        if file_extension == '.mat':
            print(f'Load {filename}... ', end='')
            with h5.File(filename, 'r') as f:
                attrs_dict = recursively_load_attrs(f)
                data_dict = recursively_load_data(f, attrs_dict)

            nested_dict = nest_dict(data_dict)

            self.RelaxParams = nested_dict['RelaxParams']
            print('Done.')
        elif file_extension == '.pickle':
            with open(filename, 'rb') as f:
                self.WFIparams = pickle.load(f)

    def save_RelaxParams(self, savename=None, mat_file=True):
        removelist = ["dictionary"]
        self._save_params(params2save='RelaxParams', savename=savename, removelist=removelist, mat_file=mat_file)

    ## water-fat separation
    def set_FatModel(self, name='default'):
        if name == 'default':
            self.AlgoParams['FatModel'] = {'freqs_ppm': np.array([-3.8 , -3.4 , -3.1 , -2.68, -2.46, -1.95, -0.5 ,  0.49,  0.59]),
                                           'relAmps': np.array([0.08991009, 0.58341658, 0.05994006, 0.08491508, 0.05994006, 
                                                             0.01498501, 0.03996004, 0.00999001, 0.05694306]),
                                           'name': 'Ren marrow'}

    def get_tissueMask(self, threshold=None, min_echo=False,
                       single_object=False, wfi_image=False, iDyn=None):
        """set tissue mask based on thresholding the MIP"""
        if threshold is None:
            threshold = self.Masks['airSignalThreshold_percent']
        if wfi_image != False:
            signal = self.WFIparams[wfi_image]
        else:
            if iDyn != None:
                signal = self.ImDataParams['signal'][:,:,:,iDyn,:]
            else:
                signal = self.ImDataParams['signal']
        if min_echo:
            echoMin = np.min(np.abs(signal), axis=3)
            tissueMask = echoMin > threshold / 100 * np.percentile(echoMin, 99)
        else:
            if len(signal.shape) > 3:
                echoMIP = np.sqrt(np.sum(np.abs(signal) ** 2, axis=3))
            else:
                echoMIP = np.sqrt(np.abs(signal) ** 2)
            tissueMask = echoMIP > threshold / 100 * np.max(echoMIP)

        if single_object:
            tissueMask = self._correct_for_single_object(tissueMask)
        return tissueMask

    def get_tissueMaskFilled(self, threshold=None, min_echo=False,
                             single_object=False, wfi_image=False, iDyn=None):
        if threshold is None: threshold = self.Masks['airSignalThreshold_percent']

        tissueMask = self.get_tissueMask(threshold=threshold, min_echo=min_echo,
                                         single_object=single_object,
                                         wfi_image=wfi_image, iDyn=iDyn)

        filledMask = np.zeros_like(tissueMask)
        for ie in range(0, tissueMask.shape[-1]):
            filledMask[:, :, ie] = binary_fill_holes(tissueMask[:, :, ie])

        return filledMask
    
    def run_fieldmapping(self, ind_dynamic=None, init_fieldmap=None):
        if len(self.ImDataParams['signal'].shape) == 4:
            hmrGC = self.get_hmrGC_obj()
            hmrGC.verbose = True

            if init_fieldmap is None:
                method = 'multi-res'
            else:
                method = 'init'
                hmrGC.phasormap = init_fieldmap

            hmrGC.perform(method)

            self.WFIparams['method'] = f'{method}'
            self.WFIparams['fieldmap_Hz'] = hmrGC.fieldmap
            self.Masks['tissueMask'] = hmrGC.mask
            for key in hmrGC.images.keys():
                self.WFIparams[key] = hmrGC.images[key]
        else:
            self.WFIparams = {}
            nDyn = self.ImDataParams['signal'].shape[3]
            if ind_dynamic is None:
                range_dynamic = range(nDyn)
            else:
                range_dynamic = ind_dynamic
            for i in range_dynamic:
                print(f'Field-mapping for dyn {i+1}')
                tmp_obj = copy.deepcopy(self)
                tmp_obj.ImDataParams['signal'] = tmp_obj.ImDataParams['signal'][:, :, :, i, :]
                if 'tissueMask' in tmp_obj.Masks:
                    if len(tmp_obj.Masks['tissueMask'].shape) == 4:
                        tmp_obj.Masks['tissueMask'] = tmp_obj.Masks['tissueMask'][:, :, :, i]
                    else:
                        tmp_obj.Masks['tissueMask'] = tmp_obj.Masks['tissueMask']
                if i == 0:
                    init_fieldmap = None
                else:
                    init_fieldmap = self.WFIparams["fieldmap_Hz"][..., 0]*2*np.pi*np.diff(self.ImDataParams["TE_s"])[0]
                tmp_obj.run_fieldmapping(init_fieldmap=init_fieldmap)
                for key in tmp_obj.WFIparams:
                    arr = tmp_obj.WFIparams[key]
                    if isinstance(arr, np.ndarray):
                        shape = list(arr.shape)
                        shape.append(nDyn)
                        if key not in self.WFIparams.keys() or \
                           list(self.WFIparams[key].shape) != shape:
                            self.WFIparams[key] = np.zeros(shape, dtype=arr.dtype)
                        self.WFIparams[key][..., i] = arr
            method = tmp_obj.WFIparams['method']
            self.WFIparams['method'] = method
    
    def get_hmrGC_obj(self):
        if 'tissueMask' in self.Masks:
            mask = self.Masks['tissueMask']
        else:
            mask = self.get_tissueMask()
        signal = self.ImDataParams['signal']
        params = {}
        params['TE_s'] = self.ImDataParams['TE_s']
        params['voxelSize_mm'] = self.ImDataParams['voxelSize_mm']
        params['fieldStrength_T'] = self.ImDataParams['fieldStrength_T']
        params['centerFreq_Hz'] = self.ImDataParams['centerFreq_Hz']
        if 'FatModel' in self.AlgoParams:
            params['FatModel'] = self.AlgoParams['FatModel']
        
        gandalf = DualEcho(signal, mask, params)
        return gandalf

    ## relaxometry
    def load_dictionary(self, filename, without_kprofile=False):
        """Load dictionary from .mat or .h5 file

        :param filename: filename

        """
        print(f'Load {filename}... ', end='')
        if filename[-4:] == '.mat':
            nested_dict = loadmat(filename)
            keys = copy.copy(list(nested_dict.keys()))
            for key in keys:
                if key.startswith('__'):
                    del nested_dict[key]
                else:
                    if key != "dictionary":
                        nested_dict[key] = nested_dict[key].squeeze()
        elif filename[-3:] == '.h5':
            with h5.File(filename, 'r') as f:
                nested_dict = {}
                nested_dict['dictionary'] = np.array(f['dictionary'])
                for k, v in f.attrs.items():
                    nested_dict[k] = v

        self.RelaxParams['dictionary'] = nested_dict
        if 'prof_ordering' not in nested_dict.keys():
            self.RelaxParams['dictionary']['prof_ordering'] = np.zeros_like(self.RelaxParams['dictionary']['durations'])
        if without_kprofile:
            self.RelaxParams["dictionary"]["dictionary"] = self.RelaxParams["dictionary"]["dictionary"][..., np.newaxis]
        print(self.RelaxParams["dictionary"]["dictionary"].shape)
        print('done!')

    def perform_dictionary_matching(self, signal=None, plot_signal=None, plot_landscape=None, fieldmap=None,
                                    complex_signal=True, fat=False, compute_pd=True):
        """Match signal to Bloch-simulated dictionary for obtaining T1/T2 maps

        :param signal: signal to be matched, if None use default signal
        :param plot_signal: debug option
        :param plot_landscape: debug option
        :param fieldmap: fieldmap to be used in dictionary matching
        :returns:

        """
        if signal is None:
            signal = self._get_default_signal(magnitude=(complex_signal != True), fat=fat)

        params = {}
        params['duration'] = self.RelaxParams['TE_s'] * 1e3
        if 'angle_deg' in self.RelaxParams:
            params['angle'] = self.RelaxParams['angle_deg']
        else:
            params['angle'] = np.zeros_like(params['duration'])
        if 'delay_s' in self.RelaxParams:
            params['delay'] = self.RelaxParams['delay_s'] * 1e3
        else:
            params['delay'] = np.zeros_like(params['duration'])
        if 'startup_delay_s' in self.RelaxParams:
            params['startup_delay'] = self.RelaxParams['startup_delay_s'] * 1e3
        else:
            params['startup_delay'] = np.zeros_like(params['duration'])
        if 'prof_ordering' in self.RelaxParams:
            params['prof_ordering'] = self.RelaxParams['prof_ordering']
        else:
            params['prof_ordering'] = np.zeros_like(params['duration'])
        if 'B0s' in self.RelaxParams['dictionary'].keys():
            print('Perform dictionary matching with B0... ', end='')
        else:
            print('Perform dictionary matching without B0... ', end='')

        mask = self.Masks['tissueMask']

        if fat:
            self.RelaxParams['T1_fat_ms'] = np.zeros_like(mask, dtype=np.float32)
            params_t1 = self.RelaxParams["T1_fat_ms"]
            self.RelaxParams['T2_fat_ms'] = np.zeros_like(mask, dtype=np.float32)
            params_t2 = self.RelaxParams["T2_fat_ms"]
            self.RelaxParams['B1_fat'] = np.zeros_like(mask, dtype=np.float32)
            params_b1 = self.RelaxParams["B1_fat"]
            self.RelaxParams['PD_fat'] = np.zeros_like(mask, dtype=np.float32)
            params_pd = self.RelaxParams["PD_fat"]
            if 'psf' in self.RelaxParams["dictionary"].keys():
                psf_shape = self.RelaxParams["dictionary"]["psf"].shape
                self.RelaxParams['PSF_fat'] = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2],
                                                    psf_shape[-2], psf_shape[-1]), dtype=np.float32)
                params_psf = self.RelaxParams['PSF_fat']
        else:
            self.RelaxParams['T1_ms'] = np.zeros_like(mask, dtype=np.float32)
            params_t1 = self.RelaxParams["T1_ms"]
            self.RelaxParams['T2_ms'] = np.zeros_like(mask, dtype=np.float32)
            params_t2 = self.RelaxParams["T2_ms"]
            self.RelaxParams['B1'] = np.zeros_like(mask, dtype=np.float32)
            params_b1 = self.RelaxParams["B1"]
            self.RelaxParams['PD'] = np.zeros_like(mask, dtype=np.float32)
            params_pd = self.RelaxParams["PD"]
            if 'psf' in self.RelaxParams["dictionary"].keys():
                psf_shape = self.RelaxParams["dictionary"]["psf"].shape
                self.RelaxParams['PSF'] = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2],
                                                    psf_shape[-2], psf_shape[-1]), dtype=np.float32)
                params_psf = self.RelaxParams['PSF']

        tmp_dict, tmp_psf = prepare_dictionary(self.RelaxParams['dictionary'], params, 
                                               complex_signal)
        loop_slices = True
        if loop_slices:
            # Loop over slices
            for i in range(signal.shape[2]):
                mask_i = mask[:, :, i]
                if np.sum(mask_i) > 0:
                    signal_i = signal[:, :, i]
                    signal_i = signal_i[mask_i]

                    if 'B0s' in self.RelaxParams['dictionary'].keys():
                        if fieldmap is None:
                            params['fieldmap'] = np.mean(self.WFIparams['fieldmap_Hz'][:, :, i][mask_i],
                                                        axis=-1)
                        else:
                            params['fieldmap'] = fieldmap[:, :, i][mask_i]

                    if plot_landscape and plot_landscape[0][-1] == i:
                        print(i)
                        tmp_plot_landscape = copy.deepcopy(plot_landscape)
                        tmp_plot_landscape[0] = plot_landscape[0][:2]
                    else:
                        tmp_plot_landscape = None

                    t1map_i, t2map_i, b1map_i, pdmap_i, psf_i = dictionary_matching(signal_i,
                        self.RelaxParams['dictionary'], params, mask_i, tmp_dict, tmp_psf, 
                        plot_signal=plot_signal, plot_landscape=tmp_plot_landscape, complex_dict=complex_signal, 
                        compute_pd=compute_pd)

                    t1map = np.zeros_like(mask_i, dtype=np.float32)
                    t1map[mask_i] = t1map_i
                    params_t1[:, :, i] = t1map*1e3
                    t2map = np.zeros_like(mask_i, dtype=np.float32)
                    t2map[mask_i] = t2map_i
                    params_t2[:, :, i] = t2map*1e3
                    b1map = np.zeros_like(mask_i, dtype=np.float32)
                    b1map[mask_i] = b1map_i
                    params_b1[:, :, i] = b1map
                    pdmap = np.zeros_like(mask_i, dtype=np.float32)
                    pdmap[mask_i] = pdmap_i
                    params_pd[:, :, i] = pdmap
                    if psf_i is not None:
                        psf = np.zeros((mask_i.shape[0], mask_i.shape[1],
                                        psf_shape[-2], psf_shape[-1]), dtype=np.float32)
                        #import pdb; pdb.set_trace()
                        mask_i_reshaped = np.repeat(np.repeat(mask_i[..., np.newaxis, np.newaxis], psf_shape[-2], axis=-2), psf_shape[-1], axis=-1)
                        psf[mask_i_reshaped] = np.abs(psf_i.flatten())
                        params_psf[:, :, i, :, :] = psf
        else:
            signal = signal[mask]

            if 'B0s' in self.RelaxParams['dictionary'].keys():
                if fieldmap is None:
                    params['fieldmap'] = np.mean(self.WFIparams['fieldmap_Hz'][mask],
                                                 axis=-1)
                else:
                    params['fieldmap'] = fieldmap[mask]

            t1map, t2map, b1map, pdmap, psf = dictionary_matching(signal,
                self.RelaxParams['dictionary'], params, mask, tmp_dict, tmp_psf, plot_signal=plot_signal,
                plot_landscape=plot_landscape, complex_dict=complex_signal, compute_pd=compute_pd)

            params_t1[mask] = t1map*1e3
            params_t2[mask] = t2map*1e3
            params_b1[mask] = b1map
            params_pd[mask] = pdmap
            if psf is not None:
                mask_reshaped = np.repeat(np.repeat(mask[..., np.newaxis, np.newaxis], psf_shape[-2], axis=-2), psf_shape[-1], axis=-1)
                params_psf[mask_reshaped] = np.abs(psf.flatten())

        self.RelaxParams['method'] = 'dictionary_matching'
        print('done!')

    def _get_default_signal(self, magnitude=False, fat=False):
        if fat:
            if magnitude:
                signal = np.abs(self.WFIparams['fat'])
            else:
                signal = np.real(self.WFIparams['fat']*np.exp(-1j*np.angle(self.WFIparams['fat'][:,:,:,0][:,:,:,np.newaxis])))
        else:
            if len(self.ImDataParams['TE_s']) > 1:
                if magnitude:
                    signal = np.abs(self.WFIparams['water'])
                else:
                    signal = np.real(self.WFIparams['water']*np.exp(-1j*np.angle(self.WFIparams['water'][:,:,:,0][:,:,:,np.newaxis])))
                    # signal = np.imag(self.WFIparams['water'])#*np.exp(-1j*np.angle(self.WFIparams['water'][:,:,:,0][:,:,:,np.newaxis])))
            else:
                if magnitude:
                    signal = np.abs(self.ImDataParams['signal'])
                else:
                    signal = self.ImDataParams['signal']
        return signal
    
    def _save_params(self, params2save, savename=None, removelist=None, mat_file=True):
        save_params = getattr(self, params2save).copy()

        if mat_file:
            end = '.mat'
        else:
            end = '.pickle'
        path, filename = self._get_path_to_save(savename)
        if 'method' in save_params:
            savename = path + '/' + filename + '_' + params2save + '_' + \
                       save_params['method'] + end
        else:
            savename = path + '/' + filename + '_' + params2save + end

        if removelist:
            save_params = removeElementsInDict(save_params, removelist)

        print('save ' + savename, '...', end='')
        if mat_file:
            hdf5storage.savemat(savename, {params2save: save_params})
        else:
            with open(savename, 'wb') as f:
                pickle.dump(save_params, f)
        print('done!')

    def _get_path_to_save(self, savename):
        """Get the path and filename based on the passed savename

        :param savename: filename/path for saved object
        :returns: path, filename

        """
        if not savename:
            path = os.path.dirname(self.ImDataParams['filename'])
            if path == '':
                path = '.'
            filename = self.ImDataParams['fileID']
        else:
            if os.path.isdir(savename):
                path = os.path.dirname(savename)
                filename = self.ImDataParams['fileID']
            else:
                path = os.path.dirname(savename)
                filename = os.path.basename(savename)
        return path, filename
    
    def set_T2s_ms(self):
        if 'tissueMask' in self.Masks:
            mask = self.Masks['tissueMask']
        else:
            mask = self.get_tissueMask()

        if 'R2s_Hz' in self.WFIparams or 'waterR2s_Hz' in self.WFIparams or \
                'fatR2s_Hz' in self.WFIparams:
            self.WFIparams = get_T2s_ms(self.WFIparams, mask)

    def set_fatFraction_percent(self):
        if {'PD', 'PD_fat'} <= self.RelaxParams.keys():
            ff = calculate_pdff_percent(self.RelaxParams['PD'],
                                        self.RelaxParams['PD_fat'])

            ff[np.isnan(ff)] = 0
            self.WFIparams['fatFraction_percent'] = ff
        elif {'water', 'fat'} <= self.WFIparams.keys():
            ff = calculate_pdff_percent(self.WFIparams['water'],
                                        self.WFIparams['fat'])
            ff[np.isnan(ff)] = 0
            self.WFIparams['fatFraction_percent'] = ff



def recursively_load_attrs(h5file, path='/', load_data=False):
    """
    recursively load attributes for all groups and datasets in
    hdf5 file as python dict
    :param h5file: h5py.File(<filename>, 'r')
    :param path: "directory path" in h5 File
    :returns:
    :rtype: nested dicts
    """

    attrs_dict = {}
    for k, v in h5file[path].items():

        d = {}
        for ak, av in v.attrs.items():
            d[ak] = av

        if isinstance(v, h5._hl.dataset.Dataset):  # FIXME: call to a protected class function
            if load_data:
                attrs_dict[k] = np.array(v)
            else:
                attrs_dict[k] = d

        elif isinstance(v, h5._hl.group.Group):  # FIXME: call to a protected class function
            d.update(recursively_load_attrs(
                h5file, os.path.join(path, k), load_data))
            attrs_dict[k] = d

    return attrs_dict


def recursively_load_data(h5file, attrs_dict, path='/'):
    """
    recursively load data for all groups and datasets in
    hdf5 file as python dict corresponding to attrs_dict
    (see function recursively_load_attrs)
    :param h5file: h5py.File(<filename>, 'r')
    :param attrs_dict: output of function recursively_load_attrs
    :returns:
    :rtype: nested dicts
    """

    result = {}
    for k, v in attrs_dict.items():

        if k == '#refs#':
            continue

        if k == '#subsystem#':
            continue

        if isinstance(v, dict):

            if v.get('MATLAB_class') == b'function_handle':
                continue
            elif v.get('MATLAB_class') != b'struct':

                val = h5file[path + k + '/'][...]
                arrays3d = np.array(['signal', 'refsignal', 'fieldmap_Hz', 'R2s_Hz',
                                     'water', 'fat', 'silicone', 'fatFraction_percent'])
                if ~np.isin(k, arrays3d):
                    val = np.squeeze(val)

                if isinstance(val, np.ndarray) and \
                        val.dtype == [('real', '<f4'), ('imag', '<f4')]:
                    val = np.transpose(val.view(np.complex64)).astype(np.complex64)
                elif isinstance(val, np.ndarray) and \
                        (val.dtype == [('real', '<f8'), ('imag', '<f8')]):
                    val = np.transpose(val.view(np.complex)).astype(np.complex64)
                elif isinstance(val, np.ndarray) and \
                        (val.dtype == 'float64' or val.dtype == 'float32'):
                    val = (np.transpose(val).astype(np.float32))
                elif isinstance(val, np.ndarray) and \
                        (val.dtype == 'uint64' or val.dtype == 'uint32'):
                    val = (np.transpose(val).astype(np.uint32))
                elif isinstance(val, np.ndarray) and \
                        (val.dtype == 'bool_' or val.dtype == 'uint8'):
                    val = (np.transpose(val).astype(np.bool_))

                if v.get('MATLAB_class') == b'char':
                    try:
                        val = ''.join([chr(c) for c in val])
                    except:  # FIXME: bare except is bad practice
                        val = ''

                result[path + k + '/'] = val
            else:
                result.update(recursively_load_data(h5file, v, path + k + '/'))

    return result


def nest_dict(flat_dict):
    seperator = '/'
    nested_dict = {}
    for k, v in flat_dict.items():

        path_list = list(filter(None, k.split(seperator)))  # removes '' elements
        split_key = path_list.pop(0)
        left_key = seperator.join(path_list)

        if left_key == '':
            nested_dict[split_key] = v
            continue

        if not nested_dict.get(split_key):  # init new dict
            nested_dict[split_key] = {}

        if left_key != '':
            nested_dict[split_key].update({left_key: v})

    return nested_dict


def removeElementsInDict(inDict, listofstrings):
    for item in listofstrings:
        try:
            del inDict[item]
        except:  # FIXME: bare except is bad practice
            print('Dictionary has no item {}'.format(item))
    return inDict


def get_T2s_ms(inParams, mask):
    # mask = inParams['fieldmap_Hz'] != 0 # implicit mask definition
    if 'R2s_Hz' in inParams:
        T2s_ms = 1e3 / inParams['R2s_Hz']
        T2max = np.max(T2s_ms[~np.isinf(T2s_ms)])
        T2s_ms[np.isinf(T2s_ms)] = T2max
        inParams['T2s_ms'] = T2s_ms * mask

    if 'waterR2s_Hz' in inParams:
        wT2s_ms = 1e3 / inParams['waterR2s_Hz']
        T2max = np.max(wT2s_ms[~np.isinf(wT2s_ms)])
        wT2s_ms[np.isinf(wT2s_ms)] = T2max
        inParams['waterT2s_ms'] = wT2s_ms * mask

    if 'fatR2s_Hz' in inParams:
        fT2s_ms = 1e3 / inParams['fatR2s_Hz']
        T2max = np.max(fT2s_ms[~np.isinf(fT2s_ms)])
        fT2s_ms[np.isinf(fT2s_ms)] = T2max
        inParams['fatT2s_ms'] = fT2s_ms * mask

    return inParams