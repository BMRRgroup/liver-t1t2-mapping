import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange
import copy
EPSILON = 1e-6

def dictionary_matching(signal, dictionary, params, mask, tmp_dict, tmp_psf, plot_signal=None, 
                        plot_landscape=None, complex_dict=True, compute_pd=True):
    real_signal = np.reshape(signal, [int(np.prod(signal.shape[:-1])),
                                      signal.shape[-1]])
    real_signal_test = copy.deepcopy(real_signal)
    nte = real_signal.shape[-1]

    l2_signal = np.sqrt(np.sum(np.square(real_signal), axis=-1))
    real_signal /= l2_signal[..., np.newaxis]

    if 'fieldmap' in params.keys():
        fieldmap = params['fieldmap'][:, np.newaxis]
        fieldmap_dict = dictionary['B0s'].flatten()[:, np.newaxis]

        fieldmap_ind = np.argmin(njit_rmse_prange_dict(fieldmap, fieldmap_dict),
                                 axis=-1)
        tmp_dict = np.reshape(tmp_dict, [tmp_dict.shape[0], int(np.prod(tmp_dict.shape[1:-1])), tmp_dict.shape[-1]])

        min_err = np.argmin(njit_rmse_prange_voxels(real_signal, tmp_dict, fieldmap_ind),
                            axis=-1)
        t1, t2 = np.unravel_index(min_err, dictionary['dictionary'].shape[1:3])
        pd = np.zeros(real_signal.shape[0])
        if compute_pd:
            for i in range(pd.shape[0]):
                a, _, _, _ = np.linalg.lstsq(tmp_dict[fieldmap_ind[i], min_err[i],:,np.newaxis], real_signal_test[i,:], rcond=None)
                pd[i] = a
    else:
        if plot_signal:
            _plot_signal(real_signal, tmp_dict, mask, dictionary, plot_signal)

        tmp_dict = np.reshape(tmp_dict, [int(np.prod(tmp_dict.shape[:-1])),
                                         tmp_dict.shape[-1]])

        err = np.inner(real_signal, tmp_dict)
        min_err = np.argmax(err, axis=-1)
        # err = njit_rmse_prange_dict(real_signal, tmp_dict)
        # min_err = np.argmin(err, axis=-1)

        if plot_landscape:
            _plot_landscape(err, mask, dictionary, plot_landscape)

        #if len(dictionary['dictionary'].shape) == 3:
        t1, t2 = np.unravel_index(min_err, dictionary['dictionary'].shape[:2])
        pd = np.zeros(real_signal.shape[0])
        if compute_pd:
            for i in range(pd.shape[0]):
                a, _, _, _ = np.linalg.lstsq(tmp_dict[min_err[i],:,np.newaxis], real_signal_test[i,:], rcond=None)
                pd[i] = a
        # else:
        #     b1, t1, t2 = np.unravel_index(min_err, dictionary['dictionary'].shape[:3])

    #pd = np.min(np.divide(real_signal_test, tmp_dict[min_err, :]), axis=-1)
    #pd = np.diag(np.inner(real_signal_test, tmp_dict[min_err, :]))
    # for i in range(pd.shape[0]):
    #     a, _, _, _ = np.linalg.lstsq(tmp_dict[min_err[i],:,np.newaxis], real_signal_test[i,:], rcond=None)
    #     pd[i] = a
    #
    # real_signal_test = real_signal_test.T
    # pd = np.linalg.inv(real_signal_test.T @ real_signal_test) @ real_signal_test.T @ tmp_dict[min_err,:]
    # print(pd)
    # pd = l2_signal / l2_dict[min_err]
    t1map = dictionary['T1s'].flatten()[t1]
    t2map = dictionary['T2s'].flatten()[t2]
    if 'b1' in locals():
        b1map = dictionary['B1'].flatten()[b1]
    else:
        b1map = None
    if tmp_psf is not None:
        psf = tmp_psf[fieldmap_ind, t1, t2, :, :]
    else:
        psf = None
    return t1map, t2map, b1map, pd, psf

def prepare_dictionary(dictionary, params, complex_dict=True):
    nte = len(params['duration'])
    if complex_dict:
        tmp_dict = np.real(dictionary['dictionary']*np.exp(-1j*np.angle(dictionary['dictionary'][...,0,:][...,np.newaxis,:])))
    else:
        tmp_dict = np.abs(dictionary['dictionary'])
    if 'psf' in dictionary.keys():
        tmp_psf = dictionary['psf']
    else:
        tmp_psf = None
    if "kz_profiles" in dictionary.keys():
        tmp_dict = tmp_dict[..., 0]
    if 'angle_deg' in dictionary and 'delays' in dictionary and not 'startup_delays' in dictionary:
        dict_param = np.concatenate((dictionary['durations'][:, np.newaxis],
                                     dictionary['angle_deg'][:, np.newaxis],
                                     dictionary['delays'][:, np.newaxis]), axis=-1)
        dict_te_series = np.zeros((np.concatenate((list(tmp_dict.shape[:-1]),
                                                   [nte]))))
        if tmp_psf is not None:
            psf_te_series = np.zeros((np.concatenate((list(tmp_psf.shape[:-2]),
                                                      [nte, tmp_psf.shape[-1]]))))

        for i in range(nte):
            ind = np.argwhere((np.abs(dict_param - np.array([params['duration'][i],
                                                             params['angle'][i],
                                                             params['delay'][i]]))<1e-6).all(axis=1) == True)
            dict_te_series[..., i] = tmp_dict[..., ind.flatten()[0]]
            if tmp_psf is not None:
                psf_te_series[..., i, :] = tmp_psf[..., ind.flatten()[0], :]
        tmp_dict = dict_te_series
        if tmp_psf is not None:
            tmp_psf = psf_te_series
    elif 'angle_deg' in dictionary and 'delays' in dictionary and 'startup_delays' in dictionary \
         and 'prof_ordering' in dictionary:
        dict_param = np.concatenate((dictionary['durations'][:, np.newaxis],
                                     dictionary['angle_deg'][:, np.newaxis],
                                     dictionary['delays'][:, np.newaxis],
                                     dictionary['startup_delays'][:, np.newaxis],
                                     dictionary['prof_ordering'][:, np.newaxis]), axis=-1)
        dict_te_series = np.zeros((np.concatenate((list(tmp_dict.shape[:-2]),
                                                   [nte]))))
        # dict_te_series = np.zeros((np.concatenate((list(tmp_dict.shape[:-1]),
        #                                            [nte]))))
        if tmp_psf is not None:
            psf_te_series = np.zeros((np.concatenate((list(tmp_psf.shape[:-2]),
                                                      [nte, tmp_psf.shape[-1]]))))
        for i in range(nte):
            ind = np.argwhere((np.abs(dict_param - np.array([params['duration'][i],
                                                             params['angle'][i],
                                                             params['delay'][i],
                                                             params['startup_delay'][i],
                                                             params['prof_ordering'][i]]))<1e-6).all(axis=1) == True)

            # dict_te_series[..., i] = tmp_dict[..., ind.flatten()[0]]
            dict_te_series[..., i] = tmp_dict[..., ind.flatten()[0], 0]
            if tmp_psf is not None:
                psf_te_series[..., i, :] = tmp_psf[..., ind.flatten()[0], :]
        tmp_dict = dict_te_series
        if tmp_psf is not None:
            tmp_psf = psf_te_series
    else:
        mask_durs = np.squeeze((dictionary['durations'] - params['duration']) < 1e-6)
        tmp_dict = tmp_dict[..., mask_durs, 0]
        tmp_psf = None
    assert tmp_dict.shape[-1] == nte
    l2_dict = np.sqrt(np.sum(np.square(tmp_dict), axis=-1))
    tmp_dict /= (l2_dict[..., np.newaxis] + EPSILON)
    return tmp_dict, tmp_psf

@njit(parallel=True)
def njit_rmse_prange_dict(real_signal, tmp_dict):
    nte = real_signal.shape[-1]
    err = np.zeros((real_signal.shape[0], tmp_dict.shape[0]))
    for i in prange(tmp_dict.shape[0]):
        simulated_signal = tmp_dict[i]
        err[:, i] = 1 / nte * (np.sum(np.square(real_signal -
                                                simulated_signal), axis=-1))
    return err


@njit(parallel=True)
def njit_rmse_prange_voxels(real_signal, tmp_dict, fieldmap_ind):
    nte = real_signal.shape[-1]
    err = np.zeros((real_signal.shape[0], tmp_dict.shape[1]))
    for i in prange(real_signal.shape[0]):
        real_signal_voxel = real_signal[i]
        simulated_signal_voxel = tmp_dict[fieldmap_ind[i]]
        err[i, :] = 1 / nte * (np.sum(np.square(real_signal_voxel -
                                                simulated_signal_voxel), axis=-1))
    return err

def _plot_landscape(rmse, mask, dictionary, plot_landscape):
    rmse3d = np.zeros((mask.shape[0], mask.shape[1],
                        rmse.shape[-1]))
    rmse3d[mask] = rmse
    if len(plot_landscape) == 2:
        clim = plot_landscape[1]
    else:
        clim = np.log([np.min(rmse3d[plot_landscape[0][0], plot_landscape[0][1]]),
                        np.max(rmse3d[plot_landscape[0][0], plot_landscape[0][1]])])
    plt.figure()
    print(dictionary['dictionary'].shape)
    if len(dictionary['dictionary'].shape) == 4:
        plt.imshow(np.reshape(np.log(rmse3d[plot_landscape[0][0], plot_landscape[0][1]]), (dictionary['dictionary'].shape[0]*dictionary['dictionary'].shape[1],
                                                                                            dictionary['dictionary'].shape[2])),
                vmin=clim[0], vmax=clim[1], extent=[dictionary['T2s'][0],
                                                    dictionary['T2s'][-1],
                                                    dictionary['T1s'][-1],
                                                    dictionary['T1s'][0]],
                aspect='auto')
    else:
        plt.imshow(np.reshape(np.log(rmse3d[plot_landscape[0][0], plot_landscape[0][1]]), (dictionary['dictionary'].shape[0],
                                                                dictionary['dictionary'].shape[1])),
                vmin=clim[0], vmax=clim[1], extent=[dictionary['T2s'][0],
                                                    dictionary['T2s'][-1],
                                                    dictionary['T1s'][-1],
                                                    dictionary['T1s'][0]],
                aspect='auto')
    plt.ylabel('T1 (ms)')
    plt.xlabel('T2 (ms)')
    plt.colorbar()
    plt.show()

def _plot_signal(real_signal, tmp_dict, mask, dictionary, plot_signal):
    nte = real_signal.shape[-1]
    plt.figure()
    plt.scatter(np.arange(nte), tmp_dict[np.abs(dictionary['T1s'] - plot_signal[1][0]) < 0.001,
                                         np.abs(dictionary['T2s'] - plot_signal[1][1]) < 0.001], label='dictionary')
    sig3d = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2], nte))
    sig3d[mask] = real_signal
    plt.scatter(np.arange(nte), sig3d[plot_signal[0][0], plot_signal[0][1], plot_signal[0][2], :], label='signal')
    plt.legend()
    plt.show()