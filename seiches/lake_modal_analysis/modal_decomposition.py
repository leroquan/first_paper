import numpy as np
import xarray as xr


def old_get_eta2_from_mitgcm(theta_mitgcm, h1):
    theta_h1 = theta_mitgcm.sel(Z=h1, method='nearest')
    theta_h1 = theta_h1.where(theta_h1!=0, np.nan)
    theta_h1_mean = theta_h1.mean(dim=['XC','YC'])
    
    theta = theta_mitgcm.isel(Z=range(40))
    h_thetah1 = np.abs(theta - theta_h1_mean).idxmin(dim='Z')
    h_thetah1 = h_thetah1.where(~np.isnan(theta_h1), np.nan)
    eta2 = h_thetah1 - h1

    return h_thetah1, eta2


def get_eta2_from_mitgcm(theta_mitgcm, h1):
    theta_h1 = theta_mitgcm.sel(Z=h1, method='nearest')
    theta_h1 = theta_h1.where(theta_h1 != 0, np.nan)
    theta_h1_median = theta_h1.median(dim=['XC', 'YC']).compute()

    theta = theta_mitgcm.isel(Z=slice(40)).load()
    theta = theta.where(~np.isnan(theta_h1), np.nan)

    diff_theta = theta - theta_h1_median
    z = theta.Z.values
    diff_theta_np = diff_theta.values

    valid = ~np.isnan(diff_theta_np)
    last_valid_idx = valid.shape[1] - 1 - np.argmax(valid[:, ::-1, :, :], axis=1)

    crossing = diff_theta_np[:, :-1] * diff_theta_np[:, 1:] <= 0
    idx = crossing.argmax(axis=1) # idx of the closest temperature to theta_h1_median

    nt, nz, ny, nx = diff_theta_np.shape
    tt, yy, xx = np.indices((nt, ny, nx))

    d1 = diff_theta_np[tt, idx, yy, xx] # diffence between theta_h1_median and theta
    d2 = diff_theta_np[tt, idx + 1, yy, xx]

    z1 = z[idx]
    z2 = z[idx + 1]

    dz_over_dd = (z2 - z1) / (d2 - d1)
    dz_over_dd[(d2 - d1)==0] = 0

    h_thetah1 = z1 - d1 * dz_over_dd
    h_thetah1[(np.count_nonzero(crossing, axis=1)==0)&(diff_theta_np[:,0]<=0)] = z[0]
    h_thetah1[(np.count_nonzero(crossing, axis=1)==0)&(diff_theta_np[:,0]>0)] = z[last_valid_idx[(np.count_nonzero(crossing, axis=1)==0)&(diff_theta_np[:,0]>0)]]
    eta2 = h_thetah1 - h1

    h_thetah1 = xr.DataArray(
        h_thetah1,
        dims=('time', 'YC', 'XC'),
        coords={
        'time': theta.time,
        'YC': theta.YC,
        'XC': theta.XC,
    },
        name='h_thetah1',
    )

    eta2 = h_thetah1 - h1
    eta2.name = 'eta2'

    return h_thetah1, eta2


def get_velocities_from_mitgcm(ds_mitgcm, h1):
    u1 = ds_mitgcm.UVEL.sel(Z=slice(0,h1)).mean(dim=['Z'])
    v1 = ds_mitgcm.VVEL.sel(Z=slice(0,h1)).mean(dim=['Z'])
    u2 = ds_mitgcm.UVEL.sel(Z=slice(h1, None)).mean(dim=['Z'])
    v2 = ds_mitgcm.VVEL.sel(Z=slice(h1, None)).mean(dim=['Z'])

    return (u1,v1,u2,v2)


def compute_signed_hvsquare(aligned_u, aligned_v, h_thermo, h_fact=2):
    u1_sign = np.sign(aligned_u.where(aligned_u.Z >= h_thermo, np.nan).mean(dim=['Z']))
    v1_sign = np.sign(aligned_v.where(aligned_u.Z >= h_thermo, np.nan).mean(dim=['Z']))

    u1 = aligned_u.where(u1_sign *aligned_u >= 0, np.nan).where(aligned_u.Z >= h_thermo*h_fact, np.nan)
    v1 = aligned_v.where(v1_sign *aligned_v >= 0, np.nan).where(aligned_u.Z >= h_thermo*h_fact, np.nan)
    u2 = aligned_u.where(u1_sign *aligned_u < 0, np.nan)
    v2 = aligned_v.where(v1_sign *aligned_v < 0, np.nan)

    hu1square_sum = u1_sign * ((u1**2) * aligned_u.drF).sum(dim=['Z'])
    hv1square_sum = v1_sign * ((v1**2) * aligned_u.drF).sum(dim=['Z'])
    hu2square_sum = -u1_sign * ((u2**2) * aligned_u.drF).sum(dim=['Z'])
    hv2square_sum = -v1_sign * ((v2**2) * aligned_u.drF).sum(dim=['Z'])

    return (hu1square_sum, hv1square_sum, hu2square_sum, hv2square_sum)


def coarsen_mitgcm_results(coarse_fact, h1, eta2, velocities):
    (u1,v1,u2,v2) = velocities
    if coarse_fact == 1:
        return (h1.where(h1!=0, np.nan),
                eta2.where(eta2!=0, np.nan), 
                u1.where(u1!=0, np.nan), 
                v1.where(v1!=0, np.nan), 
                u2.where(u2!=0, np.nan), 
                v2.where(v2!=0, np.nan))
    
    h1_coarse = h1.where(h1!=0, np.nan).coarsen(YC=coarse_fact, XC=coarse_fact, boundary='trim').mean()
    eta2_coarse = eta2.where(eta2!=0, np.nan).coarsen(YC=coarse_fact, XC=coarse_fact, boundary='trim').mean()
    u1_coarse = u1.where(u1!=0, np.nan).coarsen(YC=coarse_fact, XC=coarse_fact, boundary='trim').mean()
    v1_coarse = v1.where(v1!=0, np.nan).coarsen(YC=coarse_fact, XC=coarse_fact, boundary='trim').mean()
    u2_coarse = u2.where(u2!=0, np.nan).coarsen(YC=coarse_fact, XC=coarse_fact, boundary='trim').mean()
    v2_coarse = v2.where(v2!=0, np.nan).coarsen(YC=coarse_fact, XC=coarse_fact, boundary='trim').mean()

    return (h1_coarse, eta2_coarse, u1_coarse, v1_coarse, u2_coarse, v2_coarse)


def get_xi_variables_from_mitgcm(ds_crop, aligned_u, aligned_v, h1, coarsening_fact, H1, H2):
    h_thermo_highres, eta2_highres = get_eta2_from_mitgcm(ds_crop.THETA, h1)
    hvsquare_mitgcm = compute_signed_hvsquare(aligned_u, aligned_v, h_thermo_highres)

    print('Coarsening MITgcm results...')
    (h1_coarse, eta2_coarse, hu1square_sum, hv1square_sum, hu2square_sum, hv2square_sum) = coarsen_mitgcm_results(coarsening_fact, h_thermo_highres, eta2_highres, hvsquare_mitgcm)

    xr_h1 = xr.DataArray(data=H1,
                        dims=("YC", "XC"),
                        coords={
                            "YC": eta2_coarse.YC,
                            "XC": eta2_coarse.XC
                        },
                        name="h1")
    xr_h2= xr.DataArray(data=H2,
                        dims=("YC", "XC"),
                        coords={
                            "YC": eta2_coarse.YC,
                            "XC": eta2_coarse.XC
                        },
                        name="h2")
    
    u1_corr = np.sign(hu1square_sum) * np.sqrt(np.abs(hu1square_sum / xr_h1))
    v1_corr = np.sign(hv1square_sum) * np.sqrt(np.abs(hv1square_sum / xr_h1))
    u2_corr = np.sign(hu2square_sum) * np.sqrt(np.abs(hu2square_sum / xr_h2))
    v2_corr = np.sign(hv2square_sum) * np.sqrt(np.abs(hv2square_sum / xr_h2))

    return (eta2_coarse, u1_corr, v1_corr, u2_corr, v2_corr)


def build_xi(indices, eta2, u1, v1, u2, v2, xitilde, results):
    xi = np.zeros(len(xitilde))
    k=indices['ieta2']-1
    for i in range(results['mesh']['neta']):
        for j in range(results['mesh']['meta']):
            if results['mesh']['nodos_eta2'][i,j] !=0:
                k += 1
                xi[k] = eta2[i,j]

    k=indices['iniu1']-1
    s=-1
    for i in range(results['mesh']['nu']):
        for j in range(results['mesh']['mu']):
            if results['mesh']['nodos_u1'][i,j] !=0:
                s += 1
                if results['mesh']['CBu1'][s] != 0:
                    k += 1
                    xi[k] = u1[i,j]

    k=indices['iniv1']-1
    s=-1
    for i in range(results['mesh']['nv']):
        for j in range(results['mesh']['mv']):
            if results['mesh']['nodos_v1'][i,j] !=0:
                s += 1
                if results['mesh']['CBv1'][s] != 0:
                    k += 1
                    xi[k] = v1[i,j]

    k=indices['iniu2']-1
    s=-1
    for i in range(results['mesh']['nu']):
        for j in range(results['mesh']['mu']):
            if results['mesh']['nodos_u2'][i,j] !=0:
                s += 1
                if results['mesh']['CBu2'][s] != 0:
                    k += 1
                    xi[k] = u2[i,j]

    k=indices['iniv2']-1
    s=-1
    for i in range(results['mesh']['nv']):
        for j in range(results['mesh']['mv']):
            if results['mesh']['nodos_v2'][i,j] !=0:
                s += 1
                if results['mesh']['CBv2'][s] != 0:
                    k += 1
                    xi[k] = v2[i,j]

    return xi