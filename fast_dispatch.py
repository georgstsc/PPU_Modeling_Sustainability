"""
Light-weight 15-min Swiss dispatch
author : you
date   : 2025-06-XX
"""
import numpy as np
import pandas as pd
from numba import njit, prange
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------------------------------------------
# 1.  CONSTANTS
# ------------------------------------------------------------------
ETA = {   # chain efficiency lookup (same as old CSV)
    'PV':1.0, 'WD_ON':1.0, 'WD_OFF':1.0, 'HYD_R':0.9, 'HYD_S':0.87,
    'PHS':0.82, 'H2_G':0.48, 'H2P_G':0.48, 'H2P_L':0.48, 'THERM':0.58,
    'THERM_G':0.56, 'BIO_OIL_ICE':0.42, 'BIO_WOOD':0.35, 'SOL_STEAM':0.45,
    'SOL_SALT':0.45, 'SYN_FT':0.52, 'SYN_METH':0.58, 'NH3_P':0.48
}
COST = {  # CHF/kWh variable cost
    'PV':0.03, 'WD_ON':0.025, 'WD_OFF':0.028, 'HYD_R':0.015, 'HYD_S':0.018,
    'PHS':0.025, 'H2_G':0.12, 'THERM':0.095, 'BIO_OIL_ICE':0.11,
    'BIO_WOOD':0.09, 'SOL_STEAM':0.10, 'SYN_FT':0.13, 'NH3_P':0.11
}

# ------------------------------------------------------------------
# 2.  DATA LOADING  (minimal, NumPy only)
# ------------------------------------------------------------------
def load_ts(data_dir='data'):
    """Return demand, spot, ror, solar, wind  -> all numpy float64 (T,)"""
    dem = pd.read_csv(Path(data_dir)/'monthly_hourly_load_values_2024.csv', sep='\t')
    dem = dem[dem.CountryCode=='CH'].copy()
    dem['datetime'] = pd.to_datetime(dem.DateUTC, dayfirst=True)
    dem = dem.set_index('datetime').Value.resample('15min').interpolate()
    demand = dem.to_numpy()

    spot = pd.read_csv(Path(data_dir)/'spot_price_hourly.csv', parse_dates=['time']
                      ).set_index('time').price.resample('15min').interpolate()
    price = spot.reindex(dem.index, method='nearest').to_numpy()

    ror  = pd.read_csv(Path(data_dir)/'water_quarterly_ror_2024.csv',
                      parse_dates=['timestamp']).set_index('timestamp').RoR_MW
    ror  = ror.reindex(dem.index, method='nearest').fillna(method='ffill').to_numpy()

    # solar incidence  (hourly -> 15 min)
    solar = _incidence_matrix(Path(data_dir)/'solar_incidence_hourly_2024.csv', dem.index)
    wind  = _incidence_matrix(Path(data_dir)/'wind_incidence_hourly_2024.csv',  dem.index)
    return demand, price, ror, solar, wind

def _incidence_matrix(csv, target_index):
    """return (T,N) array with incidence for N locations, re-indexed to 15-min"""
    df = pd.read_csv(csv, header=None)
    lat = df.iloc[0,1:].astype(float).values
    lon = df.iloc[1,1:].astype(float).values
    times = pd.to_datetime(df.iloc[3:,0])
    data  = df.iloc[3:,1:].astype(float).values
    df = pd.DataFrame(data, index=times, columns=range(len(lat)))
    df = df.resample('15min').interpolate().reindex(target_index, method='nearest')
    return df.to_numpy()

# ------------------------------------------------------------------
# 3.  PPU CONFIG  ->  lightweight structured array
# ------------------------------------------------------------------
ppu_dtype = np.dtype([
    ('name','U16'),  ('type','U12'),  ('kind','U8'),      # PV, WD_ON ...
    ('eta','f8'),    ('cost','f8'),   ('cap_MW','f8'),    # 1 GW default
    ('extract_sto','i4'), ('input_sto','i4'),  # bit-mask
    ('loc_rank','i4'),    ('lat','f8'), ('lon','f8')
])

def build_ppu_array(ppu_counts, solar_top=20, wind_top=20):
    """return (n_ppu,) structured array with one row per *instance*"""
    items = []
    idx = 0
    for kind,cnt in ppu_counts.items():
        eta  = ETA.get(kind, 0.5)
        cost = COST.get(kind, 0.10)
        for inst in range(cnt):
            name = f'{kind}_{inst+1}'
            lat=lon=rank=0
            if kind=='PV':
                rank = (idx % solar_top) +1
                lat,lon = 46.8, 8.2      # dummy – use real ranking if needed
            elif kind in ('WD_ON','WD_OFF'):
                rank = (idx % wind_top) +1
                lat,lon = 47.0, 7.5
            items.append((name, kind, 'Incidence' if kind in {'PV','WD_ON','WD_OFF','HYD_R'} else
                               'Store' if kind in {'PHS','H2_G','H2_L','SYN_FT','SYN_METH','NH3_FULL','SOL_SALT'} else
                               'Flex', eta, cost, 1000., 0,0, rank, lat,lon))
            idx +=1
    return np.array(items, dtype=ppu_dtype)

# ------------------------------------------------------------------
# 4.  STORAGE  ->  structured array + bit-masks
# ------------------------------------------------------------------
sto_list = ['Lake','Fuel Tank','H2 Storage UG 200bar','Liquid storage',
            'Solar concentrator salt','Biooil','Palm oil','Biogas (50% CH4)',
            'CH4 storage 200bar','Ammonia storage']
sto2idx = {s:i for i,s in enumerate(sto_list)}

sto_dtype = np.dtype([
    ('name','U22'),  ('max_MWh','f8'), ('SoC','f8'), ('target','f8'),
    ('extract_mask','i8'), ('input_mask','i8')
])

def build_storage_array(raw, ppu_array):
    """raw = list of dicts (your old raw_energy_storage)"""
    arr = np.zeros(len(raw), dtype=sto_dtype)
    for i,d in enumerate(raw):
        arr[i] = (d['storage'], d['value'], d['current_value']/d['value'],
                  d['target_SoC'], 0,0)
    # fill bit-masks
    for k,ppu in enumerate(ppu_array):
        for sname in ppu['extract_sto']:
            if sname in sto2idx:
                arr[sto2idx[sname]]['extract_mask'] |= 1<<k
        for sname in ppu['input_sto']:
            if sname in sto2idx:
                arr[sto2idx[sname]]['input_mask']   |= 1<<k
    return arr

# ------------------------------------------------------------------
# 5.  VECTORISED DISPATCH KERNEL  (Numba)
# ------------------------------------------------------------------
@njit(cache=True)
def _dispatch(dem, price, ror, solar, wind, ppu, sto,
              history_every=96*4, store_history=False):
    T, Nppu = dem.size, ppu.size
    phi_bar = 0.0
    # pre-allocate light history
    hist_p = {i:[(0,0.)] for i in range(Nppu)} if store_history else None
    sto_hist = np.zeros((len(sto), (T//history_every)+1)) if store_history else None

    for t in range(T):
        # 1. incidence
        solar_av = solar[t].mean() if solar.size else 0.0
        wind_av  = wind[t].mean()  if wind.size else 0.0
        grid_MW  = ror[t] + solar_av + wind_av

        # 2. imbalance
        overflow = dem[t] - grid_MW
        phi_bar  = 0.8*phi_bar + 0.2*overflow

        # 3. storage indices
        d_stor = np.tanh((sto['SoC'] - sto['target'])/0.05)
        u_dis  = np.tanh(phi_bar/5000.)
        u_chg  = np.tanh(-phi_bar/5000.) if phi_bar<0 else 0.

        # 4. merit order
        if overflow > 1.0:        # deficit
            merit = ppu['cost'].copy()
            merit[ppu['kind']=='Store'] = 1e6
            order = np.argsort(merit)
            rem   = overflow
            for k in order:
                if rem<1: break
                avail = 0.0
                for s in range(sto.size):
                    if (sto[s]['extract_mask']>>k)&1:
                        avail += sto[s]['max_MWh']*sto[s]['SoC']*sto[s]['eta']
                out  = min(rem, 1000/ppu[k]['eta'], avail*ppu[k]['eta'])
                rem -= out
                if store_history: hist_p[k].append((t, out))
                # update SoC
                for s in range(sto.size):
                    if (sto[s]['extract_mask']>>k)&1 and avail>0:
                        frac = (sto[s]['max_MWh']*sto[s]['SoC'])/avail
                        sto[s]['SoC'] -= frac*out/ppu[k]['eta']/sto[s]['max_MWh']
        elif overflow < -1.0:     # surplus
            merit = -ppu['cost'].copy()
            merit[ppu['kind']=='Flex'] = 1e6
            order = np.argsort(merit)
            rem   = -overflow
            for k in order:
                if rem<1: break
                space = 0.0
                for s in range(sto.size):
                    if (sto[s]['input_mask']>>k)&1:
                        space += sto[s]['max_MWh']*(1-sto[s]['SoC'])
                inp = min(rem, 1000*ppu[k]['eta'], space)
                rem -= inp
                if store_history: hist_p[k].append((t, -inp))
                for s in range(sto.size):
                    if (sto[s]['input_mask']>>k)&1 and space>0:
                        frac = (sto[s]['max_MWh']*(1-sto[s]['SoC']))/space
                        sto[s]['SoC'] += frac*inp/sto[s]['max_MWh']
        # store history
        if store_history and t%history_every==0:
            sto_hist[:, t//history_every] = sto['SoC']
    return sto, hist_p, phi_bar

# ------------------------------------------------------------------
# 6.  SINGLE CALL  (replaces run_complete_pipeline)
# ------------------------------------------------------------------
def run_fast_pipeline(ppu_counts, raw_storage,
                      history_every=96*4, data_dir='data'):
    dem, prc, ror, sol, win = load_ts(data_dir)
    ppu = build_ppu_array(ppu_counts)
    sto = build_storage_array(raw_storage, ppu)
    sto, hist, phi = _dispatch(dem, prc, ror, sol, win, ppu, sto,
                               history_every, store_history=True)
    # metrics
    cost_ppu = sum(sum(mw for _,mw in lst)*0.25*ppu[k]['cost']
                   for k,lst in hist.items()) if hist else 0
    cost_spot = (dem*prc*0.25).sum()
    return {
        'sto_final': sto,
        'history': hist,
        'SoC_series': hist['sto'] if 'sto' in hist else None,
        'total_cost': cost_ppu,
        'spot_cost': cost_spot,
        'savings': cost_spot - cost_ppu,
        'volatility_pct': (prc.groupby(np.arange(len(prc))//96).mean().std()
                           / prc.groupby(np.arange(len(prc))//96).mean().mean())*100
    }