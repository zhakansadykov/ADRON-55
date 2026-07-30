import logging
import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
import astropy.units as u

logger = logging.getLogger(__name__)

def calibrate_energy(adc_counts: float, cfg: dict) -> float:
    """
    Convert ADC counts to MeV.
    Uses: E_MeV = ADC * (ADC_to_mV / MIP_mV) * MIP_MeV
    """
    cal_cfg = cfg.get('calibration', {})
    adc_to_mv = cal_cfg.get('adc_to_mv', 1.0)
    mip_in_mv = cal_cfg.get('mip_in_mv', 0.39)
    mip_in_mev = cal_cfg.get('mip_in_mev', 2.0)
    
    mv = adc_counts * adc_to_mv
    mips = mv / mip_in_mv
    mev = mips * mip_in_mev
    return mev

def altaz_to_radec(zenith_deg: float, azimuth_deg: float, event_time: str, cfg: dict):
    """
    Convert local coordinates (zenith, azimuth) to equatorial ones (RA, Dec).
    Uses Astropy with the event time and the site coordinates.
    """
    loc_cfg = cfg['location']
    
    try:
        # Site of the detector
        location = EarthLocation(
            lat=loc_cfg['latitude'] * u.deg,
            lon=loc_cfg['longitude'] * u.deg,
            height=loc_cfg['altitude'] * u.m
        )
        
        # Event time, as stored in the catalogue: '2021-01-06 11:09:24'
        time = Time(event_time, format='iso', scale='utc')
        
        # Zenith angle to altitude above the horizon
        altitude_deg = 90.0 - zenith_deg
        
        # Orientation of the detector X axis relative to north
        az_corrected = (azimuth_deg + loc_cfg.get('detector_angle_offset', 0)) % 360.0
        
        # Horizontal coordinates
        altaz = AltAz(
            alt=altitude_deg * u.deg, 
            az=az_corrected * u.deg,
            obstime=time, 
            location=location
        )
        
        # Transform to ICRS (RA, Dec)
        skycoord = SkyCoord(altaz).transform_to('icrs')
        
        return skycoord.ra.deg, skycoord.dec.deg
        
    except Exception as e:
        logger.error(f"AltAz -> RA/Dec conversion failed for time {event_time}: {e}")
        return None, None

# === HELPERS FOR MULTIPLE SCATTERING ===

def get_cumulative_rad_length(cfg: dict) -> dict:
    """
    Cumulative radiation length (X/X0) above each plane, used to weight the fit
    by the accumulated multiple scattering.
    Returns {layer_idx (0-7): cumulative X/X0}.

    before_row_N in the configuration is either a single layer
    {material, thickness_mm} or a list of them (for example
    Pb 220 + air 2200 + Pb 130 + Fe 100 above plane 3).
    """
    abs_cfg = cfg.get('absorber', {})
    materials = abs_cfg.get('materials', {})
    layers = abs_cfg.get('layers', {})
    
    cum_x0 = 0.0
    row_rad_lengths = {}
    
    # Top to bottom: plane 1 (idx 0) to plane 8 (idx 7)
    for i in range(1, 9):
        entry = layers.get(f"before_row_{i}")
        entry_layers = entry if isinstance(entry, list) else ([entry] if entry else [])
        for layer in entry_layers:
            cum_x0 += layer['thickness_mm'] / materials[layer['material']]['X0_mm']
        
        # Store the cumulative radiation length for this plane
        row_rad_lengths[i-1] = max(0.01, cum_x0)  # guard against division by zero
        
    return row_rad_lengths