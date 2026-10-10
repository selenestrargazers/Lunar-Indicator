"""
Lunar south-pole landing-site screening engine (corrected version).

What changed versus the Gemini script
-------------------------------------
* Terrain: real georeferenced LOLA GeoTIFF (polar stereographic) read through
  rasterio + pyproj. Lat/lon -> pixel is done with the projection, not with an
  assumed "row 0 = north" grid.
* Horizon: ray-marched over kilometres of terrain WITH lunar curvature, so distant
  crater rims and mountains (which decide lighting at the pole) are included.
* Sun is a disk: upper limb (+0.267 deg) is used instead of the centre.
* Slope: plane fit over a lander-scale footprint (never below ~3 DEM pixels).
* Ephemeris: SPICE (fast, offline, vectorised) for whole-area scans; JPL Horizons
  kept only as a single-site cross-check. Topocentric parallax of Earth is included.
* Time window: full lunar cycle by default; reports longest dark / Earth-link gaps.
* Thresholds are ASSUMPTIONS in `Config`, not NASA requirements.

Needs: numpy pandas scipy rasterio pyproj spiceypy (+ astroquery for the check).
NOTE: SPICE/Horizons parts need internet or local kernels; test them yourself.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

R_M = 1_737_400.0            # LOLA reference sphere radius (m)
SUN_SEMIDIAMETER_DEG = 0.267


@dataclass
class Config:
    site_height_m: float = 2.0          # antenna / panel height above ground
    sun_min_elev_deg: float = 0.0
    earth_min_elev_deg: float = 0.0     # extra antenna limit above the terrain horizon
    max_horizon_dist_m: float = 20_000.0
    az_step_deg: float = 1.0
    footprint_radius_m: float = 15.0
    slope_max_deg: float = 10.0         # assumption
    solar_min_pct: float = 40.0         # assumption
    dte_min_pct: float = 50.0           # assumption
    p_max_w: float = 1000.0
    panel_az_deg: float = 0.0           # vertical panel normal azimuth (0 = north)


# ---------------------------------------------------------------- terrain ----
def _destination(lat_deg, lon_deg, az_rad, ang_rad):
    """Great-circle destination on a sphere. Returns (lat_deg, lon_deg)."""
    phi, lam = np.radians(lat_deg), np.radians(lon_deg)
    lat2 = np.arcsin(np.sin(phi) * np.cos(ang_rad)
                     + np.cos(phi) * np.sin(ang_rad) * np.cos(az_rad))
    lon2 = lam + np.arctan2(np.sin(az_rad) * np.sin(ang_rad) * np.cos(phi),
                            np.cos(ang_rad) - np.sin(phi) * np.sin(lat2))
    return np.degrees(lat2), np.degrees(lon2)


class LunarDEM:
    """Georeferenced DEM (metres). Elevation is height above the 1737.4 km sphere."""

    def __init__(self, path, height_scale=1.0):
        import rasterio
        from pyproj import CRS, Transformer
        self.ds = rasterio.open(path)
        self.inv = ~self.ds.transform
        self.res = abs(self.ds.transform.a)           # metres / pixel (metric CRS)
        self.scale = height_scale                     # e.g. 1000 if file is in km
        moon = CRS.from_proj4(f"+proj=longlat +R={R_M} +no_defs")
        self.fwd = Transformer.from_crs(moon, self.ds.crs, always_xy=True)

    def sample(self, lat, lon):
        """Bilinear elevation at arrays of lat/lon (same shape). NaN outside data."""
        from rasterio.windows import Window
        from scipy.ndimage import map_coordinates
        lat, lon = np.asarray(lat, float), np.asarray(lon, float)
        x, y = self.fwd.transform(lon, lat)
        cols, rows = self.inv * (np.asarray(x), np.asarray(y))
        cols, rows = cols - 0.5, rows - 0.5           # pixel-centre index space
        r0, c0 = int(np.floor(np.nanmin(rows))) - 1, int(np.floor(np.nanmin(cols))) - 1
        r1, c1 = int(np.ceil(np.nanmax(rows))) + 1, int(np.ceil(np.nanmax(cols))) + 1
        sub = self.ds.read(1, window=Window(c0, r0, c1 - c0 + 1, r1 - r0 + 1),
                           boundless=True, fill_value=np.nan, out_dtype="float64")
        nod = self.ds.nodata
        if nod is not None and not np.isnan(nod):
            sub[sub == nod] = np.nan
        return self.scale * map_coordinates(sub, [rows - r0, cols - c0],
                                            order=1, mode="constant", cval=np.nan)


def horizon_profile(dem, lat, lon, cfg: Config):
    """Terrain horizon elevation (deg) for azimuths 0..360 (clockwise from north)."""
    az_deg = np.arange(0.0, 360.0, cfg.az_step_deg)
    az = np.radians(az_deg)[:, None]
    dstep = max(dem.res, 20.0)
    d = np.arange(dstep, cfg.max_horizon_dist_m + dstep, dstep)[None, :]
    th = d / R_M
    la, lo = _destination(lat, lon, az, th)
    z = dem.sample(la, lo)
    z0 = float(dem.sample(np.array([lat]), np.array([lon]))[0])
    r0 = R_M + z0 + cfg.site_height_m
    r = R_M + z
    ang = np.degrees(np.arctan2(r * np.cos(th) - r0, r * np.sin(th)))  # includes curvature
    with np.errstate(all="ignore"):
        hor = np.nanmax(np.where(np.isnan(ang), -90.0, ang), axis=1)
    return az_deg, hor


def footprint_slope_deg(dem, lat, lon, cfg: Config):
    """Least-squares plane slope over a lander-scale footprint (>= ~3 pixels wide)."""
    rad = max(cfg.footprint_radius_m, 1.5 * dem.res)
    g = np.arange(-rad, rad + dem.res, dem.res)
    e, n = np.meshgrid(g, g)
    m = e ** 2 + n ** 2 <= rad ** 2 + 1e-9
    la, lo = _destination(lat, lon, np.arctan2(e, n), np.hypot(e, n) / R_M)
    z = dem.sample(la, lo)
    ok = m & ~np.isnan(z)
    if ok.sum() < 4:
        return float("nan")
    A = np.c_[e[ok], n[ok], np.ones(ok.sum())]
    (a, b, _), *_ = np.linalg.lstsq(A, z[ok], rcond=None)
    return float(np.degrees(np.arctan(np.hypot(a, b))))


# -------------------------------------------------------------- ephemeris ----
def moon_fixed_vectors(start, stop, step_s, kernels):
    """Sun & Earth position vectors (km) from Moon centre in the MOON_ME frame.
    kernels: list of paths: naif0012.tls, de440s.bsp, moon_pa_de440_*.bpc,
    moon_de44x_*.tf  (download from NASA NAIF generic_kernels)."""
    import spiceypy as sp
    for k in kernels:
        sp.furnsh(str(k))
    ets = np.arange(sp.str2et(start), sp.str2et(stop), step_s)
    get = lambda body: np.array([sp.spkpos(body, t, "MOON_ME", "LT+S", "MOON")[0] for t in ets])
    times = pd.to_datetime([sp.et2utc(t, "ISOC", 0) for t in ets])
    return times, get("SUN"), get("EARTH")


def topocentric_azel(vec_km, lat, lon, site_radius_m):
    """Az (deg from north, clockwise) / El (deg) of a body-fixed vector seen from a
    surface site. Subtracting the site position gives correct parallax."""
    phi, lam = np.radians(lat), np.radians(lon)
    up = np.array([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)])
    east = np.array([-np.sin(lam), np.cos(lam), 0.0])
    north = np.array([-np.sin(phi) * np.cos(lam), -np.sin(phi) * np.sin(lam), np.cos(phi)])
    v = vec_km - up * site_radius_m / 1000.0
    dist = np.linalg.norm(v, axis=1)
    e, n, u = v @ east, v @ north, v @ up
    return np.degrees(np.arctan2(e, n)) % 360.0, np.degrees(np.arcsin(u / dist))


def horizons_azel(target, lat, lon, start, stop, step="1h"):
    """Single-site cross-check against JPL Horizons (observer ON the Moon)."""
    from astroquery.jplhorizons import Horizons
    loc = {"lon": lon, "lat": lat, "elevation": 0.0, "body": 301}
    eph = Horizons(id=target, location=loc, id_type="majorbody",
                   epochs={"start": start, "stop": stop, "step": step}).ephemerides()
    return eph[["datetime_str", "AZ", "EL"]].to_pandas()


# ------------------------------------------------------------- assessment ----
def _longest_run_h(mask, step_h):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best * step_h


def evaluate_site(dem, lat, lon, times, sun_vec, earth_vec, cfg: Config = Config()):
    lat = max(min(lat, 89.999), -89.999)
    step_h = (times[1] - times[0]).total_seconds() / 3600.0
    z0 = float(dem.sample(np.array([lat]), np.array([lon]))[0])
    if np.isnan(z0):
        raise ValueError('site is outside the DEM coverage')
    rs = R_M + z0 + cfg.site_height_m
    s_az, s_el = topocentric_azel(sun_vec, lat, lon, rs)
    e_az, e_el = topocentric_azel(earth_vec, lat, lon, rs)
    az_grid, hor = horizon_profile(dem, lat, lon, cfg)
    h_sun = np.interp(s_az, az_grid, hor, period=360.0)
    h_earth = np.interp(e_az, az_grid, hor, period=360.0)

    lit = (s_el + SUN_SEMIDIAMETER_DEG > np.maximum(h_sun, cfg.sun_min_elev_deg))
    dte = e_el > np.maximum(h_earth, 0.0) + cfg.earth_min_elev_deg
    golden = lit & dte
    cos_i = np.clip(np.cos(np.radians(s_el)) * np.cos(np.radians(s_az - cfg.panel_az_deg)), 0, None)
    power = np.where(lit, cfg.p_max_w * cos_i, 0.0)
    slope = footprint_slope_deg(dem, lat, lon, cfg)

    solar_pct, dte_pct, gold_pct = 100 * lit.mean(), 100 * dte.mean(), 100 * golden.mean()
    viable = (solar_pct >= cfg.solar_min_pct and dte_pct >= cfg.dte_min_pct
              and not np.isnan(slope) and slope <= cfg.slope_max_deg)
    tl = pd.DataFrame({"time": times, "sun_az": s_az, "sun_el": s_el, "sun_horizon": h_sun,
                       "earth_az": e_az, "earth_el": e_el, "earth_horizon": h_earth,
                       "lit": lit, "dte": dte, "golden": golden, "power_w": power})
    return {"lat": lat, "lon": lon, "elevation_m": round(z0, 1),
            "slope_deg": round(slope, 2), "solar_pct": round(float(solar_pct), 1),
            "dte_pct": round(float(dte_pct), 1), "golden_pct": round(float(gold_pct), 1),
            "mean_power_w": round(float(power.mean()), 1),
            "longest_dark_h": _longest_run_h(~lit, step_h),
            "longest_no_earth_h": _longest_run_h(~dte, step_h),
            "longest_golden_h": _longest_run_h(golden, step_h),
            "viable": bool(viable), "timeline": tl}


def scan_area(dem, times, sun_vec, earth_vec, cfg: Config = Config(),
              lat_min=-90.0, lat_max=-88.0, spacing_km=5.0):
    """Evaluate a polar grid of candidate sites; returns a DataFrame sorted by score.
    Cost grows fast: 85-90S at 2 km is ~240k sites (far too slow). Start with a small
    region / coarse spacing, then refine around the best candidates."""
    rows = []
    for lat in np.arange(lat_max, lat_min, -spacing_km / (R_M / 1000 * np.pi / 180)):
        ring = 2 * np.pi * (R_M / 1000) * np.cos(np.radians(lat))
        n = max(1, int(ring / spacing_km))
        for lon in np.linspace(0, 360, n, endpoint=False):
            try:
                r = evaluate_site(dem, lat, lon, times, sun_vec, earth_vec, cfg)
            except Exception:
                continue
            r.pop("timeline")
            rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["viable", "golden_pct"], ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------- demo ----
def make_demo_dem(path="demo_dem.tif", n=1000, res=40.0):
    """SYNTHETIC 40 km x 40 km polar DEM (fake hills) so you can test without downloads."""
    import rasterio
    from rasterio.transform import from_origin
    from pyproj import CRS
    x0 = -n * res / 2
    xs = x0 + res * (np.arange(n) + 0.5)
    X, Y = np.meshgrid(xs, -xs)
    rng = np.random.default_rng(1)
    z = np.zeros((n, n))
    for _ in range(25):
        cx, cy = rng.uniform(-15000, 15000, 2)
        z += rng.uniform(50, 400) * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * rng.uniform(800, 3000) ** 2))
    crs = CRS.from_proj4(f"+proj=stere +lat_0=-90 +lon_0=0 +k=1 +R={R_M} +units=m +no_defs")
    with rasterio.open(path, "w", driver="GTiff", height=n, width=n, count=1, dtype="float32",
                       crs=crs, transform=from_origin(x0, -x0, res, res), nodata=-9999) as d:
        d.write(z.astype("float32"), 1)
    return path


def demo_vectors(start, stop, step_s):
    """APPROXIMATE analytic Sun/Earth vectors (no SPICE). For testing only."""
    t0, t1 = pd.Timestamp(start), pd.Timestamp(stop)
    times = pd.date_range(t0, t1, freq=f"{step_s}s", inclusive="left")
    d = (times - pd.Timestamp("2000-01-01 12:00")).total_seconds().to_numpy() / 86400.0
    w = lambda x: np.mod(x, 360.0)
    g = np.radians(357.528 + 0.9856003 * d)
    Ls = w(280.46 + 0.9856474 * d + 1.915 * np.sin(g) + 0.02 * np.sin(2 * g))
    D, M = w(297.8502 + 12.190749 * d), np.radians(w(134.9634 + 13.064993 * d))
    F, Om = np.radians(w(93.2721 + 13.22935 * d)), 125.04 - 0.052954 * d
    e_lon, e_lat = 7.6 * np.sin(M), -6.7 * np.sin(F)
    s_lon, s_lat = e_lon - (D - 180.0), 1.54 * np.sin(np.radians(Ls - Om))
    def vec(lat, lon, dist):
        la, lo = np.radians(lat), np.radians(lon)
        return dist * np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]
    return times, vec(s_lat, s_lon, 1.496e8), vec(e_lat, e_lon, 384400.0)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="run on a synthetic DEM, no downloads")
    ap.add_argument("--dem", help="LOLA south-pole GeoTIFF")
    ap.add_argument("--kernels", nargs="+", help="SPICE kernel files")
    ap.add_argument("--lat", type=float, default=-89.5)
    ap.add_argument("--lon", type=float, default=0.0)
    ap.add_argument("--start", default="2028-10-01")
    ap.add_argument("--stop", default="2028-10-31")
    ap.add_argument("--scan", action="store_true")
    a = ap.parse_args()
    if a.demo:
        print("DEMO MODE: synthetic terrain + approximate ephemeris. Not real results.")
        dem = LunarDEM(make_demo_dem())
        t, sun, earth = demo_vectors(a.start, a.stop, 3600)
        if a.lat == -89.5:
            a.lat = -89.95
        scan_kw = dict(lat_max=-89.5, lat_min=-90.0, spacing_km=2.0)
    else:
        if not (a.dem and a.kernels):
            ap.error("give --dem and --kernels, or use --demo")
        dem = LunarDEM(a.dem)
        t, sun, earth = moon_fixed_vectors(a.start, a.stop, 3600, a.kernels)
        scan_kw = {}
    if a.scan:
        print(scan_area(dem, t, sun, earth, **scan_kw).head(20).to_string())
    else:
        r = evaluate_site(dem, a.lat, a.lon, t, sun, earth)
        r.pop("timeline")
        print(r)