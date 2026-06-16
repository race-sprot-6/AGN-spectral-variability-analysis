from __future__ import annotations

import math
import re
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from scipy.optimize import curve_fit

from ppxf.ppxf import ppxf
from ppxf import ppxf_util as util
import ppxf.sps_util as lib

try:
    from dl import queryClient as qc
    HAVE_DATALAB = True
except Exception:
    qc = None
    HAVE_DATALAB = False


warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================
# НАСТРОЙКИ
# ============================================================

ROOT_DIR = Path(r"C:\Users\IvanK\Astro\Course5\downloaded_spectra")
PPXF_SPS_FILE = Path(r"C:\Users\IvanK\Astro\Pract_2025\ppxf_sps\spectra_emiles_9.0.npz")

WRITE_RESULTS_TO_FITS_HEADER = True
OVERWRITE_PLOTS = True
MC_SAMPLES = 1000
DEFAULT_FWHM_GAL = 7.0
PPXF_NOISE_LEVEL = 0.0047
REMOTE_SQL_CHUNK = 100

# Если True, то при провале pPXF будет сделана попытка фита прямо по исходному спектру
# без вычитания звёздного населения. По умолчанию False, чтобы сохранить вашу процедуру.
FALLBACK_TO_DIRECT_SPECTRUM_IF_PPXF_FAIL = False

OBJECT_SUMMARY_NAME = "line_fluxes_vs_time.tsv"
MASTER_SUMMARY_NAME = "all_line_fluxes_vs_time.tsv"
REDSHIFT_CACHE_NAME = "_redshift_cache.tsv"


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def apply_article_axis_style(ax):
    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
    )
    ax.minorticks_on()

def safe_int(value) -> Optional[int]:
    try:
        if value is None or pd.isna(value):
            return None
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return None


def safe_float(value) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def strip_spectrum_suffix(path: Path) -> str:
    name = path.name
    lower = name.lower()
    for suf in [".fits.gz", ".fit.gz", ".fits", ".fit", ".fz"]:
        if lower.endswith(suf):
            return name[: -len(suf)]
    return path.stem


def list_spectrum_files(object_dir: Path) -> list[Path]:
    files = []
    for sub in ["sdss", "desi"]:
        subdir = object_dir / sub
        if not subdir.exists():
            continue
        for pattern in ["*.fits", "*.fit", "*.fits.gz", "*.fit.gz", "*.fz"]:
            files.extend(sorted(subdir.glob(pattern)))
    # уникализируем
    uniq = []
    seen = set()
    for p in files:
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def load_object_manifest(object_dir: Path) -> dict[str, dict]:
    manifest = object_dir / "spectra_manifest.tsv"
    if not manifest.exists():
        return {}

    try:
        df = pd.read_csv(manifest, sep="\t")
    except Exception:
        return {}

    out = {}
    for _, row in df.iterrows():
        local_file = row.get("local_file")
        if local_file is None or pd.isna(local_file):
            continue
        name = Path(str(local_file)).name
        out[name] = row.to_dict()
    return out


def parse_time_to_mjd(date_str: str) -> Optional[float]:
    if not date_str:
        return None

    candidates = [
        date_str.strip(),
        date_str.strip().replace(" ", "T"),
        date_str.strip().replace("+00", ""),
        date_str.strip().replace(" ", "T").replace("+00", ""),
    ]

    for s in candidates:
        if not s:
            continue
        try:
            return float(Time(s).mjd)
        except Exception:
            pass

    return None


def mjd_to_iso(mjd: Optional[float]) -> str:
    if mjd is None:
        return ""
    try:
        return Time(float(mjd), format="mjd").isot
    except Exception:
        return ""


def run_query(sql: str) -> pd.DataFrame:
    if not HAVE_DATALAB:
        return pd.DataFrame()

    df = qc.query(sql=sql, fmt="pandas")
    if df is None:
        return pd.DataFrame()

    df = pd.DataFrame(df)
    df.columns = [str(c).lower() for c in df.columns]
    return df


def load_redshift_cache(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}

    try:
        df = pd.read_csv(path, sep="\t")
    except Exception:
        return {}

    cache = {}
    for _, row in df.iterrows():
        specid = safe_int(row.get("specid"))
        if specid is None:
            continue
        cache[specid] = {
            "redshift": safe_float(row.get("redshift")),
            "release": str(row.get("release", "")) if row.get("release") is not None else "",
            "targetid": safe_int(row.get("targetid")),
        }
    return cache


def save_redshift_cache(path: Path, cache: dict[int, dict]) -> None:
    rows = []
    for specid, info in sorted(cache.items()):
        rows.append({
            "specid": specid,
            "redshift": info.get("redshift"),
            "release": info.get("release", ""),
            "targetid": info.get("targetid"),
        })
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def query_redshifts_from_sparcl(specids: list[int]) -> dict[int, dict]:
    specids = sorted(set(int(x) for x in specids if x is not None))
    if not specids or not HAVE_DATALAB:
        return {}

    out = {}

    sql_templates = [
        "SELECT specid, targetid, redshift, data_release FROM sparcl.main WHERE specid IN ({ids})",
        "SELECT specid, redshift, data_release FROM sparcl.main WHERE specid IN ({ids})",
        "SELECT specid, redshift FROM sparcl.main WHERE specid IN ({ids})",
    ]

    for i in range(0, len(specids), REMOTE_SQL_CHUNK):
        chunk = specids[i:i + REMOTE_SQL_CHUNK]
        ids = ",".join(str(x) for x in chunk)

        df = pd.DataFrame()
        for template in sql_templates:
            sql = template.format(ids=ids)
            try:
                df = run_query(sql)
                if not df.empty:
                    break
            except Exception:
                pass

        if df.empty:
            continue

        for _, row in df.iterrows():
            specid = safe_int(row.get("specid"))
            if specid is None:
                continue
            out[specid] = {
                "redshift": safe_float(row.get("redshift")),
                "release": str(row.get("data_release", "")) if "data_release" in row else "",
                "targetid": safe_int(row.get("targetid")),
            }

    return out


# ============================================================
# МОДЕЛИ ЛИНИЙ
# ============================================================

def double_gaussian(x, amp1, cen1, sigma1, amp2, cen2, sigma2, offset):
    return (
        amp1 * np.exp(-0.5 * ((x - cen1) / sigma1) ** 2) +
        amp2 * np.exp(-0.5 * ((x - cen2) / sigma2) ** 2) +
        offset
    )


def ha_model_full(x,
                  amp_narrow, cen_narrow, sigma_narrow,
                  amp_broad, cen_broad, sigma_broad,
                  amp_nii_6548, cen_nii_6548, sigma_nii_6548,
                  amp_nii_6583, cen_nii_6583, sigma_nii_6583,
                  offset):
    gauss_narrow = amp_narrow * np.exp(-0.5 * ((x - cen_narrow) / sigma_narrow) ** 2)
    gauss_broad = amp_broad * np.exp(-0.5 * ((x - cen_broad) / sigma_broad) ** 2)
    gauss_nii_6548 = amp_nii_6548 * np.exp(-0.5 * ((x - cen_nii_6548) / sigma_nii_6548) ** 2)
    gauss_nii_6583 = amp_nii_6583 * np.exp(-0.5 * ((x - cen_nii_6583) / sigma_nii_6583) ** 2)
    return gauss_narrow + gauss_broad + gauss_nii_6548 + gauss_nii_6583 + offset


# ============================================================
# СТРУКТУРЫ ДАННЫХ
# ============================================================

@dataclass
class SpectrumRecord:
    object_name: str
    path: Path
    survey: str
    release: str
    specid: Optional[int]
    targetid: Optional[int]
    date_obs: str
    date_mjd: Optional[float]
    redshift: Optional[float]
    redshift_source: str
    wave_obs: np.ndarray
    flux: np.ndarray
    ivar: Optional[np.ndarray]


@dataclass
class LineFitResult:
    line_name: str
    ok: bool
    reason: str
    wave: Optional[np.ndarray] = None
    flux: Optional[np.ndarray] = None
    orig_flux: Optional[np.ndarray] = None
    model: Optional[np.ndarray] = None
    resid: Optional[np.ndarray] = None
    broad_component: Optional[np.ndarray] = None
    narrow_component: Optional[np.ndarray] = None
    extra1_component: Optional[np.ndarray] = None
    extra2_component: Optional[np.ndarray] = None
    offset: Optional[float] = None
    params: Optional[np.ndarray] = None
    pcov: Optional[np.ndarray] = None
    broad_flux_det: Optional[float] = None
    broad_flux_mc_mean: Optional[float] = None
    broad_flux_mc_std: Optional[float] = None
    integration_mask: Optional[np.ndarray] = None


# ============================================================
# ЧТЕНИЕ СПЕКТРОВ
# ============================================================

def extract_manifest_date(manifest_row: Optional[dict]) -> tuple[str, Optional[float]]:
    if not manifest_row:
        return "", None

    date_obs = str(manifest_row.get("date_obs", "")).strip()
    if date_obs.lower() == "nan":
        date_obs = ""

    mjd = safe_float(manifest_row.get("mjd"))
    if mjd is None and date_obs:
        mjd = parse_time_to_mjd(date_obs)

    return date_obs, mjd


def extract_date_from_hdul(hdul: fits.HDUList) -> tuple[str, Optional[float]]:
    for h in hdul:
        hdr = h.header
        for key in ("DATE-OBS", "DATEOBS"):
            val = hdr.get(key)
            if val:
                s = str(val).strip()
                mjd = parse_time_to_mjd(s)
                return s, mjd

    for h in hdul:
        hdr = h.header
        for key in ("MJD", "MJD-OBS", "MJDOBS", "MEANMJD"):
            val = safe_float(hdr.get(key))
            if val is not None:
                return mjd_to_iso(val), val

    return "", None


def extract_id_from_filename(path: Path) -> Optional[int]:
    name = path.name
    patterns = [
        r"_desi_(\d+)\.(?:fits|fit)(?:\.gz)?$",
        r"desi_spec_(\d+)\.(?:fits|fit)(?:\.gz)?$",
        r"_sdss_(\d+)\.(?:fits|fit)(?:\.gz)?$",
    ]
    for p in patterns:
        m = re.search(p, name, flags=re.IGNORECASE)
        if m:
            return safe_int(m.group(1))
    return None


def extract_redshift_from_hdul(hdul: fits.HDUList) -> tuple[Optional[float], str]:
    header_keys = ["Z", "REDSHIFT", "Z_OBJ", "ZOBJ", "SPEC_Z"]
    for h in hdul:
        hdr = h.header
        for key in header_keys:
            z = safe_float(hdr.get(key))
            if z is not None:
                return z, f"header:{key}"

    # сначала пробуем табличные HDU с Z / REDSHIFT
    for h in hdul:
        if not isinstance(h, (fits.BinTableHDU, fits.TableHDU)) or h.data is None:
            continue
        cols = {str(c).lower(): c for c in h.columns.names}
        for cand in ["z", "redshift"]:
            if cand in cols and len(h.data) > 0:
                try:
                    z = safe_float(h.data[cols[cand]][0])
                    if z is not None:
                        return z, f"table:{h.name or 'TABLE'}:{cand}"
                except Exception:
                    pass

    return None, ""


def read_sdss_spectrum(hdul: fits.HDUList) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    for h in hdul:
        if not isinstance(h, (fits.BinTableHDU, fits.TableHDU)) or h.data is None:
            continue
        cols = {str(c).lower(): c for c in h.columns.names}
        if "flux" in cols and "loglam" in cols:
            flux = np.asarray(h.data[cols["flux"]], dtype=float)
            wave = 10 ** np.asarray(h.data[cols["loglam"]], dtype=float)
            ivar = np.asarray(h.data[cols["ivar"]], dtype=float) if "ivar" in cols else None
            return wave, flux, ivar

    raise ValueError("Не удалось распознать SDSS-спектр")


def read_desi_spectrum(hdul: fits.HDUList) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    name_to_idx = {h.name.upper(): i for i, h in enumerate(hdul)}

    if "WAVELENGTH" in name_to_idx and "FLUX" in name_to_idx:
        wave = np.asarray(hdul[name_to_idx["WAVELENGTH"]].data, dtype=float).ravel()
        flux = np.asarray(hdul[name_to_idx["FLUX"]].data, dtype=float).ravel()
        ivar = None
        if "IVAR" in name_to_idx:
            ivar = np.asarray(hdul[name_to_idx["IVAR"]].data, dtype=float).ravel()
        return wave, flux, ivar

    # поддержка старого формата: primary data shape = (>=2, N)
    if hdul[0].data is not None:
        arr = np.asarray(hdul[0].data, dtype=float)
        if arr.ndim == 2 and arr.shape[0] >= 2:
            wave = arr[0, :].astype(float)
            flux = arr[1, :].astype(float)
            ivar = None
            return wave, flux, ivar

    raise ValueError("Не удалось распознать DESI-спектр")


def load_spectrum_record(path: Path, object_name: str, manifest_row: Optional[dict]) -> SpectrumRecord:
    with fits.open(path, memmap=False) as hdul:
        hdr0 = hdul[0].header

        survey = ""
        if manifest_row and "survey" in manifest_row and not pd.isna(manifest_row["survey"]):
            survey = str(manifest_row["survey"]).strip().upper()
        elif hdr0.get("SURVEY"):
            survey = str(hdr0.get("SURVEY")).strip().upper()
        elif path.parent.name.lower() == "sdss":
            survey = "SDSS"
        elif path.parent.name.lower() == "desi":
            survey = "DESI"

        release = ""
        if manifest_row and "release" in manifest_row and not pd.isna(manifest_row["release"]):
            release = str(manifest_row["release"]).strip()
        elif hdr0.get("RELEASE"):
            release = str(hdr0.get("RELEASE")).strip()

        specid = safe_int(hdr0.get("SPECID"))
        if specid is None and manifest_row:
            specid = safe_int(manifest_row.get("specid"))
        if specid is None:
            specid = extract_id_from_filename(path)

        targetid = safe_int(hdr0.get("TARGETID"))
        if targetid is None and manifest_row:
            targetid = safe_int(manifest_row.get("targetid"))

        date_obs, date_mjd = extract_manifest_date(manifest_row)
        if not date_obs:
            date_obs, date_mjd = extract_date_from_hdul(hdul)

        redshift, z_source = extract_redshift_from_hdul(hdul)

        if survey == "SDSS":
            wave, flux, ivar = read_sdss_spectrum(hdul)
        elif survey == "DESI":
            wave, flux, ivar = read_desi_spectrum(hdul)
        else:
            # пробуем автоопределение
            try:
                wave, flux, ivar = read_sdss_spectrum(hdul)
                survey = "SDSS"
            except Exception:
                wave, flux, ivar = read_desi_spectrum(hdul)
                survey = "DESI"

    return SpectrumRecord(
        object_name=object_name,
        path=path,
        survey=survey,
        release=release,
        specid=specid,
        targetid=targetid,
        date_obs=date_obs,
        date_mjd=date_mjd,
        redshift=redshift,
        redshift_source=z_source,
        wave_obs=wave,
        flux=flux,
        ivar=ivar,
    )


# ============================================================
# pPXF: ВЫЧИТАНИЕ ЗВЁЗДНОГО НАСЕЛЕНИЯ
# ============================================================

def preprocess_spectrum(wave: np.ndarray, flux: np.ndarray, ivar: Optional[np.ndarray] = None):
    wave = np.asarray(wave, dtype=float).copy()
    flux = np.asarray(flux, dtype=float).copy()

    mask = np.isfinite(wave) & np.isfinite(flux) & (wave > 0)
    if ivar is not None:
        ivar = np.asarray(ivar, dtype=float).copy()
        if ivar.shape == flux.shape:
            mask &= np.isfinite(ivar)

    wave = wave[mask]
    flux = flux[mask]
    if ivar is not None and ivar.shape == mask.shape:
        ivar = ivar[mask]
    else:
        ivar = None

    if wave.size < 50:
        raise ValueError("Слишком мало валидных точек в спектре")

    order = np.argsort(wave)
    wave = wave[order]
    flux = flux[order]
    if ivar is not None:
        ivar = ivar[order]

    # убираем дубликаты по длине волны
    uniq_idx = np.concatenate([[True], np.diff(wave) > 0])
    wave = wave[uniq_idx]
    flux = flux[uniq_idx]
    if ivar is not None:
        ivar = ivar[uniq_idx]

    return wave, flux, ivar


def remove_stellar_population(
    spectrum: np.ndarray,
    wl_obs: np.ndarray,
    redshift_0: float,
    sps_file: Path = PPXF_SPS_FILE,
    fwhm_gal: float = DEFAULT_FWHM_GAL,
    noise_level: float = PPXF_NOISE_LEVEL,
):
    if not sps_file.exists():
        raise FileNotFoundError(f"SPS file not found: {sps_file}")

    wl_obs = np.asarray(wl_obs, dtype=float).copy()
    spectrum = np.asarray(spectrum, dtype=float).copy()

    wl_rest = wl_obs / (1.0 + redshift_0)
    fwhm_rest = fwhm_gal / (1.0 + redshift_0)

    galaxy_flux, ln_lam, velscale = util.log_rebin(wl_rest, spectrum)

    finite = np.isfinite(galaxy_flux)
    if not finite.any():
        raise ValueError("После log_rebin получен невалидный спектр")

    norm = np.nanmedian(galaxy_flux[finite])
    if not np.isfinite(norm) or abs(norm) < 1e-12:
        norm = np.nanmedian(np.abs(galaxy_flux[finite]))
    if not np.isfinite(norm) or abs(norm) < 1e-12:
        norm = 1.0

    galaxy = galaxy_flux / norm
    noise = np.full_like(galaxy, float(noise_level))

    sps = lib.sps_lib(str(sps_file), velscale, fwhm_rest)

    lam_range_gal = np.array([np.min(np.exp(ln_lam)), np.max(np.exp(ln_lam))])
    c = 299792.458
    vel = c * np.log(1.0)
    start = [vel, 250.0]

    tie_balmer = False
    limit_doublets = True

    gas_templates, gas_names, _ = util.emission_lines(
        sps.ln_lam_temp,
        lam_range_gal,
        fwhm_rest,
        tie_balmer=tie_balmer,
        limit_doublets=limit_doublets,
    )

    stars_templates = sps.templates.reshape(sps.templates.shape[0], -1)
    templates = np.column_stack([stars_templates, gas_templates])

    n_temps = stars_templates.shape[1]
    n_forbidden = np.sum(["[" in a for a in gas_names])
    n_balmer = len(gas_names) - n_forbidden

    component = [0] * n_temps + [1] * n_balmer + [2] * n_forbidden
    gas_component = np.array(component) > 0
    start = [start, start, start]

    pp = ppxf(
        templates,
        galaxy,
        noise,
        velscale,
        start,
        moments=[4, 2, 2],
        degree=-1,
        mdegree=60,
        lam=np.exp(ln_lam),
        lam_temp=sps.lam_temp,
        reg_dim=sps.templates.shape[1:],
        component=component,
        gas_component=gas_component,
        gas_names=gas_names,
        quiet=True,
    )

    wavelength_rest = np.exp(ln_lam)
    stellar_model = pp.matrix[:, :n_temps].dot(pp.weights[:n_temps]) * norm
    fit_spectrum = galaxy_flux - stellar_model

    return wavelength_rest, fit_spectrum, pp


# ============================================================
# ФИТ Hα / Hβ
# ============================================================

def integrate_broad_component(wave: np.ndarray, broad_flux: np.ndarray, threshold_fraction: float = 0.1):
    if wave is None or broad_flux is None or len(wave) == 0:
        return None, None
    peak = np.nanmax(broad_flux)
    if not np.isfinite(peak) or peak <= 0:
        return 0.0, np.zeros_like(broad_flux, dtype=bool)

    mask = broad_flux > threshold_fraction * peak
    if np.sum(mask) < 2:
        return 0.0, mask

    val = np.trapz(broad_flux[mask], wave[mask])
    return float(val), mask


def mc_broad_flux(
    params: np.ndarray,
    pcov: np.ndarray,
    param_slice: slice,
    wave: np.ndarray,
    build_broad,
    lower_bounds: list[float],
    upper_bounds: list[float],
    n_mc: int = MC_SAMPLES,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    samples = []

    base = params[param_slice].copy()

    subcov = None
    if pcov is not None and np.ndim(pcov) == 2:
        try:
            subcov = np.array(pcov[param_slice, param_slice], dtype=float)
            if not np.all(np.isfinite(subcov)):
                subcov = None
        except Exception:
            subcov = None

    for _ in range(n_mc):
        p = params.copy()

        drawn = None
        if subcov is not None:
            try:
                drawn = rng.multivariate_normal(base, subcov)
            except Exception:
                drawn = None

        if drawn is None:
            # fallback-расброс как в духе вашего кода
            if len(base) != 3:
                continue
            drawn = base.copy()
            drawn[0] += rng.normal(0, max(abs(base[0]) * 0.05, 1e-8))
            drawn[1] += rng.normal(0, 1.0)
            drawn[2] += rng.normal(0, 1.0)

        # обрезаем по bounds
        ok = True
        for i in range(len(drawn)):
            lo = lower_bounds[i]
            hi = upper_bounds[i]
            if not np.isfinite(drawn[i]):
                ok = False
                break
            if drawn[i] < lo:
                drawn[i] = lo
            if np.isfinite(hi) and drawn[i] > hi:
                drawn[i] = hi

        if not ok:
            continue

        p[param_slice] = drawn
        broad = build_broad(wave, p)
        if broad is None or len(broad) == 0:
            continue

        flux, _ = integrate_broad_component(wave, broad, threshold_fraction=0.1)
        if flux is not None and np.isfinite(flux):
            samples.append(flux)

    if not samples:
        broad0 = build_broad(wave, params)
        det_flux, _ = integrate_broad_component(wave, broad0, threshold_fraction=0.1)
        return det_flux, 0.0

    return float(np.mean(samples)), float(np.std(samples))


def failed_line_result(line_name: str, reason: str) -> LineFitResult:
    return LineFitResult(line_name=line_name, ok=False, reason=reason)


def fit_ha_line(wave_rest: np.ndarray, flux_resid: np.ndarray, flux_orig_rest: np.ndarray) -> LineFitResult:
    mask = (wave_rest > 6500) & (wave_rest < 6620)
    if np.sum(mask) < 20:
        return failed_line_result("Halpha", "слишком мало точек в окне Hα")

    wl = wave_rest[mask]
    flux = flux_resid[mask]
    orig_flux = flux_orig_rest[mask]

    if not np.all(np.isfinite(flux)):
        return failed_line_result("Halpha", "невалидные значения в окне Hα")

    med = np.nanmedian(flux)
    peak = np.nanmax(flux) - med
    if not np.isfinite(peak) or peak <= 0:
        peak = max(np.nanpercentile(flux, 95) - med, 1e-6)

    p0 = [
        peak,           6563.0, 3.0,
        peak / 2.0,     6563.0, 12.0,
        peak / 4.0,     6548.0, 3.0,
        peak / 3.0,     6583.0, 3.0,
        med
    ]
    bounds = (
        [0, 6561, 0.5,   0, 6561, 4,    0, 6546, 0.5,   0, 6581, 0.5, -np.inf],
        [np.inf, 6566, 10, np.inf, 6566, 30, np.inf, 6550, 8, np.inf, 6586, 8, np.inf]
    )

    try:
        params, pcov = curve_fit(
            ha_model_full,
            wl,
            flux,
            p0=p0,
            bounds=bounds,
            maxfev=200000
        )
    except Exception as e:
        return failed_line_result("Halpha", f"curve_fit failed: {e}")

    narrow = params[0] * np.exp(-0.5 * ((wl - params[1]) / params[2]) ** 2)
    broad = params[3] * np.exp(-0.5 * ((wl - params[4]) / params[5]) ** 2)
    nii_6548 = params[6] * np.exp(-0.5 * ((wl - params[7]) / params[8]) ** 2)
    nii_6583 = params[9] * np.exp(-0.5 * ((wl - params[10]) / params[11]) ** 2)
    model = ha_model_full(wl, *params)
    resid = flux - model
    offset = params[12]

    det_flux, int_mask = integrate_broad_component(wl, broad, threshold_fraction=0.1)

    def build_broad(x, p):
        return p[3] * np.exp(-0.5 * ((x - p[4]) / p[5]) ** 2)

    mc_mean, mc_std = mc_broad_flux(
        params=params,
        pcov=pcov,
        param_slice=slice(3, 6),
        wave=wl,
        build_broad=build_broad,
        lower_bounds=[0, 6561, 4],
        upper_bounds=[np.inf, 6566, 30],
        n_mc=MC_SAMPLES,
        seed=42,
    )

    return LineFitResult(
        line_name="Halpha",
        ok=True,
        reason="",
        wave=wl,
        flux=flux,
        orig_flux=orig_flux,
        model=model,
        resid=resid,
        broad_component=broad,
        narrow_component=narrow,
        extra1_component=nii_6548,
        extra2_component=nii_6583,
        offset=offset,
        params=params,
        pcov=pcov,
        broad_flux_det=det_flux,
        broad_flux_mc_mean=mc_mean,
        broad_flux_mc_std=mc_std,
        integration_mask=int_mask,
    )


def fit_hb_line(wave_rest: np.ndarray, flux_resid: np.ndarray, flux_orig_rest: np.ndarray) -> LineFitResult:
    mask = (wave_rest > 4830) & (wave_rest < 4900)
    if np.sum(mask) < 20:
        return failed_line_result("Hbeta", "слишком мало точек в окне Hβ")

    wl = wave_rest[mask]
    flux = flux_resid[mask]
    orig_flux = flux_orig_rest[mask]

    if not np.all(np.isfinite(flux)):
        return failed_line_result("Hbeta", "невалидные значения в окне Hβ")

    med = np.nanmedian(flux)
    peak = np.nanmax(flux) - med
    if not np.isfinite(peak) or peak <= 0:
        peak = max(np.nanpercentile(flux, 95) - med, 1e-6)

    p0 = [peak, 4866.0, 2.0, peak / 2.0, 4866.0, 10.0, med]
    bounds = (
        [0, 4860, 0.5, 0, 4860, 1, -np.inf],
        [np.inf, 4872, 8, np.inf, 4872, 40, np.inf]
    )

    try:
        params, pcov = curve_fit(
            double_gaussian,
            wl,
            flux,
            p0=p0,
            bounds=bounds,
            maxfev=200000
        )
    except Exception as e:
        return failed_line_result("Hbeta", f"curve_fit failed: {e}")

    narrow = params[0] * np.exp(-0.5 * ((wl - params[1]) / params[2]) ** 2)
    broad = params[3] * np.exp(-0.5 * ((wl - params[4]) / params[5]) ** 2)
    model = double_gaussian(wl, *params)
    resid = flux - model
    offset = params[6]

    det_flux, int_mask = integrate_broad_component(wl, broad, threshold_fraction=0.1)

    def build_broad(x, p):
        return p[3] * np.exp(-0.5 * ((x - p[4]) / p[5]) ** 2)

    mc_mean, mc_std = mc_broad_flux(
        params=params,
        pcov=pcov,
        param_slice=slice(3, 6),
        wave=wl,
        build_broad=build_broad,
        lower_bounds=[0, 4860, 1],
        upper_bounds=[np.inf, 4872, 40],
        n_mc=MC_SAMPLES,
        seed=43,
    )

    return LineFitResult(
        line_name="Hbeta",
        ok=True,
        reason="",
        wave=wl,
        flux=flux,
        orig_flux=orig_flux,
        model=model,
        resid=resid,
        broad_component=broad,
        narrow_component=narrow,
        extra1_component=None,
        extra2_component=None,
        offset=offset,
        params=params,
        pcov=pcov,
        broad_flux_det=det_flux,
        broad_flux_mc_mean=mc_mean,
        broad_flux_mc_std=mc_std,
        integration_mask=int_mask,
    )


# ============================================================
# ГРАФИКИ
# ============================================================

def plot_line_panel(ax, fit: LineFitResult, title: str):
    if not fit.ok:
        ax.text(0.5, 0.5, f"{title}\nfit failed:\n{fit.reason}",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        return

    ax.plot(fit.wave, fit.orig_flux, color="crimson", lw=1.5, alpha=0.45, label="Original spectrum")
    ax.plot(fit.wave, fit.flux, color="black", lw=1.0, label="Stellar removed")

    if fit.narrow_component is not None:
        ax.plot(fit.wave, fit.narrow_component, color="turquoise", label="Narrow")

    if fit.broad_component is not None:
        ax.plot(fit.wave, fit.broad_component, color="limegreen", label="Broad")

    if fit.extra1_component is not None:
        ax.plot(fit.wave, fit.extra1_component, color="orchid", ls="--", label="Extra 1")

    if fit.extra2_component is not None:
        ax.plot(fit.wave, fit.extra2_component, color="mediumslateblue", ls="--", label="Extra 2")

    if fit.model is not None:
        ax.plot(fit.wave, fit.model, color="orange", lw=2.0, label="Model")

    if fit.offset is not None:
        ax.axhline(fit.offset, color="cyan", ls="-", label="Continuum")

    if fit.resid is not None:
        ax.plot(fit.wave, fit.resid, color="gray", lw=1.0, label="Residual")

    if fit.integration_mask is not None and fit.broad_component is not None:
        mask = fit.integration_mask
        if np.sum(mask) >= 2:
            ax.fill_between(
                fit.wave[mask],
                fit.broad_component[mask],
                0,
                color="limegreen",
                alpha=0.2,
                label="Integration range"
            )

    flux_txt = ""
    if fit.broad_flux_mc_mean is not None:
        flux_txt = f"{fit.broad_flux_mc_mean:.2e} ± {fit.broad_flux_mc_std:.2e}"

    ax.set_title(f"{title}\nBroad flux = {flux_txt}")
    ax.set_xlabel("Wavelength Å")
    ax.set_ylabel("Flux $10^{-17} erg cm^{-2} s^{-1} \AA^{-1}$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def save_analysis_figure(
    out_path: Path,
    record: SpectrumRecord,
    wave_rest_full: np.ndarray,
    flux_orig_rest_interp: np.ndarray,
    flux_resid: np.ndarray,
    ha_result: LineFitResult,
    hb_result: LineFitResult,
):
    fig = plt.figure(figsize=(14, 12))

    ax0 = plt.subplot(3, 1, 1)
    ax0.plot(wave_rest_full, flux_orig_rest_interp, color="crimson", alpha=0.45, lw=1.0, label="Original")
    ax0.plot(wave_rest_full, flux_resid, color="black", lw=0.8, label="Stellar removed")
    ax0.set_xlabel("Wavelength Å")
    ax0.set_ylabel("Flux $10^{-17} erg cm^{-2} s^{-1} \AA^{-1}$")
    ax0.set_title(
        f"{record.object_name} | {record.survey} | {record.path.name}\n"
        f"date={record.date_obs or 'NA'} | z={record.redshift if record.redshift is not None else 'NA'}"
    )
    ax0.grid(True, alpha=0.3)
    ax0.legend()

    ax1 = plt.subplot(3, 1, 2)
    plot_line_panel(ax1, ha_result, "Hα")

    ax2 = plt.subplot(3, 1, 3)
    plot_line_panel(ax2, hb_result, "Hβ")

    for ax in (ax0, ax1, ax2):
        apply_article_axis_style(ax)

    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# FITS HEADER UPDATE
# ============================================================

def update_fits_header_with_results(
    record: SpectrumRecord,
    ha_result: LineFitResult,
    hb_result: LineFitResult,
    plot_path: Path,
) -> None:
    if not WRITE_RESULTS_TO_FITS_HEADER:
        return

    try:
        with fits.open(record.path, mode="update", memmap=False) as hdul:
            hdr = hdul[0].header

            if record.redshift is not None:
                hdr["ZUSED"] = (float(record.redshift), "Redshift used in analysis")
            if record.redshift_source:
                hdr["ZSRC"] = (str(record.redshift_source)[:68], "Redshift source")

            if record.date_obs:
                hdr["DATE-OBS"] = (str(record.date_obs)[:68], "Observation date")
            if record.date_mjd is not None:
                hdr["MJDAN"] = (float(record.date_mjd), "Analysis date MJD")

            if ha_result.ok:
                if ha_result.broad_flux_mc_mean is not None:
                    hdr["HABRFLX"] = (float(ha_result.broad_flux_mc_mean), "Broad Halpha flux")
                if ha_result.broad_flux_mc_std is not None:
                    hdr["EHABRFLX"] = (float(ha_result.broad_flux_mc_std), "Err broad Halpha")
            if hb_result.ok:
                if hb_result.broad_flux_mc_mean is not None:
                    hdr["HBBRFLX"] = (float(hb_result.broad_flux_mc_mean), "Broad Hbeta flux")
                if hb_result.broad_flux_mc_std is not None:
                    hdr["EHBBRFLX"] = (float(hb_result.broad_flux_mc_std), "Err broad Hbeta")

            hdr["ANPLOT"] = (str(plot_path.name)[:68], "Analysis plot")
            hdr["ANSTAT"] = ("ok" if ha_result.ok or hb_result.ok else "failed", "Analysis status")
            hdr["FLXTYPE"] = ("broad_10pct_MC", "Flux definition")

            hdul.flush()
    except Exception as e:
        print(f"    Header update failed for {record.path.name}: {e}")


# ============================================================
# АНАЛИЗ ОДНОГО СПЕКТРА
# ============================================================

def analyze_spectrum(record: SpectrumRecord) -> dict:
    base_name = strip_spectrum_suffix(record.path)
    plot_path = record.path.with_name(f"{base_name}_analysis.png")

    row = {
        "object_name": record.object_name,
        "survey": record.survey,
        "release": record.release,
        "spectrum_file": str(record.path),
        "file_name": record.path.name,
        "specid": record.specid,
        "targetid": record.targetid,
        "date_obs": record.date_obs,
        "date_mjd": record.date_mjd,
        "redshift": record.redshift,
        "redshift_source": record.redshift_source,
        "ha_broad_flux": np.nan,
        "ha_broad_err": np.nan,
        "hb_broad_flux": np.nan,
        "hb_broad_err": np.nan,
        "ha_ok": False,
        "hb_ok": False,
        "status": "failed",
        "notes": "",
        "plot_file": str(plot_path),
    }

    if record.redshift is None:
        row["notes"] = "redshift not found"
        return row

    try:
        wave_obs, flux, ivar = preprocess_spectrum(record.wave_obs, record.flux, record.ivar)
    except Exception as e:
        row["notes"] = f"preprocess failed: {e}"
        return row

    try:
        wave_rest, flux_resid, _ = remove_stellar_population(
            spectrum=flux,
            wl_obs=wave_obs,
            redshift_0=record.redshift,
            sps_file=PPXF_SPS_FILE,
            fwhm_gal=DEFAULT_FWHM_GAL,
            noise_level=PPXF_NOISE_LEVEL,
        )
        flux_orig_rest_interp = np.interp(wave_rest, wave_obs / (1.0 + record.redshift), flux)
        notes_prefix = ""
    except Exception as e:
        if not FALLBACK_TO_DIRECT_SPECTRUM_IF_PPXF_FAIL:
            row["notes"] = f"ppxf failed: {e}"
            return row

        wave_rest = wave_obs / (1.0 + record.redshift)
        flux_resid = flux.copy()
        flux_orig_rest_interp = flux.copy()
        notes_prefix = f"ppxf failed -> direct fit used: {e}; "

    ha_result = fit_ha_line(wave_rest, flux_resid, flux_orig_rest_interp)
    hb_result = fit_hb_line(wave_rest, flux_resid, flux_orig_rest_interp)

    try:
        if OVERWRITE_PLOTS or not plot_path.exists():
            save_analysis_figure(
                out_path=plot_path,
                record=record,
                wave_rest_full=wave_rest,
                flux_orig_rest_interp=flux_orig_rest_interp,
                flux_resid=flux_resid,
                ha_result=ha_result,
                hb_result=hb_result,
            )
    except Exception as e:
        print(f"    Plot save failed for {record.path.name}: {e}")

    try:
        update_fits_header_with_results(record, ha_result, hb_result, plot_path)
    except Exception:
        pass

    row["ha_ok"] = bool(ha_result.ok)
    row["hb_ok"] = bool(hb_result.ok)

    if ha_result.ok:
        row["ha_broad_flux"] = ha_result.broad_flux_mc_mean
        row["ha_broad_err"] = ha_result.broad_flux_mc_std

    if hb_result.ok:
        row["hb_broad_flux"] = hb_result.broad_flux_mc_mean
        row["hb_broad_err"] = hb_result.broad_flux_mc_std

    if ha_result.ok or hb_result.ok:
        row["status"] = "ok" if (ha_result.ok and hb_result.ok) else "partial"
    else:
        row["status"] = "failed"

    notes = notes_prefix
    if not ha_result.ok:
        notes += f"Halpha: {ha_result.reason}; "
    if not hb_result.ok:
        notes += f"Hbeta: {hb_result.reason}; "
    row["notes"] = notes.strip(" ;")

    return row


# ============================================================
# ОБРАБОТКА ОБЪЕКТА
# ============================================================

def process_object(object_dir: Path, redshift_cache: dict[int, dict]) -> pd.DataFrame:
    print(f"\n=== {object_dir.name} ===")

    manifest_map = load_object_manifest(object_dir)
    files = list_spectrum_files(object_dir)

    if not files:
        print("  Нет спектров")
        return pd.DataFrame()

    records = []
    for path in files:
        try:
            manifest_row = manifest_map.get(path.name)
            rec = load_spectrum_record(path, object_dir.name, manifest_row)
            records.append(rec)
        except Exception as e:
            print(f"  Не удалось прочитать {path.name}: {e}")

    if not records:
        print("  Нет корректно прочитанных спектров")
        return pd.DataFrame()

    # добираем redshift по specid из кэша / SPARCL
    missing_specids = []
    for rec in records:
        if rec.redshift is None and rec.specid is not None:
            if rec.specid in redshift_cache and redshift_cache[rec.specid].get("redshift") is not None:
                rec.redshift = redshift_cache[rec.specid]["redshift"]
                rec.redshift_source = "sparcl_cache"
                if not rec.release:
                    rec.release = redshift_cache[rec.specid].get("release", "")
                if rec.targetid is None:
                    rec.targetid = redshift_cache[rec.specid].get("targetid")
            else:
                missing_specids.append(rec.specid)

    if missing_specids:
        remote = query_redshifts_from_sparcl(missing_specids)
        redshift_cache.update(remote)

        for rec in records:
            if rec.redshift is None and rec.specid is not None and rec.specid in remote:
                rec.redshift = remote[rec.specid].get("redshift")
                rec.redshift_source = "sparcl_remote"
                if not rec.release:
                    rec.release = remote[rec.specid].get("release", "")
                if rec.targetid is None:
                    rec.targetid = remote[rec.specid].get("targetid")

    rows = []
    for rec in records:
        print(f"  -> {rec.path.name} [{rec.survey}]")
        try:
            row = analyze_spectrum(rec)
        except Exception as e:
            row = {
                "object_name": rec.object_name,
                "survey": rec.survey,
                "release": rec.release,
                "spectrum_file": str(rec.path),
                "file_name": rec.path.name,
                "specid": rec.specid,
                "targetid": rec.targetid,
                "date_obs": rec.date_obs,
                "date_mjd": rec.date_mjd,
                "redshift": rec.redshift,
                "redshift_source": rec.redshift_source,
                "ha_broad_flux": np.nan,
                "ha_broad_err": np.nan,
                "hb_broad_flux": np.nan,
                "hb_broad_err": np.nan,
                "ha_ok": False,
                "hb_ok": False,
                "status": "failed",
                "notes": f"fatal: {e}",
                "plot_file": "",
            }
            print(f"     ERROR: {e}")
            traceback.print_exc()

        rows.append(row)

    df = pd.DataFrame(rows)

    if not df.empty:
        if "date_mjd" in df.columns:
            df["date_sort"] = pd.to_numeric(df["date_mjd"], errors="coerce")
            df["date_sort"] = df["date_sort"].fillna(1e12)
        else:
            df["date_sort"] = 1e12

        df = df.sort_values(
            by=["date_sort", "survey", "file_name"],
            ascending=[True, True, True]
        ).drop(columns=["date_sort"]).reset_index(drop=True)

    out_path = object_dir / OBJECT_SUMMARY_NAME
    df.to_csv(out_path, sep="\t", index=False)
    print(f"  saved: {out_path}")

    return df


# ============================================================
# MAIN
# ============================================================

def main():
    if not ROOT_DIR.exists():
        raise FileNotFoundError(f"ROOT_DIR not found: {ROOT_DIR}")

    if not PPXF_SPS_FILE.exists():
        raise FileNotFoundError(f"PPXF_SPS_FILE not found: {PPXF_SPS_FILE}")

    object_dirs = sorted([p for p in ROOT_DIR.iterdir() if p.is_dir() and p.name.startswith("J")])

    if not object_dirs:
        print("Папки объектов не найдены")
        return

    cache_path = ROOT_DIR / REDSHIFT_CACHE_NAME
    redshift_cache = load_redshift_cache(cache_path)

    all_parts = []

    for object_dir in object_dirs:
        try:
            df = process_object(object_dir, redshift_cache)
            if not df.empty:
                all_parts.append(df)
        except Exception as e:
            print(f"FATAL in {object_dir.name}: {e}")
            traceback.print_exc()

    save_redshift_cache(cache_path, redshift_cache)

    if all_parts:
        master_df = pd.concat(all_parts, ignore_index=True)
    else:
        master_df = pd.DataFrame(columns=[
            "object_name", "survey", "release", "spectrum_file", "file_name",
            "specid", "targetid", "date_obs", "date_mjd", "redshift",
            "redshift_source", "ha_broad_flux", "ha_broad_err",
            "hb_broad_flux", "hb_broad_err", "ha_ok", "hb_ok",
            "status", "notes", "plot_file"
        ])

    master_path = ROOT_DIR / MASTER_SUMMARY_NAME
    master_df.to_csv(master_path, sep="\t", index=False)

    print(f"\nГотово.")
    print(f"Общий файл: {master_path}")
    print(f"Кэш redshift: {cache_path}")


if __name__ == "__main__":
    main()