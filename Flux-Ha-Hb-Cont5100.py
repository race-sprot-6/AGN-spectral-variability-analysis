from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import re
import warnings

import matplotlib
matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# НАСТРОЙКИ
# ============================================================

ROOT_DIR = Path(r"C:\Users\IvanK\Astro\Course5\downloaded_spectra")

LINE_TABLE_NAME = "line_fluxes_vs_time.tsv"
CONT_TABLE_NAME = "continuum_5100_vs_time.tsv"
MERGED_TABLE_NAME = "line_and_continuum_vs_time.tsv"

CONT_PLOT_SUFFIX = "_cont5100.png"
LIGHTCURVE_ABS_NAME = "light_curve_Ha_Hb_Cont5100_absolute.png"
LIGHTCURVE_NORM_NAME = "light_curve_Ha_Hb_Cont5100_normalized.png"

# Окно континуума
CONT_CENTER_REST = 5100.0
CONT_HALF_WIDTH = 50.0

# Сглаживание сырого спектра
USE_SMOOTHING = True
SMOOTH_SIGMA_PIX = 3.0  # можно попробовать 5.0 если шумно

# Для красивых графиков с разрывами по времени
GAP_THRESHOLD_DAYS = 1000.0
MIN_PAD_DAYS_SINGLE_POINT = 40.0
MIN_PAD_DAYS_MULTI_POINT = 15.0

# Если хотите логарифмическую ось Y на абсолютном графике
ABSOLUTE_LOG_Y = False

# Тики внутрь, как в статьях
TICKS_INWARD = True

# Ошибки континуума слишком большими не рисуем.
# Например 0.5 означает: ошибка больше 50% от потока не будет показана.
MAX_CONT_ERR_FRACTION = 0.5

# Допуск при сопоставлении MJD из таблицы и MJD в файлах
CONT_ERR_MJD_TOL = 0.75

# ============================================================
# НАСТРОЙКИ ОФОРМЛЕНИЯ ГРАФИКОВ И ОШИБОК
# ============================================================

# Тики внутрь, как в статьях
TICKS_INWARD = True

# Подпись оси Y для нормированного графика:
# "dimensionless" — физически корректно: Normalized flux
# "article" — как в статьях: Flux × 10^{-17} erg cm^{-2} s^{-1} Å^{-1}
NORMALIZED_YLABEL_MODE = "article"

# Множитель в подписи потока.
# Для SDSS/DESI спектральные flux обычно в 10^-17 erg / (cm^2 s Å),
# поэтому по умолчанию ставим 10^-17.
FLUX_UNIT_POWER = -17

# Файл для ручного ввода ошибок континуума, если их нет/они плохие.
# Создаётся автоматически в папке объекта.
MANUAL_CONT_ERROR_TABLE_NAME = "manual_continuum_errors.tsv"

# Если True — будет спрашивать ошибку в консоли для каждой точки,
# где ошибка континуума отсутствует или отбракована.
# Если False — создаст/обновит manual_continuum_errors.tsv,
# куда можно руками вписать ошибки и перезапустить код.
ASK_MANUAL_CONT_ERRORS_IN_CONSOLE = False

# Максимальная допустимая относительная ошибка континуума.
# Например 1.0 значит: ошибка не должна быть больше 100% от потока.
MAX_CONT_ERR_RELATIVE_TO_FLUX = 1.0

# Дополнительная защита от выбросов:
# если ошибка больше median(valid_errors) * этот коэффициент,
# она не рисуется.
MAX_CONT_ERR_RELATIVE_TO_MEDIAN_ERR = 20.0

# Минимальное число хороших ошибок для медианной проверки
MIN_GOOD_ERRORS_FOR_MEDIAN_CHECK = 3

# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def safe_float(value):
    try:
        if value is None or pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def safe_int(value):
    try:
        if value is None or pd.isna(value):
            return None
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return None


def mjd_to_iso(mjd: float) -> str:
    try:
        return Time(float(mjd), format="mjd").isot
    except Exception:
        return ""


def parse_date_to_mjd(date_str: str) -> float:
    if not date_str:
        return np.nan

    s = str(date_str).strip()
    if not s or s.lower() == "nan":
        return np.nan

    candidates = [
        s,
        s.replace(" ", "T"),
        s.replace("+00", ""),
        s.replace(" ", "T").replace("+00", ""),
    ]

    for c in candidates:
        try:
            return float(Time(c).mjd)
        except Exception:
            pass

    return np.nan


def mjd_to_datetime(mjd: float):
    return Time(float(mjd), format="mjd").to_datetime()


def strip_fits_suffix(path: Path) -> str:
    name = path.name
    lower = name.lower()
    for suf in [".fits.gz", ".fit.gz", ".fits", ".fit", ".fz"]:
        if lower.endswith(suf):
            return name[:-len(suf)]
    return path.stem


def guess_survey_from_path(path: Path) -> str:
    parent = path.parent.name.lower()
    if parent == "sdss":
        return "SDSS"
    if parent == "desi":
        return "DESI"
    return "OTHER"


def list_spectrum_files(object_dir: Path) -> list[Path]:
    files = []
    for sub in ["sdss", "desi"]:
        subdir = object_dir / sub
        if not subdir.exists():
            continue
        for pattern in ["*.fits", "*.fit", "*.fits.gz", "*.fit.gz", "*.fz"]:
            files.extend(sorted(subdir.glob(pattern)))
    uniq = []
    seen = set()
    for p in files:
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            uniq.append(p)
            seen.add(key)
    return uniq


def find_object_dirs(root: Path) -> list[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("J")])


def load_table_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, sep="\t")
    except Exception:
        return pd.DataFrame()


def table_to_map_by_filename(df: pd.DataFrame) -> dict[str, dict]:
    if df.empty or "file_name" not in df.columns:
        return {}
    out = {}
    for _, row in df.iterrows():
        fname = row.get("file_name")
        if fname is None or pd.isna(fname):
            continue
        out[str(fname)] = row.to_dict()
    return out


def load_manifest_map(object_dir: Path) -> dict[str, dict]:
    path = object_dir / "spectra_manifest.tsv"
    df = load_table_if_exists(path)
    if df.empty:
        return {}
    out = {}
    for _, row in df.iterrows():
        local_file = row.get("local_file")
        if local_file is None or pd.isna(local_file):
            continue
        name = Path(str(local_file)).name
        out[name] = row.to_dict()
    return out


def load_redshift_cache(root_dir: Path) -> dict[int, dict]:
    path = root_dir / "_redshift_cache.tsv"
    df = load_table_if_exists(path)
    if df.empty or "specid" not in df.columns:
        return {}
    out = {}
    for _, row in df.iterrows():
        specid = safe_int(row.get("specid"))
        if specid is None:
            continue
        out[specid] = {
            "redshift": safe_float(row.get("redshift")),
            "release": str(row.get("release", "")) if row.get("release") is not None else "",
            "targetid": safe_int(row.get("targetid")),
        }
    return out


def extract_specid_from_filename(path: Path) -> int | None:
    name = path.name
    patterns = [
        r"desi_spec_(\d+)\.(?:fits|fit)(?:\.gz)?$",
        r"_desi_(\d+)\.(?:fits|fit)(?:\.gz)?$",
    ]
    for pat in patterns:
        m = re.search(pat, name, flags=re.IGNORECASE)
        if m:
            return safe_int(m.group(1))
    return None


def preprocess_spectrum(wave: np.ndarray, flux: np.ndarray, ivar: np.ndarray | None = None):
    wave = np.asarray(wave, dtype=float).copy()
    flux = np.asarray(flux, dtype=float).copy()

    mask = np.isfinite(wave) & np.isfinite(flux) & (wave > 0)
    if ivar is not None:
        ivar = np.asarray(ivar, dtype=float).copy()
        if ivar.shape == flux.shape:
            mask &= np.isfinite(ivar)
        else:
            ivar = None

    wave = wave[mask]
    flux = flux[mask]
    if ivar is not None:
        ivar = ivar[mask]

    if wave.size < 10:
        raise ValueError("слишком мало валидных точек")

    order = np.argsort(wave)
    wave = wave[order]
    flux = flux[order]
    if ivar is not None:
        ivar = ivar[order]

    uniq = np.concatenate([[True], np.diff(wave) > 0])
    wave = wave[uniq]
    flux = flux[uniq]
    if ivar is not None:
        ivar = ivar[uniq]

    return wave, flux, ivar

def format_flux_unit_label(power: int = FLUX_UNIT_POWER) -> str:
    """
    Подпись единиц потока в стиле статей.

    Для SDSS/DESI flux обычно хранится в единицах:
    10^-17 erg / (cm^2 s Å)

    Ангстрем пишем как Å.
    """
    return rf"Flux $\times 10^{{{power}}}$ erg cm$^{{-2}}$ s$^{{-1}}$ $\AA^{{-1}}$"


def get_normalized_ylabel() -> str:
    """
    Нормированный поток физически безразмерный.
    Но если нужно оформление 'как в статье', можно включить article-режим.
    """
    if NORMALIZED_YLABEL_MODE.lower() == "article":
        return format_flux_unit_label(FLUX_UNIT_POWER)
    return "Normalized flux"


def apply_article_axis_style(ax):
    """
    Общий стиль осей: тики внутрь, включая minor ticks.
    """
    if TICKS_INWARD:
        ax.tick_params(
            axis="both",
            which="both",
            direction="in",
            top=True,
            right=True,
        )
        ax.minorticks_on()


def load_manual_continuum_errors(object_dir: Path) -> pd.DataFrame:
    """
    Загружает файл ручных ошибок континуума.

    Ожидаемые колонки:
    file_name    cont5100_err_manual

    Можно создать файл вручную или он будет создан автоматически.
    """
    path = object_dir / MANUAL_CONT_ERROR_TABLE_NAME
    if not path.exists():
        return pd.DataFrame(columns=["file_name", "cont5100_err_manual"])

    df = load_table_if_exists(path)
    if df.empty:
        return pd.DataFrame(columns=["file_name", "cont5100_err_manual"])

    if "file_name" not in df.columns:
        df["file_name"] = ""

    if "cont5100_err_manual" not in df.columns:
        df["cont5100_err_manual"] = np.nan

    df["cont5100_err_manual"] = pd.to_numeric(df["cont5100_err_manual"], errors="coerce")
    return df[["file_name", "cont5100_err_manual"]].copy()


def save_manual_continuum_error_template(object_dir: Path, df: pd.DataFrame):
    """
    Создаёт/обновляет manual_continuum_errors.tsv.

    Туда попадают все file_name, для которых можно вписать ошибку вручную.
    Если файл уже есть, введённые значения сохраняются.
    """
    path = object_dir / MANUAL_CONT_ERROR_TABLE_NAME

    existing = load_manual_continuum_errors(object_dir)
    existing_map = {}
    if not existing.empty:
        for _, row in existing.iterrows():
            fname = str(row.get("file_name", "")).strip()
            if fname:
                existing_map[fname] = safe_float(row.get("cont5100_err_manual"))

    rows = []
    if not df.empty and "file_name" in df.columns:
        for _, row in df.iterrows():
            fname = str(row.get("file_name", "")).strip()
            if not fname:
                continue
            rows.append({
                "file_name": fname,
                "cont5100_err_manual": existing_map.get(fname, np.nan),
            })

    out = pd.DataFrame(rows).drop_duplicates(subset=["file_name"])
    out.to_csv(path, sep="\t", index=False)


def get_manual_continuum_error_for_file(object_dir: Path, file_name: str) -> float:
    """
    Берёт ручную ошибку из manual_continuum_errors.tsv.
    """
    df = load_manual_continuum_errors(object_dir)
    if df.empty:
        return np.nan

    hit = df[df["file_name"].astype(str) == str(file_name)]
    if hit.empty:
        return np.nan

    return safe_float(hit.iloc[0].get("cont5100_err_manual"))


def ask_manual_continuum_error(file_name: str, flux_value: float) -> float:
    """
    Интерактивный ввод ошибки из консоли.
    Работает только если ASK_MANUAL_CONT_ERRORS_IN_CONSOLE = True.
    """
    if not ASK_MANUAL_CONT_ERRORS_IN_CONSOLE:
        return np.nan

    print()
    print(f"Нет хорошей ошибки континуума для файла: {file_name}")
    print(f"cont5100_int_flux = {flux_value:.6e}")
    print("Введите ошибку cont5100_err или нажмите Enter, чтобы пропустить:")

    try:
        s = input("> ").strip()
    except Exception:
        return np.nan

    if not s:
        return np.nan

    return safe_float(s)


def is_good_error_value(flux_value: float,
                        err_value: float,
                        max_relative: float = MAX_CONT_ERR_RELATIVE_TO_FLUX) -> bool:
    """
    Проверка одной ошибки:
    - должна быть конечной;
    - должна быть положительной;
    - не должна быть слишком большой относительно потока.
    """
    flux_value = safe_float(flux_value)
    err_value = safe_float(err_value)

    if not np.isfinite(flux_value):
        return False
    if not np.isfinite(err_value):
        return False
    if err_value <= 0:
        return False

    scale = abs(flux_value)
    if not np.isfinite(scale) or scale <= 0:
        return False

    if err_value > max_relative * scale:
        return False

    return True


def sanitize_continuum_errors_for_plot(df: pd.DataFrame,
                                       object_dir: Path | None = None) -> pd.DataFrame:
    """
    Готовит колонку cont5100_err_plot.

    Логика:
    1. Берём cont5100_err из таблицы.
    2. Если её нет/она плохая — пробуем cont5100_err_manual из manual_continuum_errors.tsv.
    3. Если включён интерактивный режим — можно ввести ошибку в консоли.
    4. Слишком большие ошибки заменяются на NaN, то есть усы не рисуются.
    """
    out = df.copy()

    if "cont5100_int_flux" not in out.columns:
        out["cont5100_int_flux"] = np.nan
    if "cont5100_err" not in out.columns:
        out["cont5100_err"] = np.nan
    if "file_name" not in out.columns:
        out["file_name"] = ""

    out["cont5100_int_flux"] = pd.to_numeric(out["cont5100_int_flux"], errors="coerce")
    out["cont5100_err"] = pd.to_numeric(out["cont5100_err"], errors="coerce")

    if "cont5100_err_plot" not in out.columns:
        out["cont5100_err_plot"] = np.nan

    manual_df = pd.DataFrame()
    manual_map = {}

    if object_dir is not None:
        manual_df = load_manual_continuum_errors(object_dir)
        if not manual_df.empty:
            for _, row in manual_df.iterrows():
                fname = str(row.get("file_name", "")).strip()
                if fname:
                    manual_map[fname] = safe_float(row.get("cont5100_err_manual"))

    for idx, row in out.iterrows():
        fname = str(row.get("file_name", "")).strip()
        flux_value = safe_float(row.get("cont5100_int_flux"))
        err_value = safe_float(row.get("cont5100_err"))

        if is_good_error_value(flux_value, err_value):
            out.loc[idx, "cont5100_err_plot"] = err_value
            continue

        manual_err = manual_map.get(fname, np.nan)
        if is_good_error_value(flux_value, manual_err):
            out.loc[idx, "cont5100_err_plot"] = manual_err
            continue

        console_err = ask_manual_continuum_error(fname, flux_value)
        if is_good_error_value(flux_value, console_err):
            out.loc[idx, "cont5100_err_plot"] = console_err
            continue

        out.loc[idx, "cont5100_err_plot"] = np.nan

    # Дополнительная медианная фильтрация от одного объекта с огромными ошибками
    good_err = out["cont5100_err_plot"].to_numpy(dtype=float)
    good_err = good_err[np.isfinite(good_err) & (good_err > 0)]

    if len(good_err) >= MIN_GOOD_ERRORS_FOR_MEDIAN_CHECK:
        med_err = np.nanmedian(good_err)
        if np.isfinite(med_err) and med_err > 0:
            too_large = out["cont5100_err_plot"] > MAX_CONT_ERR_RELATIVE_TO_MEDIAN_ERR * med_err
            if too_large.any():
                bad_files = out.loc[too_large, "file_name"].astype(str).tolist()
                print(f"  [WARN] Слишком большие ошибки cont5100_err не будут нарисованы: {bad_files}")
                out.loc[too_large, "cont5100_err_plot"] = np.nan

    if object_dir is not None:
        save_manual_continuum_error_template(object_dir, out)

    return out

# ============================================================
# ЧТЕНИЕ FITS: SDSS / DESI
# ============================================================

def extract_date_from_hdul(hdul: fits.HDUList) -> tuple[str, float]:
    for h in hdul:
        hdr = h.header
        for key in ("DATE-OBS", "DATEOBS"):
            val = hdr.get(key)
            if val:
                s = str(val).strip()
                return s, parse_date_to_mjd(s)

    for h in hdul:
        hdr = h.header
        for key in ("MJD", "MJD-OBS", "MJDOBS", "MEANMJD", "MINMJD", "MAXMJD"):
            val = safe_float(hdr.get(key))
            if np.isfinite(val):
                return mjd_to_iso(val), val

    return "", np.nan


def extract_redshift_from_hdul(hdul: fits.HDUList) -> tuple[float, str]:
    keys = ["ZUSED", "REDSHIFT", "Z", "Z_OBJ", "ZOBJ", "SPEC_Z"]
    for h in hdul:
        hdr = h.header
        for key in keys:
            val = safe_float(hdr.get(key))
            if np.isfinite(val):
                return val, f"header:{key}"
    return np.nan, ""


def read_sdss_spectrum(hdul: fits.HDUList):
    for h in hdul:
        if not isinstance(h, (fits.BinTableHDU, fits.TableHDU)) or h.data is None:
            continue
        cols = {str(c).lower(): c for c in h.columns.names}
        if "flux" in cols and "loglam" in cols:
            flux = np.asarray(h.data[cols["flux"]], dtype=float)
            wave = 10 ** np.asarray(h.data[cols["loglam"]], dtype=float)
            ivar = np.asarray(h.data[cols["ivar"]], dtype=float) if "ivar" in cols else None
            return wave, flux, ivar
    raise ValueError("не распознан SDSS спектр")


def read_desi_spectrum(hdul: fits.HDUList):
    names = {h.name.upper(): i for i, h in enumerate(hdul)}

    if "WAVELENGTH" in names and "FLUX" in names:
        wave = np.asarray(hdul[names["WAVELENGTH"]].data, dtype=float).ravel()
        flux = np.asarray(hdul[names["FLUX"]].data, dtype=float).ravel()
        ivar = None
        if "IVAR" in names:
            ivar = np.asarray(hdul[names["IVAR"]].data, dtype=float).ravel()
        return wave, flux, ivar

    if hdul[0].data is not None:
        arr = np.asarray(hdul[0].data, dtype=float)
        if arr.ndim == 2 and arr.shape[0] >= 2:
            wave = arr[0, :].astype(float)
            flux = arr[1, :].astype(float)
            ivar = arr[2, :].astype(float) if arr.shape[0] >= 3 else None
            return wave, flux, ivar

    raise ValueError("не распознан DESI спектр")


def load_raw_spectrum(path: Path):
    survey_guess = guess_survey_from_path(path)

    with fits.open(path, memmap=False) as hdul:
        hdr0 = hdul[0].header

        survey = str(hdr0.get("SURVEY", survey_guess)).strip().upper() if hdr0.get("SURVEY") else survey_guess
        specid = safe_int(hdr0.get("SPECID"))
        targetid = safe_int(hdr0.get("TARGETID"))

        date_obs, date_mjd = extract_date_from_hdul(hdul)
        redshift, redshift_source = extract_redshift_from_hdul(hdul)

        if survey == "SDSS":
            wave, flux, ivar = read_sdss_spectrum(hdul)
        elif survey == "DESI":
            wave, flux, ivar = read_desi_spectrum(hdul)
        else:
            try:
                wave, flux, ivar = read_sdss_spectrum(hdul)
                survey = "SDSS"
            except Exception:
                wave, flux, ivar = read_desi_spectrum(hdul)
                survey = "DESI"

    return {
        "survey": survey,
        "specid": specid,
        "targetid": targetid,
        "date_obs": date_obs,
        "date_mjd": date_mjd,
        "redshift": redshift,
        "redshift_source": redshift_source,
        "wave_obs": wave,
        "flux": flux,
        "ivar": ivar,
    }


# ============================================================
# ИЗМЕРЕНИЕ КОНТИНУУМА 5100
# ============================================================

def estimate_integral_error(wave_rest_win: np.ndarray,
                            flux_win: np.ndarray,
                            flux_smooth_win: np.ndarray,
                            ivar_win: np.ndarray | None) -> float:
    dlam = np.gradient(wave_rest_win)

    if ivar_win is not None:
        good = np.isfinite(ivar_win) & (ivar_win > 0)
        if np.sum(good) >= max(5, len(wave_rest_win) // 2):
            sigma = np.full_like(wave_rest_win, np.nan, dtype=float)
            sigma[good] = 1.0 / np.sqrt(ivar_win[good])
            sigma = np.where(np.isfinite(sigma), sigma, np.nanmedian(sigma[good]))
            err = np.sqrt(np.nansum((sigma * dlam) ** 2))
            return float(err)

    # fallback: берём локальный шум как разброс raw-smoothed
    resid = flux_win - flux_smooth_win
    sigma_resid = np.nanstd(resid)
    if not np.isfinite(sigma_resid):
        sigma_resid = np.nanstd(flux_win)
    if not np.isfinite(sigma_resid):
        return np.nan

    err = sigma_resid * np.sqrt(np.nansum(dlam ** 2))
    return float(err)


def measure_continuum_5100(wave_obs: np.ndarray,
                           flux: np.ndarray,
                           ivar: np.ndarray | None,
                           redshift: float,
                           center_rest: float = CONT_CENTER_REST,
                           half_width: float = CONT_HALF_WIDTH,
                           smooth_sigma_pix: float = SMOOTH_SIGMA_PIX,
                           use_smoothing: bool = USE_SMOOTHING):
    wave_obs, flux, ivar = preprocess_spectrum(wave_obs, flux, ivar)

    if not np.isfinite(redshift):
        raise ValueError("не найден redshift")

    wave_rest = wave_obs / (1.0 + redshift)

    if use_smoothing and smooth_sigma_pix > 0:
        flux_smooth = gaussian_filter1d(flux, sigma=smooth_sigma_pix, mode="nearest")
    else:
        flux_smooth = flux.copy()

    wmin = center_rest - half_width
    wmax = center_rest + half_width
    mask = (wave_rest >= wmin) & (wave_rest <= wmax)

    if np.sum(mask) < 5:
        raise ValueError("слишком мало точек в окне 5100±50 Å")

    wave_rest_win = wave_rest[mask]
    flux_win = flux[mask]
    flux_smooth_win = flux_smooth[mask]
    ivar_win = ivar[mask] if ivar is not None else None

    cont_int_flux = float(np.trapz(flux_smooth_win, wave_rest_win))
    cont_mean_fluxdens = float(np.nanmean(flux_smooth_win))
    cont_median_fluxdens = float(np.nanmedian(flux_smooth_win))
    cont_err = estimate_integral_error(wave_rest_win, flux_win, flux_smooth_win, ivar_win)

    return {
        "wave_obs": wave_obs,
        "wave_rest": wave_rest,
        "flux_raw": flux,
        "flux_smooth": flux_smooth,
        "mask_window": mask,
        "wave_rest_win": wave_rest_win,
        "flux_win": flux_win,
        "flux_smooth_win": flux_smooth_win,
        "cont5100_int_flux": cont_int_flux,
        "cont5100_err": cont_err,
        "cont5100_mean_fluxdens": cont_mean_fluxdens,
        "cont5100_median_fluxdens": cont_median_fluxdens,
        "n_points_window": int(np.sum(mask)),
    }


def save_continuum_plot(out_path: Path,
                        object_name: str,
                        spectrum_name: str,
                        survey: str,
                        date_obs: str,
                        redshift: float,
                        meas: dict,
                        center_rest: float = CONT_CENTER_REST,
                        half_width: float = CONT_HALF_WIDTH):
    wave_rest = meas["wave_rest"]
    flux_raw = meas["flux_raw"]
    flux_smooth = meas["flux_smooth"]

    plot_mask = (wave_rest >= center_rest - 220) & (wave_rest <= center_rest + 220)
    if np.sum(plot_mask) < 10:
        plot_mask = np.ones_like(wave_rest, dtype=bool)

    fig, ax = plt.subplots(figsize=(10, 5.8))

    ax.plot(wave_rest[plot_mask], flux_raw[plot_mask], color="0.75", lw=1.0, label="Raw spectrum")
    ax.plot(wave_rest[plot_mask], flux_smooth[plot_mask], color="darkgreen", lw=1.6, label="Smoothed")

    ax.axvspan(center_rest - half_width, center_rest + half_width,
               color="gold", alpha=0.25, label="5100 ± 50 Å")

    ax.axvline(center_rest, color="black", ls="--", lw=1.0)

    ax.set_xlabel("Wavelength Å")
    ax.set_ylabel("Flux $10^{-17} erg cm^{-2} s^{-1} \AA^{-1}$")
    ax.set_title(
        f"{object_name} | {survey} | {spectrum_name}\n"
        f"date={date_obs or 'NA'} | z={redshift:.6f} | "
        f"Integral={meas['cont5100_int_flux']:.3e} ± {meas['cont5100_err']:.3e}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    #apply_article_axis_style(ax)
    apply_article_axis_style(ax)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# ПОДГОТОВКА МЕТАДАННЫХ
# ============================================================

def get_metadata_for_file(path: Path,
                          line_row: dict | None,
                          manifest_row: dict | None,
                          raw_meta: dict,
                          redshift_cache: dict[int, dict]):
    survey = raw_meta.get("survey", guess_survey_from_path(path))

    specid = raw_meta.get("specid")
    if specid is None and line_row:
        specid = safe_int(line_row.get("specid"))
    if specid is None and manifest_row:
        specid = safe_int(manifest_row.get("specid"))
    if specid is None:
        specid = extract_specid_from_filename(path)

    targetid = raw_meta.get("targetid")
    if targetid is None and line_row:
        targetid = safe_int(line_row.get("targetid"))
    if targetid is None and manifest_row:
        targetid = safe_int(manifest_row.get("targetid"))

    date_obs = ""
    date_mjd = np.nan

    if line_row:
        date_obs = str(line_row.get("date_obs", "")).strip() if line_row.get("date_obs") is not None else ""
        date_mjd = safe_float(line_row.get("date_mjd"))

    if (not date_obs or date_obs.lower() == "nan") and manifest_row:
        date_obs = str(manifest_row.get("date_obs", "")).strip() if manifest_row.get("date_obs") is not None else date_obs
    if not np.isfinite(date_mjd) and manifest_row:
        date_mjd = safe_float(manifest_row.get("mjd"))

    if (not date_obs or date_obs.lower() == "nan") and raw_meta.get("date_obs"):
        date_obs = str(raw_meta.get("date_obs", "")).strip()
    if not np.isfinite(date_mjd):
        date_mjd = safe_float(raw_meta.get("date_mjd"))

    if not np.isfinite(date_mjd) and date_obs:
        date_mjd = parse_date_to_mjd(date_obs)

    redshift = np.nan
    redshift_source = ""

    if line_row:
        redshift = safe_float(line_row.get("redshift"))
        if np.isfinite(redshift):
            redshift_source = str(line_row.get("redshift_source", "line_table"))

    if not np.isfinite(redshift) and manifest_row:
        redshift = safe_float(manifest_row.get("redshift"))
        if np.isfinite(redshift):
            redshift_source = "manifest"

    if not np.isfinite(redshift):
        redshift = safe_float(raw_meta.get("redshift"))
        if np.isfinite(redshift):
            redshift_source = str(raw_meta.get("redshift_source", "fits"))

    if not np.isfinite(redshift) and specid is not None and specid in redshift_cache:
        redshift = safe_float(redshift_cache[specid].get("redshift"))
        if np.isfinite(redshift):
            redshift_source = "redshift_cache"

    return {
        "survey": survey,
        "specid": specid,
        "targetid": targetid,
        "date_obs": date_obs if date_obs.lower() != "nan" else "",
        "date_mjd": date_mjd,
        "redshift": redshift,
        "redshift_source": redshift_source,
    }


# ============================================================
# ОБРАБОТКА ОДНОГО ОБЪЕКТА: КОНТИНУУМ
# ============================================================

def process_object_continuum(object_dir: Path, redshift_cache: dict[int, dict]) -> pd.DataFrame:
    print(f"\n=== {object_dir.name} ===")

    files = list_spectrum_files(object_dir)
    if not files:
        print("  Нет спектров")
        return pd.DataFrame()

    line_df = load_table_if_exists(object_dir / LINE_TABLE_NAME)
    line_map = table_to_map_by_filename(line_df)

    manifest_map = load_manifest_map(object_dir)

    rows = []

    for path in files:
        print(f"  -> {path.name}")

        try:
            raw_meta = load_raw_spectrum(path)
        except Exception as e:
            print(f"     Ошибка чтения FITS: {e}")
            rows.append({
                "object_name": object_dir.name,
                "survey": guess_survey_from_path(path),
                "spectrum_file": str(path),
                "file_name": path.name,
                "specid": extract_specid_from_filename(path),
                "targetid": None,
                "date_obs": "",
                "date_mjd": np.nan,
                "redshift": np.nan,
                "redshift_source": "",
                "cont5100_center_rest": CONT_CENTER_REST,
                "cont5100_half_width": CONT_HALF_WIDTH,
                "smooth_sigma_pix": SMOOTH_SIGMA_PIX if USE_SMOOTHING else 0.0,
                "cont5100_int_flux": np.nan,
                "cont5100_err": np.nan,
                "cont5100_mean_fluxdens": np.nan,
                "cont5100_median_fluxdens": np.nan,
                "n_points_window": 0,
                "status": "failed",
                "notes": f"fits read failed: {e}",
                "plot_file": "",
            })
            continue

        line_row = line_map.get(path.name)
        manifest_row = manifest_map.get(path.name)

        meta = get_metadata_for_file(path, line_row, manifest_row, raw_meta, redshift_cache)

        row = {
            "object_name": object_dir.name,
            "survey": meta["survey"],
            "spectrum_file": str(path),
            "file_name": path.name,
            "specid": meta["specid"],
            "targetid": meta["targetid"],
            "date_obs": meta["date_obs"],
            "date_mjd": meta["date_mjd"],
            "redshift": meta["redshift"],
            "redshift_source": meta["redshift_source"],
            "cont5100_center_rest": CONT_CENTER_REST,
            "cont5100_half_width": CONT_HALF_WIDTH,
            "smooth_sigma_pix": SMOOTH_SIGMA_PIX if USE_SMOOTHING else 0.0,
            "cont5100_int_flux": np.nan,
            "cont5100_err": np.nan,
            "cont5100_mean_fluxdens": np.nan,
            "cont5100_median_fluxdens": np.nan,
            "n_points_window": 0,
            "status": "failed",
            "notes": "",
            "plot_file": "",
        }

        if not np.isfinite(meta["redshift"]):
            row["notes"] = "redshift not found"
            rows.append(row)
            continue

        try:
            meas = measure_continuum_5100(
                wave_obs=raw_meta["wave_obs"],
                flux=raw_meta["flux"],
                ivar=raw_meta["ivar"],
                redshift=meta["redshift"],
                center_rest=CONT_CENTER_REST,
                half_width=CONT_HALF_WIDTH,
                smooth_sigma_pix=SMOOTH_SIGMA_PIX,
                use_smoothing=USE_SMOOTHING,
            )
        except Exception as e:
            row["notes"] = f"continuum measure failed: {e}"
            rows.append(row)
            continue

        plot_path = path.with_name(f"{strip_fits_suffix(path)}{CONT_PLOT_SUFFIX}")
        try:
            save_continuum_plot(
                out_path=plot_path,
                object_name=object_dir.name,
                spectrum_name=path.name,
                survey=meta["survey"],
                date_obs=meta["date_obs"],
                redshift=meta["redshift"],
                meas=meas,
                center_rest=CONT_CENTER_REST,
                half_width=CONT_HALF_WIDTH,
            )
            row["plot_file"] = str(plot_path)
        except Exception as e:
            row["notes"] = f"plot save failed: {e}"

        row["cont5100_int_flux"] = meas["cont5100_int_flux"]
        row["cont5100_err"] = meas["cont5100_err"]
        row["cont5100_mean_fluxdens"] = meas["cont5100_mean_fluxdens"]
        row["cont5100_median_fluxdens"] = meas["cont5100_median_fluxdens"]
        row["n_points_window"] = meas["n_points_window"]
        row["status"] = "ok" if np.isfinite(meas["cont5100_int_flux"]) else "failed"
        if not row["notes"]:
            row["notes"] = "ok"

        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["date_mjd"] = pd.to_numeric(df["date_mjd"], errors="coerce")
        df = df.sort_values(by=["date_mjd", "survey", "file_name"], ascending=[True, True, True]).reset_index(drop=True)

    out_path = object_dir / CONT_TABLE_NAME
    df.to_csv(out_path, sep="\t", index=False)
    print(f"  saved: {out_path.name}")

    return df


# ============================================================
# ОБЪЕДИНЕНИЕ С Hα / Hβ
# ============================================================

def merge_line_and_continuum(object_dir: Path, cont_df: pd.DataFrame) -> pd.DataFrame:
    line_df = load_table_if_exists(object_dir / LINE_TABLE_NAME)

    if line_df.empty and cont_df.empty:
        return pd.DataFrame()

    if line_df.empty:
        merged = cont_df.copy()
        if "ha_broad_flux" not in merged.columns:
            merged["ha_broad_flux"] = np.nan
            merged["ha_broad_err"] = np.nan
            merged["hb_broad_flux"] = np.nan
            merged["hb_broad_err"] = np.nan
            merged["ha_ok"] = False
            merged["hb_ok"] = False
        merged.to_csv(object_dir / MERGED_TABLE_NAME, sep="\t", index=False)
        return merged

    if cont_df.empty:
        merged = line_df.copy()
        if "cont5100_int_flux" not in merged.columns:
            merged["cont5100_int_flux"] = np.nan
            merged["cont5100_err"] = np.nan
            merged["cont5100_mean_fluxdens"] = np.nan
            merged["cont5100_median_fluxdens"] = np.nan
        merged.to_csv(object_dir / MERGED_TABLE_NAME, sep="\t", index=False)
        return merged

    merged_raw = pd.merge(
        line_df,
        cont_df,
        on="file_name",
        how="outer",
        suffixes=("_line", "_cont")
    )

    merged = pd.DataFrame()

    for col in ["object_name", "survey", "spectrum_file", "specid", "targetid", "date_obs", "date_mjd", "redshift"]:
        line_col = f"{col}_line"
        cont_col = f"{col}_cont"

        if line_col in merged_raw.columns and cont_col in merged_raw.columns:
            merged[col] = merged_raw[line_col].combine_first(merged_raw[cont_col])
        elif line_col in merged_raw.columns:
            merged[col] = merged_raw[line_col]
        elif cont_col in merged_raw.columns:
            merged[col] = merged_raw[cont_col]
        elif col in merged_raw.columns:
            merged[col] = merged_raw[col]
        else:
            merged[col] = pd.NA

    # поля из line_df
    for col in line_df.columns:
        if col == "file_name":
            continue
        if col.endswith("_line"):
            continue
        if col in merged.columns:
            continue
        src = f"{col}_line" if f"{col}_line" in merged_raw.columns else col
        if src in merged_raw.columns:
            merged[col] = merged_raw[src]

    # поля из cont_df
    for col in cont_df.columns:
        if col == "file_name":
            continue
        if col in merged.columns:
            continue
        src = f"{col}_cont" if f"{col}_cont" in merged_raw.columns else col
        if src in merged_raw.columns:
            merged[col] = merged_raw[src]

    merged["file_name"] = merged_raw["file_name"]

    for c in ["date_mjd", "ha_broad_flux", "ha_broad_err", "hb_broad_flux", "hb_broad_err",
              "cont5100_int_flux", "cont5100_err",
              "cont5100_mean_fluxdens", "cont5100_median_fluxdens"]:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce")

    if "date_obs" in merged.columns and "date_mjd" in merged.columns:
        missing = merged["date_mjd"].isna() & merged["date_obs"].notna()
        if missing.any():
            merged.loc[missing, "date_mjd"] = merged.loc[missing, "date_obs"].apply(parse_date_to_mjd)

    merged = merged.sort_values(by=["date_mjd", "survey", "file_name"], ascending=[True, True, True]).reset_index(drop=True)
    merged.to_csv(object_dir / MERGED_TABLE_NAME, sep="\t", index=False)
    print(f"  saved: {MERGED_TABLE_NAME}")

    return merged

# ============================================================
# ГРАФИКИ КРИВЫХ БЛЕСКА
# ============================================================

QUANTITY_STYLES = {
    "ha_broad_flux": {
        "err": "ha_broad_err",
        "color": "crimson",
        "label": "Hα broad",
    },
    "hb_broad_flux": {
        "err": "hb_broad_err",
        "color": "royalblue",
        "label": "Hβ broad",
    },
    "cont5100_int_flux": {
        "err": "cont5100_err_plot",
        "color": "seagreen",
        "label": "Continuum 5100±50 Å",
    },
}

SURVEY_MARKERS = {
    "SDSS": "o",
    "DESI": "s",
    "OTHER": "^",
}

# Ошибки континуума 5100 Å из итоговой LaTeX-таблицы.
# Формат:
# object_name: [(survey, mjd, flux5100, err5100), ...]
CONTINUUM_5100_TABLE_ERRORS = {
    "J160700.60+553809.2": [
        ("SDSS", 56809.24, 2379.0, 11.1),
        ("DESI", 59312.44, 2056.4, 10.8),
        ("DESI", 59333.39, 2159.4, 25.1),
    ],
    "J110538.73+304959.2": [
        ("SDSS", 53472.00, 3008.5, 81.8),
        ("DESI", 59632.31, 2928.3, 115.0),
    ],
    "J121540.80+615323.4": [
        ("SDSS", 52342.00, 10589.2, 464.8),
        ("DESI", 59600.46, 10443.6, 337.4),
    ],
    "J130009.14+275159.2": [
        ("SDSS", 54156.00, 5032.5, 650.5),
        ("DESI", 59321.24, 1660.5, 124.0),
    ],
    "J131229.55+340321.4": [
        ("SDSS", 53818.00, 2346.9, 193.1),
        ("DESI", 59351.20, 2408.7, 330.2),
    ],
    "J132523.35+593643.4": [
        ("SDSS", 56684.40, 4413.2, 208.3),
        ("DESI", 59738.14, 6531.0, 938.0),
    ],
    "J162303.13+433626.3": [
        ("SDSS", 56076.33, 2895.3, 96.6),
        ("DESI", 59316.46, 2916.5, 154.2),
    ],
    "J171246.14+231324.8": [
        ("SDSS", 55706.38, 5060.7, 268.3),
        ("DESI", 59385.30, 13.9, 4688.6),
    ],
}

def apply_article_axis_style(ax):
    if TICKS_INWARD:
        ax.tick_params(
            axis="both",
            which="both",
            direction="in",
            top=True,
            right=True,
        )
        ax.minorticks_on()


def find_continuum_error_from_table(object_name: str, survey: str, date_mjd: float) -> tuple[float, float]:
    object_name = str(object_name).strip()
    survey = str(survey).strip().upper()
    date_mjd = safe_float(date_mjd)

    if not object_name or not np.isfinite(date_mjd):
        return np.nan, np.nan

    rows = CONTINUUM_5100_TABLE_ERRORS.get(object_name, [])
    best = None
    best_dmjd = np.inf

    for row_survey, row_mjd, row_flux, row_err in rows:
        if str(row_survey).upper() != survey:
            continue

        dmjd = abs(float(row_mjd) - date_mjd)
        if dmjd < best_dmjd:
            best = (safe_float(row_flux), safe_float(row_err))
            best_dmjd = dmjd

    if best is None or best_dmjd > CONT_ERR_MJD_TOL:
        return np.nan, np.nan

    return best


def is_good_continuum_error(flux_value: float, err_value: float) -> bool:
    flux_value = safe_float(flux_value)
    err_value = safe_float(err_value)

    if not np.isfinite(flux_value) or not np.isfinite(err_value):
        return False
    if flux_value <= 0 or err_value <= 0:
        return False
    if err_value > MAX_CONT_ERR_FRACTION * abs(flux_value):
        return False

    return True


def add_continuum_errors_for_plot(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "cont5100_int_flux" not in out.columns:
        out["cont5100_int_flux"] = np.nan
    if "cont5100_err" not in out.columns:
        out["cont5100_err"] = np.nan
    if "cont5100_err_plot" not in out.columns:
        out["cont5100_err_plot"] = np.nan

    out["cont5100_int_flux"] = pd.to_numeric(out["cont5100_int_flux"], errors="coerce")
    out["cont5100_err"] = pd.to_numeric(out["cont5100_err"], errors="coerce")

    for idx, row in out.iterrows():
        object_name = row.get("object_name", "")
        survey = row.get("survey", "")
        date_mjd = safe_float(row.get("date_mjd"))

        table_flux, table_err = find_continuum_error_from_table(object_name, survey, date_mjd)

        if is_good_continuum_error(table_flux, table_err):
            out.loc[idx, "cont5100_err_plot"] = table_err
            continue

        current_flux = safe_float(row.get("cont5100_int_flux"))
        current_err = safe_float(row.get("cont5100_err"))

        if is_good_continuum_error(current_flux, current_err):
            out.loc[idx, "cont5100_err_plot"] = current_err
        else:
            out.loc[idx, "cont5100_err_plot"] = np.nan

    skipped = out[
        out["cont5100_int_flux"].notna()
        & out["cont5100_err_plot"].isna()
    ]

    if not skipped.empty:
        bad_names = skipped[["object_name", "survey", "date_mjd"]].astype(str).agg(" | ".join, axis=1).tolist()
        print(f"  [WARN] Ошибки континуума не нарисованы для: {bad_names}")

    return out

def prepare_plot_dataframe(df: pd.DataFrame, object_dir: Path | None = None) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out

    if "date_mjd" not in out.columns:
        out["date_mjd"] = np.nan
    out["date_mjd"] = pd.to_numeric(out["date_mjd"], errors="coerce")

    if "date_obs" in out.columns:
        missing = out["date_mjd"].isna() & out["date_obs"].notna()
        if missing.any():
            out.loc[missing, "date_mjd"] = out.loc[missing, "date_obs"].apply(parse_date_to_mjd)

    if "survey" not in out.columns:
        out["survey"] = "OTHER"
    out["survey"] = out["survey"].fillna("OTHER").astype(str).str.upper()
    out.loc[~out["survey"].isin(["SDSS", "DESI"]), "survey"] = "OTHER"

    out = add_continuum_errors_for_plot(out)

    needed_any = []
    for col, st in QUANTITY_STYLES.items():
        if col not in out.columns:
            out[col] = np.nan
        if st["err"] not in out.columns:
            out[st["err"]] = np.nan

        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[st["err"]] = pd.to_numeric(out[st["err"]], errors="coerce")

        out.loc[~np.isfinite(out[st["err"]]) | (out[st["err"]] <= 0), st["err"]] = np.nan

        needed_any.append(out[col].notna())

    mask_any = needed_any[0]
    for m in needed_any[1:]:
        mask_any = mask_any | m

    out = out[mask_any & out["date_mjd"].notna()].copy()
    if out.empty:
        return out

    out["date_dt"] = out["date_mjd"].apply(mjd_to_datetime)
    out = out.sort_values("date_mjd").reset_index(drop=True)
    return out

def split_into_time_segments(df: pd.DataFrame, gap_threshold_days: float) -> list[pd.DataFrame]:
    if df.empty:
        return []

    mjd = df["date_mjd"].to_numpy(dtype=float)
    if len(mjd) == 1:
        return [df.copy()]

    breaks = np.where(np.diff(mjd) > gap_threshold_days)[0]
    segments = []
    start = 0
    for b in breaks:
        segments.append(df.iloc[start:b + 1].copy())
        start = b + 1
    segments.append(df.iloc[start:].copy())
    return segments


def compute_width_ratios(segments: list[pd.DataFrame]) -> list[float]:
    ratios = []
    for seg in segments:
        if len(seg) <= 1:
            span = MIN_PAD_DAYS_SINGLE_POINT
        else:
            span = float(seg["date_mjd"].max() - seg["date_mjd"].min())
            span = max(span, MIN_PAD_DAYS_MULTI_POINT)
        width = np.clip(np.log10(span + 10.0) + 1.0, 1.2, 3.2)
        ratios.append(float(width))
    return ratios

def compute_ylim(df: pd.DataFrame, flux_col: str, err_col: str) -> tuple[float, float]:
    good = df[flux_col].notna()
    if not good.any():
        return -1.0, 1.0

    y = df.loc[good, flux_col].to_numpy(dtype=float)

    if err_col in df.columns:
        e = df.loc[good, err_col].to_numpy(dtype=float)
        e = np.where(np.isfinite(e) & (e > 0), e, 0.0)
    else:
        e = np.zeros_like(y)

    y_low = np.nanmin(y - e)
    y_high = np.nanmax(y + e)

    if not np.isfinite(y_low) or not np.isfinite(y_high):
        y_low = np.nanmin(y)
        y_high = np.nanmax(y)

    if not np.isfinite(y_low) or not np.isfinite(y_high):
        return -1.0, 1.0

    if y_low == y_high:
        pad = max(abs(y_low) * 0.15, 1.0)
    else:
        pad = 0.12 * (y_high - y_low)

    return y_low - pad, y_high + pad

def set_segment_xlim(ax, seg: pd.DataFrame):
    d0 = seg["date_dt"].min()
    d1 = seg["date_dt"].max()

    if pd.Timestamp(d0) == pd.Timestamp(d1):
        pad = timedelta(days=MIN_PAD_DAYS_SINGLE_POINT / 2.0)
    else:
        span_days = max((pd.Timestamp(d1) - pd.Timestamp(d0)).days, MIN_PAD_DAYS_MULTI_POINT)
        pad = timedelta(days=max(0.08 * span_days, MIN_PAD_DAYS_MULTI_POINT))

    ax.set_xlim(pd.Timestamp(d0) - pad, pd.Timestamp(d1) + pad)


def draw_break_marks(ax_left, ax_right, d: float = 0.012):
    kwargs = dict(color="k", clip_on=False, lw=1.0)
    ax_left.plot((1 - d, 1 + d), (-d, +d), transform=ax_left.transAxes, **kwargs)
    ax_left.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax_left.transAxes, **kwargs)
    ax_right.plot((-d, +d), (-d, +d), transform=ax_right.transAxes, **kwargs)
    ax_right.plot((-d, +d), (1 - d, 1 + d), transform=ax_right.transAxes, **kwargs)


def segment_label(seg: pd.DataFrame) -> str:
    d0 = seg["date_dt"].min()
    d1 = seg["date_dt"].max()
    if pd.Timestamp(d0) == pd.Timestamp(d1):
        return pd.Timestamp(d0).strftime("%Y-%m-%d")
    return f"{pd.Timestamp(d0).strftime('%Y-%m-%d')} — {pd.Timestamp(d1).strftime('%Y-%m-%d')}"

def apply_date_format(ax):
    locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    for lab in ax.get_xticklabels():
        lab.set_rotation(25)
        lab.set_ha("right")

    apply_article_axis_style(ax)

def plot_quantity_on_axis(ax, seg: pd.DataFrame, flux_col: str, err_col: str, color: str):
    data = seg[seg[flux_col].notna()].copy()
    if data.empty:
        ax.text(0.5, 0.5, "Нет данных", ha="center", va="center", transform=ax.transAxes)
        ax.grid(True, alpha=0.3)
        set_segment_xlim(ax, seg)
        apply_date_format(ax)
        return

    data_all = data.sort_values("date_mjd")
    ax.plot(data_all["date_dt"], data_all[flux_col], color=color, lw=1.2, alpha=0.75, zorder=2)

    for survey in ["SDSS", "DESI", "OTHER"]:
        grp = data[data["survey"] == survey].copy().sort_values("date_mjd")
        if grp.empty:
            continue

        marker = SURVEY_MARKERS[survey]

        if err_col in grp.columns:
            yerr = grp[err_col].to_numpy(dtype=float)
            yerr = np.where(np.isfinite(yerr) & (yerr > 0), yerr, np.nan)
        else:
            yerr = np.full(len(grp), np.nan)

        good_err = np.isfinite(yerr) & (yerr > 0)

        if np.any(good_err):
            ax.errorbar(
                grp.loc[good_err, "date_dt"],
                grp.loc[good_err, flux_col],
                yerr=yerr[good_err],
                fmt=marker,
                color=color,
                ecolor=color,
                elinewidth=1.0,
                capsize=3,
                markersize=6,
                linestyle="none",
                alpha=0.95,
                zorder=3,
            )

        if np.any(~good_err):
            ax.plot(
                grp.loc[~good_err, "date_dt"],
                grp.loc[~good_err, flux_col],
                marker=marker,
                color=color,
                markersize=6,
                linestyle="none",
                alpha=0.95,
                zorder=3,
            )

    ax.grid(True, linestyle="--", alpha=0.35)
    set_segment_xlim(ax, seg)
    apply_date_format(ax)

def make_absolute_lightcurve(object_dir: Path, df: pd.DataFrame):
    dfp = prepare_plot_dataframe(df, object_dir=object_dir)
    if dfp.empty:
        return

    segments = split_into_time_segments(dfp, GAP_THRESHOLD_DAYS)
    nseg = len(segments)
    width_ratios = compute_width_ratios(segments)

    fig, axes = plt.subplots(
        3,
        nseg,
        figsize=(4.8 * nseg + 1.5, 10.5),
        sharey="row",
        gridspec_kw={"width_ratios": width_ratios},
        squeeze=False,
    )

    flux_unit_label = format_flux_unit_label(FLUX_UNIT_POWER)

    rows_info = [
        ("ha_broad_flux", "ha_broad_err", "Hα broad flux"),
        ("hb_broad_flux", "hb_broad_err", "Hβ broad flux"),
        ("cont5100_int_flux", "cont5100_err_plot", "Continuum 5100±50 Å flux"),
    ]

    ylims = {}
    for flux_col, err_col, _ in rows_info:
        ylims[flux_col] = compute_ylim(dfp, flux_col, err_col)

    for j, seg in enumerate(segments):
        for i, (flux_col, err_col, ylabel) in enumerate(rows_info):
            ax = axes[i, j]
            style = QUANTITY_STYLES[flux_col]
            plot_quantity_on_axis(ax, seg, flux_col, err_col, style["color"])

            ax.set_ylim(*ylims[flux_col])
            if ABSOLUTE_LOG_Y:
                vals = dfp[flux_col].dropna().to_numpy(dtype=float)
                if len(vals) > 0 and np.all(vals > 0):
                    ax.set_yscale("log")

            if j == 0:
                ax.set_ylabel(ylabel)
            else:
                ax.tick_params(axis="y", labelleft=False)

            if i == 0:
                ax.set_title(segment_label(seg), fontsize=11)

            if i == 2:
                ax.set_xlabel("Date")

            if j < nseg - 1:
                ax.spines["right"].set_visible(False)
            if j > 0:
                ax.spines["left"].set_visible(False)

            apply_article_axis_style(ax)

    if nseg > 1:
        for j in range(nseg - 1):
            for i in range(3):
                draw_break_marks(axes[i, j], axes[i, j + 1])

    quantity_handles = [
        Line2D([0], [0], color=QUANTITY_STYLES["ha_broad_flux"]["color"], lw=1.5, label="Hα broad"),
        Line2D([0], [0], color=QUANTITY_STYLES["hb_broad_flux"]["color"], lw=1.5, label="Hβ broad"),
        Line2D([0], [0], color=QUANTITY_STYLES["cont5100_int_flux"]["color"], lw=1.5, label="Continuum 5100±50"),
    ]
    survey_handles = [
        Line2D([0], [0], color="black", marker="o", linestyle="none", markersize=6, label="SDSS"),
        Line2D([0], [0], color="black", marker="s", linestyle="none", markersize=6, label="DESI"),
    ]

    fig.suptitle(f"{object_dir.name}\nAbsolute light curves", fontsize=14, y=0.985)
    fig.legend(handles=quantity_handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.955))
    fig.legend(handles=survey_handles, loc="upper right", ncol=2, frameon=False, bbox_to_anchor=(0.98, 0.955))

    fig.text(
        0.5, 0.012,
        f"Time gaps > {GAP_THRESHOLD_DAYS:.0f} days are shown as compressed breaks",
        ha="center",
        va="bottom",
        fontsize=9,
        alpha=0.8,
    )

    plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.92])
    out_path = object_dir / LIGHTCURVE_ABS_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

def make_normalized_lightcurve(object_dir: Path, df: pd.DataFrame):
    dfp = prepare_plot_dataframe(df, object_dir=object_dir)
    if dfp.empty:
        return

    # Создаём нормированные колонки
    norm_df = dfp.copy()
    ok_any = False

    for flux_col, style in QUANTITY_STYLES.items():
        err_col = style["err"]
        norm_col = f"{flux_col}_norm"
        norm_err_col = f"{err_col}_norm"

        vals = norm_df[flux_col].dropna().to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        vals = vals[vals > 0]

        if len(vals) == 0:
            norm_df[norm_col] = np.nan
            norm_df[norm_err_col] = np.nan
            continue

        med = np.nanmedian(vals)
        if not np.isfinite(med) or med == 0:
            norm_df[norm_col] = np.nan
            norm_df[norm_err_col] = np.nan
            continue

        norm_df[norm_col] = norm_df[flux_col] / med
        norm_df[norm_err_col] = norm_df[err_col] / med

        if err_col in norm_df.columns:
            err_values = pd.to_numeric(norm_df[err_col], errors="coerce")
            err_values = err_values.where(np.isfinite(err_values) & (err_values > 0), np.nan)
            norm_df[norm_err_col] = err_values / med
        else:
            norm_df[norm_err_col] = np.nan

        ok_any = True

    if not ok_any:
        return

    segments = split_into_time_segments(norm_df, GAP_THRESHOLD_DAYS)
    nseg = len(segments)
    width_ratios = compute_width_ratios(segments)

    fig, axes = plt.subplots(
        1,
        nseg,
        figsize=(4.8 * nseg + 1.5, 4.8),
        sharey=True,
        gridspec_kw={"width_ratios": width_ratios},
        squeeze=False,
    )
    axes = axes[0]

    # Общий ylim
    yvals = []
    for flux_col, style in QUANTITY_STYLES.items():
        ncol = f"{flux_col}_norm"
        ecol = f"{style['err']}_norm"

        if ncol in norm_df.columns:
            v = norm_df[ncol].to_numpy(dtype=float)

            if ecol in norm_df.columns:
                e = norm_df[ecol].to_numpy(dtype=float)
                e = np.where(np.isfinite(e) & (e > 0), e, 0.0)
            else:
                e = np.zeros_like(v)

            mask = np.isfinite(v)
            if np.any(mask):
                yvals.extend(list(v[mask] - e[mask]))
                yvals.extend(list(v[mask] + e[mask]))

    if len(yvals) == 0:
        return

    ymin = np.nanmin(yvals)
    ymax = np.nanmax(yvals)

    if not np.isfinite(ymin) or not np.isfinite(ymax):
        return

    if ymin == ymax:
        pad = 0.15 * abs(ymin) if ymin != 0 else 0.5
    else:
        pad = 0.12 * (ymax - ymin)

    ylim = (ymin - pad, ymax + pad)

    for j, seg in enumerate(segments):
        ax = axes[j]

        for flux_col, style in QUANTITY_STYLES.items():
            ncol = f"{flux_col}_norm"
            ecol = f"{style['err']}_norm"

            if ncol not in seg.columns:
                continue

            data = seg[seg[ncol].notna()].copy()
            if data.empty:
                continue

            data_all = data.sort_values("date_mjd")
            ax.plot(
                data_all["date_dt"],
                data_all[ncol],
                color=style["color"],
                lw=1.2,
                alpha=0.8,
                zorder=2,
            )

            for survey in ["SDSS", "DESI", "OTHER"]:
                grp = data[data["survey"] == survey].copy().sort_values("date_mjd")
                if grp.empty:
                    continue

                marker = SURVEY_MARKERS[survey]

                if ecol in grp.columns:
                    yerr = grp[ecol].to_numpy(dtype=float)
                    yerr = np.where(np.isfinite(yerr) & (yerr > 0), yerr, np.nan)
                else:
                    yerr = np.full(len(grp), np.nan)

                good_err = np.isfinite(yerr) & (yerr > 0)

                # Точки с ошибками
                if np.any(good_err):
                    ax.errorbar(
                        grp.loc[good_err, "date_dt"],
                        grp.loc[good_err, ncol],
                        yerr=yerr[good_err],
                        fmt=marker,
                        color=style["color"],
                        ecolor=style["color"],
                        elinewidth=1.0,
                        capsize=3,
                        markersize=6,
                        linestyle="none",
                        alpha=0.95,
                        zorder=3,
                    )

                # Точки без ошибок
                if np.any(~good_err):
                    ax.plot(
                        grp.loc[~good_err, "date_dt"],
                        grp.loc[~good_err, ncol],
                        marker=marker,
                        color=style["color"],
                        markersize=6,
                        linestyle="none",
                        alpha=0.95,
                        zorder=3,
                    )

        ax.set_ylim(*ylim)
        ax.grid(True, linestyle="--", alpha=0.35)
        set_segment_xlim(ax, seg)
        apply_date_format(ax)
        ax.set_title(segment_label(seg), fontsize=11)

        if j == 0:
            ax.set_ylabel("Normalized flux")

        ax.set_xlabel("Date")

        if j < nseg - 1:
            ax.spines["right"].set_visible(False)
        if j > 0:
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", labelleft=False)

        apply_article_axis_style(ax)

    if nseg > 1:
        for j in range(nseg - 1):
            draw_break_marks(axes[j], axes[j + 1])

    quantity_handles = [
        Line2D([0], [0], color=QUANTITY_STYLES["ha_broad_flux"]["color"], lw=1.5, label="Hα broad"),
        Line2D([0], [0], color=QUANTITY_STYLES["hb_broad_flux"]["color"], lw=1.5, label="Hβ broad"),
        Line2D([0], [0], color=QUANTITY_STYLES["cont5100_int_flux"]["color"], lw=1.5, label="Continuum 5100±50"),
    ]
    survey_handles = [
        Line2D([0], [0], color="black", marker="o", linestyle="none", markersize=6, label="SDSS"),
        Line2D([0], [0], color="black", marker="s", linestyle="none", markersize=6, label="DESI"),
    ]

    fig.suptitle(f"{object_dir.name}\nNormalized combined light curve", fontsize=14, y=0.98)
    fig.legend(handles=quantity_handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.93))
    fig.legend(handles=survey_handles, loc="upper right", ncol=2, frameon=False, bbox_to_anchor=(0.98, 0.93))

    fig.text(
        0.5,
        0.02,
        f"Each series is divided by its median. Time gaps > {GAP_THRESHOLD_DAYS:.0f} days are compressed.",
        ha="center",
        va="bottom",
        fontsize=9,
        alpha=0.8,
    )

    plt.tight_layout(rect=[0.02, 0.05, 0.98, 0.88])
    out_path = object_dir / LIGHTCURVE_NORM_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    if not ROOT_DIR.exists():
        raise FileNotFoundError(f"Не найдена директория: {ROOT_DIR}")

    object_dirs = find_object_dirs(ROOT_DIR)
    if not object_dirs:
        print("Папки объектов не найдены.")
        return

    redshift_cache = load_redshift_cache(ROOT_DIR)

    n_ok = 0
    n_fail = 0

    for object_dir in object_dirs:
        try:
            cont_df = process_object_continuum(object_dir, redshift_cache)
            merged_df = merge_line_and_continuum(object_dir, cont_df)

            if not merged_df.empty:
                make_absolute_lightcurve(object_dir, merged_df)
                make_normalized_lightcurve(object_dir, merged_df)

            n_ok += 1
        except Exception as e:
            n_fail += 1
            print(f"[ERR] {object_dir.name}: {e}")

    print("\nГотово.")
    print(f"Успешно обработано объектов: {n_ok}")
    print(f"С ошибками: {n_fail}")


if __name__ == "__main__":
    main()