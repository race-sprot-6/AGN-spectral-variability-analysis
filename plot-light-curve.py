from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from astropy.time import Time


# ============================================================
# НАСТРОЙКИ
# ============================================================

ROOT_DIR = Path(r"C:\Users\IvanK\Astro\Course5\downloaded_spectra")
OBJECT_TABLE_NAME = "line_fluxes_vs_time.tsv"
OUT_PLOT_NAME = "light_curve_broad_Ha_Hb.png"

# Если разрыв между соседними эпохами больше этого порога,
# график разбивается на соседние временные сегменты.
GAP_THRESHOLD_DAYS = 1000.0

# Отступ по времени слева/справа внутри сегмента
MIN_PAD_DAYS_SINGLE_POINT = 40.0
MIN_PAD_DAYS_MULTI_POINT = 15.0

# Рисовать ли линии между точками одного survey
CONNECT_POINTS = True

SURVEY_STYLES = {
    "SDSS": {"color": "royalblue", "marker": "o", "label": "SDSS"},
    "DESI": {"color": "darkorange", "marker": "s", "label": "DESI"},
    "OTHER": {"color": "gray", "marker": "^", "label": "Other"},
}


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


def parse_date_to_mjd(date_str: str) -> float:
    if not date_str or str(date_str).strip().lower() == "nan":
        return np.nan

    s = str(date_str).strip()
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


def find_object_dirs(root: Path) -> list[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("J")])


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Нормализуем названия колонок
    out.columns = [str(c).strip() for c in out.columns]

    if "date_mjd" not in out.columns:
        out["date_mjd"] = np.nan

    out["date_mjd"] = pd.to_numeric(out["date_mjd"], errors="coerce")

    # Если MJD отсутствует — пытаемся получить из date_obs
    if "date_obs" in out.columns:
        missing = out["date_mjd"].isna()
        if missing.any():
            out.loc[missing, "date_mjd"] = out.loc[missing, "date_obs"].apply(parse_date_to_mjd)

    # Нужны строки хотя бы с одной валидной линией
    for col in ["ha_broad_flux", "ha_broad_err", "hb_broad_flux", "hb_broad_err"]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if "survey" not in out.columns:
        out["survey"] = "OTHER"

    out["survey"] = out["survey"].fillna("OTHER").astype(str).str.upper()
    out.loc[~out["survey"].isin(["SDSS", "DESI"]), "survey"] = "OTHER"

    mask_any_flux = out["ha_broad_flux"].notna() | out["hb_broad_flux"].notna()
    mask_date = out["date_mjd"].notna()

    out = out[mask_any_flux & mask_date].copy()
    if out.empty:
        return out

    out = out.sort_values("date_mjd").reset_index(drop=True)
    out["date_dt"] = out["date_mjd"].apply(mjd_to_datetime)

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

        # Логарифмическое сжатие, чтобы длинные сегменты не были слишком широкими
        width = np.clip(np.log10(span + 10.0) + 1.0, 1.2, 3.2)
        ratios.append(float(width))

    return ratios


def compute_ylim(df: pd.DataFrame, flux_col: str, err_col: str) -> tuple[float, float]:
    good = df[flux_col].notna()
    if not good.any():
        return -1.0, 1.0

    y = df.loc[good, flux_col].to_numpy(dtype=float)
    e = df.loc[good, err_col].fillna(0.0).to_numpy(dtype=float)

    y_low = np.min(y - e)
    y_high = np.max(y + e)

    if not np.isfinite(y_low) or not np.isfinite(y_high):
        return -1.0, 1.0

    if y_low == y_high:
        pad = max(abs(y_low) * 0.15, 1.0)
    else:
        pad = 0.12 * (y_high - y_low)

    return y_low - pad, y_high + pad


def segment_label(seg: pd.DataFrame) -> str:
    d0 = seg["date_dt"].min()
    d1 = seg["date_dt"].max()

    if pd.Timestamp(d0) == pd.Timestamp(d1):
        return pd.Timestamp(d0).strftime("%Y-%m-%d")

    return f"{pd.Timestamp(d0).strftime('%Y-%m-%d')} — {pd.Timestamp(d1).strftime('%Y-%m-%d')}"


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

    # справа у левой оси
    ax_left.plot((1 - d, 1 + d), (-d, +d), transform=ax_left.transAxes, **kwargs)
    ax_left.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax_left.transAxes, **kwargs)

    # слева у правой оси
    ax_right.plot((-d, +d), (-d, +d), transform=ax_right.transAxes, **kwargs)
    ax_right.plot((-d, +d), (1 - d, 1 + d), transform=ax_right.transAxes, **kwargs)


def get_style_for_survey(survey: str) -> dict:
    return SURVEY_STYLES.get(survey.upper(), SURVEY_STYLES["OTHER"])


def plot_line_row(ax, seg: pd.DataFrame, flux_col: str, err_col: str):
    has_any = False

    for survey in ["SDSS", "DESI", "OTHER"]:
        grp = seg[seg["survey"] == survey].copy()
        grp = grp[grp[flux_col].notna()].sort_values("date_mjd")

        if grp.empty:
            continue

        has_any = True
        style = get_style_for_survey(survey)

        x = grp["date_dt"].to_list()
        y = grp[flux_col].to_numpy(dtype=float)
        yerr = grp[err_col].fillna(0.0).to_numpy(dtype=float)

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt=style["marker"],
            color=style["color"],
            ecolor=style["color"],
            elinewidth=1.0,
            capsize=3,
            markersize=6,
            linestyle="none",
            alpha=0.95,
            zorder=3,
        )

        if CONNECT_POINTS and len(grp) >= 2:
            ax.plot(
                x,
                y,
                color=style["color"],
                lw=1.2,
                alpha=0.8,
                zorder=2,
            )

    if not has_any:
        ax.text(
            0.5,
            0.5,
            "Нет данных",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
        )

    locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    ax.grid(True, linestyle="--", alpha=0.35)
    set_segment_xlim(ax, seg)

    for label in ax.get_xticklabels():
        label.set_rotation(25)
        label.set_ha("right")


def build_legend_handles():
    handles = []
    for survey in ["SDSS", "DESI", "OTHER"]:
        style = get_style_for_survey(survey)
        handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                marker=style["marker"],
                linestyle="-",
                markersize=6,
                label=style["label"],
            )
        )
    return handles


# ============================================================
# ПОСТРОЕНИЕ ОДНОГО ОБЪЕКТА
# ============================================================

def plot_object_light_curve(object_dir: Path) -> bool:
    table_path = object_dir / OBJECT_TABLE_NAME
    if not table_path.exists():
        print(f"[SKIP] {object_dir.name}: нет {OBJECT_TABLE_NAME}")
        return False

    try:
        df = pd.read_csv(table_path, sep="\t")
    except Exception as e:
        print(f"[SKIP] {object_dir.name}: ошибка чтения таблицы: {e}")
        return False

    df = prepare_dataframe(df)
    if df.empty:
        print(f"[SKIP] {object_dir.name}: нет валидных дат/потоков")
        return False

    segments = split_into_time_segments(df, GAP_THRESHOLD_DAYS)
    if not segments:
        print(f"[SKIP] {object_dir.name}: пустые сегменты")
        return False

    nseg = len(segments)
    width_ratios = compute_width_ratios(segments)

    fig, axes = plt.subplots(
        2,
        nseg,
        figsize=(4.8 * nseg + 1.5, 8.0),
        sharey="row",
        gridspec_kw={"width_ratios": width_ratios},
        squeeze=False,
    )

    ha_ylim = compute_ylim(df, "ha_broad_flux", "ha_broad_err")
    hb_ylim = compute_ylim(df, "hb_broad_flux", "hb_broad_err")

    # Рисуем по сегментам
    for j, seg in enumerate(segments):
        ax_ha = axes[0, j]
        ax_hb = axes[1, j]

        plot_line_row(ax_ha, seg, "ha_broad_flux", "ha_broad_err")
        plot_line_row(ax_hb, seg, "hb_broad_flux", "hb_broad_err")

        ax_ha.set_ylim(*ha_ylim)
        ax_hb.set_ylim(*hb_ylim)

        ax_ha.set_title(segment_label(seg), fontsize=11)

        if j == 0:
            ax_ha.set_ylabel("Hα broad flux")
            ax_hb.set_ylabel("Hβ broad flux")
        else:
            ax_ha.tick_params(axis="y", labelleft=False)
            ax_hb.tick_params(axis="y", labelleft=False)

        ax_hb.set_xlabel("Date")

        # немного убираем внутренние рамки
        if j < nseg - 1:
            ax_ha.spines["right"].set_visible(False)
            ax_hb.spines["right"].set_visible(False)
        if j > 0:
            ax_ha.spines["left"].set_visible(False)
            ax_hb.spines["left"].set_visible(False)

    # Рисуем "разрывы" между сегментами
    if nseg > 1:
        for j in range(nseg - 1):
            draw_break_marks(axes[0, j], axes[0, j + 1])
            draw_break_marks(axes[1, j], axes[1, j + 1])

    # Заголовок
    fig.suptitle(
        f"{object_dir.name}\nBroad-line light curves (Hα, Hβ)",
        fontsize=14,
        y=0.98,
    )

    # Общая легенда
    handles = build_legend_handles()
    fig.legend(
        handles=handles[:2],  # показываем SDSS и DESI, OTHER обычно не нужен
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.94),
    )

    # Подпись о разрывах
    fig.text(
        0.5,
        0.015,
        f"Time gaps > {GAP_THRESHOLD_DAYS:.0f} days are shown as compressed breaks",
        ha="center",
        va="bottom",
        fontsize=9,
        alpha=0.8,
    )

    plt.tight_layout(rect=[0.02, 0.04, 0.98, 0.90])

    out_path = object_dir / OUT_PLOT_NAME
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK]   {object_dir.name}: {out_path.name}")
    return True


# ============================================================
# MAIN
# ============================================================

def main():
    if not ROOT_DIR.exists():
        raise FileNotFoundError(f"Не найдена директория: {ROOT_DIR}")

    object_dirs = find_object_dirs(ROOT_DIR)
    if not object_dirs:
        print("Папки объектов не найдены.")
        return

    n_ok = 0
    n_skip = 0

    for object_dir in object_dirs:
        try:
            ok = plot_object_light_curve(object_dir)
            if ok:
                n_ok += 1
            else:
                n_skip += 1
        except Exception as e:
            n_skip += 1
            print(f"[ERR]  {object_dir.name}: {e}")

    print("\nГотово.")
    print(f"Построено графиков: {n_ok}")
    print(f"Пропущено объектов: {n_skip}")


if __name__ == "__main__":
    main()