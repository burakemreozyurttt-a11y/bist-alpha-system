import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

TR_TZ = timezone(timedelta(hours=3))
BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "templates"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def _clamp(v, lo=0.0, hi=100.0):
    try:
        return max(lo, min(hi, float(v)))
    except Exception:
        return 50.0


def _safe_float(v, default=None):
    try:
        if v is None or v == "N/A":
            return default
        x = float(v)
        if math.isnan(x):
            return default
        return x
    except Exception:
        return default


def confidence_label(score):
    score = _safe_float(score, 0) or 0
    if score >= 80:
        return "Yüksek"
    if score >= 60:
        return "Orta"
    return "Düşük"


def profile_label(alpha):
    alpha = _safe_float(alpha, 0) or 0
    if alpha >= 78:
        return "GÜÇLÜ PROFİL"
    if alpha >= 68:
        return "POZİTİF PROFİL"
    if alpha >= 58:
        return "DENGELİ PROFİL"
    return "İZLEME PROFİLİ"


def profile_class(alpha):
    alpha = _safe_float(alpha, 0) or 0
    if alpha >= 78:
        return "strong"
    if alpha >= 68:
        return "positive"
    if alpha >= 58:
        return "balanced"
    return "watch"


def score_to_grade(score):
    score = _clamp(score)
    if score >= 75:
        return "güçlü"
    if score >= 60:
        return "olumlu"
    if score >= 45:
        return "dengeli"
    return "zayıf"


def safe_analysis_summary(item, radar):
    # Yatırım eylemi / yönlendirme dili kullanmadan tamamen ölçülen bileşenleri özetler.
    ordered = sorted(radar.items(), key=lambda kv: kv[1], reverse=True)
    strongest = ordered[:2]
    weakest = ordered[-2:]
    strong_txt = " ve ".join(k for k, _ in strongest)
    weak_txt = " ve ".join(k for k, _ in weakest)
    return (
        f"Model çıktıları {strong_txt} başlıklarında görece güçlü bir profil gösterirken, "
        f"{weak_txt} başlıkları izlenmesi gereken alanlar olarak öne çıkıyor. "
        "Bu değerlendirme, yalnızca mevcut temel ve teknik veri setinin karşılaştırmalı analizidir."
    )


def _normalize_cash_quality(cfo_to_net_income):
    x = _safe_float(cfo_to_net_income)
    if x is None:
        return 50.0
    # 0 -> 10, 1 -> ~70, 1.5+ -> 100; negatif nakit dönüşümü sert cezalanır.
    if x <= 0:
        return 10.0
    return _clamp(20 + (x / 1.5) * 80)


def _normalize_roe(roe):
    x = _safe_float(roe)
    if x is None:
        return 50.0
    # Finansal kaynaktan decimal gelmesi beklenir (0.20 = %20). Yüzde gelirse normalize et.
    if abs(x) > 2:
        x = x / 100.0
    return _clamp((x / 0.30) * 100)


def _normalize_upside(price, base_fv):
    p = _safe_float(price)
    fv = _safe_float(base_fv)
    if not p or fv is None:
        return 50.0
    upside = (fv / p - 1.0) * 100.0
    # -10% veya altı -> 0, +30% ve üzeri -> 100
    return _clamp((upside + 10.0) / 40.0 * 100.0)


def compute_radar_scores(item):
    valuation = _clamp((_safe_float(item.get("score_valuation"), 0.5) or 0.5) * 100)
    momentum = _clamp((_safe_float(item.get("score_momentum"), 0.5) or 0.5) * 100)
    accel = _clamp((_safe_float(item.get("score_acceleration"), 0.5) or 0.5) * 100)
    margin = _clamp((_safe_float(item.get("score_margin"), 0.5) or 0.5) * 100)
    balance = _clamp((_safe_float(item.get("score_balance"), 0.5) or 0.5) * 100)
    divergence = _clamp((_safe_float(item.get("score_divergence"), 0.5) or 0.5) * 100)
    cash_q = _normalize_cash_quality(item.get("cfo_to_net_income"))
    roe_q = _normalize_roe(item.get("roe"))
    upside_q = _normalize_upside(item.get("price"), item.get("base_fv"))
    technical = _clamp(item.get("technical_score", 50))

    return {
        "Değerleme": round(valuation, 1),
        "Büyüme & İvme": round(momentum * 0.55 + accel * 0.45, 1),
        "Kârlılık & Nakit": round(margin * 0.45 + cash_q * 0.35 + roe_q * 0.20, 1),
        "Bilanço Sağlığı": round(balance, 1),
        "Yeniden Fiyatlama": round(divergence * 0.65 + upside_q * 0.35, 1),
        "Teknik Yapı": round(technical, 1),
    }


def radar_svg(scores, size=240):
    labels = list(scores.keys())
    values = [scores[k] for k in labels]
    cx = cy = size / 2
    radius = size * 0.31
    label_r = size * 0.39

    def pt(angle_deg, r):
        a = math.radians(angle_deg - 90)
        return cx + math.cos(a) * r, cy + math.sin(a) * r

    angles = [i * 360 / 6 for i in range(6)]
    grid = []
    for frac in (0.25, 0.5, 0.75, 1.0):
        pts = " ".join(f"{pt(a, radius*frac)[0]:.1f},{pt(a, radius*frac)[1]:.1f}" for a in angles)
        grid.append(f'<polygon points="{pts}" fill="none" stroke="#cad7e5" stroke-width="1"/>')
    axes = []
    for a in angles:
        x, y = pt(a, radius)
        axes.append(f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{x:.1f}" y2="{y:.1f}" stroke="#d4deea" stroke-width="1"/>')

    data_pts = " ".join(
        f"{pt(a, radius * (_clamp(v)/100))[0]:.1f},{pt(a, radius * (_clamp(v)/100))[1]:.1f}"
        for a, v in zip(angles, values)
    )
    label_parts = []
    for a, label, value in zip(angles, labels, values):
        x, y = pt(a, label_r)
        anchor = "middle"
        if x < cx - 12:
            anchor = "end"
        elif x > cx + 12:
            anchor = "start"
        # two-line labels where needed
        words = label.split(" ")
        if len(words) > 1:
            mid = max(1, len(words)//2)
            l1 = " ".join(words[:mid])
            l2 = " ".join(words[mid:])
            label_parts.append(
                f'<text x="{x:.1f}" y="{y-4:.1f}" text-anchor="{anchor}" class="radar-label">{l1}</text>'
                f'<text x="{x:.1f}" y="{y+9:.1f}" text-anchor="{anchor}" class="radar-label">{l2}</text>'
                f'<text x="{x:.1f}" y="{y+22:.1f}" text-anchor="{anchor}" class="radar-value">{value:.0f}</text>'
            )
        else:
            label_parts.append(
                f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" class="radar-label">{label}</text>'
                f'<text x="{x:.1f}" y="{y+14:.1f}" text-anchor="{anchor}" class="radar-value">{value:.0f}</text>'
            )

    return (
        f'<svg viewBox="0 0 {size} {size}" class="radar-svg" xmlns="http://www.w3.org/2000/svg">'
        + "".join(grid) + "".join(axes)
        + f'<polygon points="{data_pts}" fill="rgba(38,154,105,.22)" stroke="#1b8e64" stroke-width="2"/>'
        + "".join(label_parts) + '</svg>'
    )


def fetch_week_price_series(ticker):
    end = datetime.now(TR_TZ).date()
    start = end - timedelta(days=12)
    try:
        from isyatirimhisse import fetch_stock_data as isy_fetch_stock_data
        df = isy_fetch_stock_data(symbols=ticker, start_date=start.strftime("%d-%m-%Y"), end_date=end.strftime("%d-%m-%Y"))
        if df is None or df.empty:
            return []
        close_col = next((c for c in ["HGDG_KAPANIS", "KAPANIS", "CLOSING_TL", "CLOSING"] if c in df.columns), None)
        date_col = next((c for c in ["HGDG_TARIH", "TARIH"] if c in df.columns), None)
        if not close_col:
            return []
        d = df.copy()
        d[close_col] = pd.to_numeric(d[close_col], errors="coerce")
        if date_col:
            d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
            d = d.sort_values(date_col)
        vals = [float(x) for x in d[close_col].dropna().tail(6).tolist()]
        return vals
    except Exception as e:
        print(f"[GÖRSEL] {ticker} 1 haftalık fiyat serisi alınamadı: {e}")
        return []


def sparkline_svg(values, width=260, height=82):
    if not values or len(values) < 2:
        return f'<svg viewBox="0 0 {width} {height}" class="sparkline-svg"><line x1="4" y1="{height/2}" x2="{width-4}" y2="{height/2}" stroke="#aebdcd" stroke-width="2" stroke-dasharray="5 5"/></svg>'
    mn, mx = min(values), max(values)
    span = mx - mn if mx != mn else 1.0
    pad = 7
    pts = []
    for i, v in enumerate(values):
        x = pad + i * (width - 2*pad) / (len(values)-1)
        y = height - pad - ((v-mn)/span) * (height - 2*pad)
        pts.append((x, y))
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area = f"{pts[0][0]:.1f},{height-pad} " + line + f" {pts[-1][0]:.1f},{height-pad}"
    last_x, last_y = pts[-1]
    return (
        f'<svg viewBox="0 0 {width} {height}" class="sparkline-svg">'
        '<defs><linearGradient id="spg" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#20a56f" stop-opacity=".34"/><stop offset="1" stop-color="#20a56f" stop-opacity="0"/></linearGradient></defs>'
        f'<polygon points="{area}" fill="url(#spg)"/>'
        f'<polyline points="{line}" fill="none" stroke="#169766" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>'
        f'<circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="4" fill="#0f8057"/>'
        '</svg>'
    )


def _format_num(v, decimals=2):
    x = _safe_float(v)
    return "N/A" if x is None else f"{x:,.{decimals}f}"


def _upside_pct(item):
    p = _safe_float(item.get("price"))
    fv = _safe_float(item.get("base_fv"))
    if not p or fv is None:
        return None
    return (fv/p - 1) * 100


def rank_delta(item, current_rank, last_seen_map):
    prev = last_seen_map.get(item.get("ticker")) if last_seen_map else None
    if not prev or prev.get("rank") is None:
        return "YENİ", "new"
    d = int(prev.get("rank")) - current_rank
    if d > 0:
        return f"▲ +{d}", "up"
    if d < 0:
        return f"▼ {d}", "down"
    return "=", "flat"


def _public_points(items, limit=3):
    out = []
    for x in (items or [])[:limit]:
        txt = str(x).strip()
        # Eylem dilini yumuşatan küçük güvenlik katmanı.
        replacements = {
            "alım": "pozitif sinyal", "satım": "negatif sinyal", "alınabilir": "izlenebilir",
            "satılabilir": "izlenebilir", "hedef fiyat": "değerleme referansı",
        }
        low = txt.lower()
        for bad, good in replacements.items():
            if bad in low:
                txt = txt.replace(bad, good).replace(bad.capitalize(), good.capitalize())
        out.append(txt)
    return out


def build_daily_context(ranked, last_seen_map, previous_day_top10_map, total_scanned, deep_count):
    top15 = ranked[:15]
    top10 = top15[:10]
    top3 = top15[:3]
    cards = []
    for idx, item in enumerate(top3, 1):
        radar = compute_radar_scores(item)
        delta, delta_class = rank_delta(item, idx, last_seen_map)
        spark_vals = fetch_week_price_series(item.get("ticker"))
        cards.append({
            "rank": idx,
            "ticker": item.get("ticker"),
            "cap_bucket": item.get("cap_bucket") or "—",
            "alpha": round(_safe_float(item.get("alpha_score"), 0) or 0, 1),
            "confidence": confidence_label(item.get("data_confidence")),
            "price": _format_num(item.get("price")),
            "bear_fv": _format_num(item.get("bear_fv")),
            "base_fv": _format_num(item.get("base_fv")),
            "bull_fv": _format_num(item.get("bull_fv")),
            "upside": _upside_pct(item),
            "previous_rank": last_seen_map.get(item.get("ticker"), {}).get("rank", "—") if last_seen_map else "—",
            "delta": delta,
            "delta_class": delta_class,
            "profile": profile_label(item.get("alpha_score")),
            "profile_class": profile_class(item.get("alpha_score")),
            "radar_svg": radar_svg(radar),
            "sparkline_svg": sparkline_svg(spark_vals),
            "sparkline_min": min(spark_vals) if spark_vals else None,
            "sparkline_max": max(spark_vals) if spark_vals else None,
            "radar": radar,
            "analysis_summary": safe_analysis_summary(item, radar),
            "positives": _public_points(item.get("catalysts") or [], 3),
            "risks": _public_points(item.get("risks") or [], 3),
        })

    rows = []
    for i, item in enumerate(top10, 1):
        delta, delta_class = rank_delta(item, i, last_seen_map)
        rows.append({
            "rank": i,
            "ticker": item.get("ticker"),
            "alpha": round(_safe_float(item.get("alpha_score"), 0) or 0, 1),
            "price": _format_num(item.get("price")),
            "base_fv": _format_num(item.get("base_fv")),
            "delta": delta,
            "delta_class": delta_class,
        })

    reserve = []
    for i, item in enumerate(top15[10:15], 11):
        reserve.append({
            "rank": i,
            "ticker": item.get("ticker"),
            "alpha": round(_safe_float(item.get("alpha_score"), 0) or 0, 1),
            "base_fv": _format_num(item.get("base_fv")),
        })

    current = {x.get("ticker") for x in top10}
    previous = set(previous_day_top10_map.keys()) if previous_day_top10_map else set()
    new_entries = sorted(current - previous)
    dropped = sorted(previous - current)

    return {
        "date": datetime.now(TR_TZ).strftime("%d.%m.%Y"),
        "total_scanned": total_scanned,
        "deep_count": deep_count,
        "opportunity_count": len(top10),
        "cards": cards,
        "top10": rows,
        "reserve": reserve,
        "new_entries": new_entries,
        "dropped": dropped,
    }


def _env():
    return Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
    )


def render_html_to_png(template_name, context, output_path, viewport):
    env = _env()
    html = env.get_template(template_name).render(**context)
    html_path = OUTPUT_DIR / (Path(output_path).stem + ".html")
    html_path.write_text(html, encoding="utf-8")

    from playwright.sync_api import sync_playwright
    import shutil
    with sync_playwright() as p:
        launch_kwargs = {"headless": True}
        # Lokal geliştirme ortamında sistem Chromium'u kullan; GitHub Actions'ta
        # workflow'un kurduğu Playwright Chromium otomatik bulunur.
        system_chromium = shutil.which("chromium") or shutil.which("chromium-browser") or shutil.which("google-chrome")
        if system_chromium:
            launch_kwargs["executable_path"] = system_chromium
        browser = p.chromium.launch(**launch_kwargs)
        page = browser.new_page(viewport=viewport, device_scale_factor=1)
        page.set_content(html, wait_until="load")
        try:
            page.wait_for_load_state("networkidle", timeout=5000)
        except Exception:
            pass
        page.screenshot(path=str(output_path), full_page=True)
        browser.close()
    return str(output_path)


def render_daily_report(ranked, last_seen_map, previous_day_top10_map, total_scanned, deep_count):
    ctx = build_daily_context(ranked, last_seen_map, previous_day_top10_map, total_scanned, deep_count)
    out = OUTPUT_DIR / f"beiq_daily_{datetime.now(TR_TZ).strftime('%Y%m%d')}.png"
    return render_html_to_png("daily_report.html", ctx, out, {"width": 1080, "height": 1580})


def build_weekly_context(entries, agg, summary_text=None):
    dates = [e.get("date") for e in entries if e.get("date")]
    start = min(dates) if dates else "—"
    end = max(dates) if dates else "—"

    stats = []
    for ticker, records in agg.items():
        valid = [r for r in records if r.get("rank") is not None]
        if not valid:
            continue
        valid = sorted(valid, key=lambda r: r.get("date", ""))
        alphas = [_safe_float(r.get("alpha_score"), 0) or 0 for r in valid]
        ranks = [_safe_float(r.get("rank"), 99) or 99 for r in valid]
        stats.append({
            "ticker": ticker,
            "days": len(valid),
            "avg_alpha": sum(alphas)/len(alphas),
            "first_alpha": alphas[0], "last_alpha": alphas[-1],
            "alpha_change": alphas[-1]-alphas[0],
            "first_rank": ranks[0], "last_rank": ranks[-1],
            "rank_change": ranks[0]-ranks[-1],
            "best_rank": min(ranks),
            "avg_rank": sum(ranks)/len(ranks),
        })

    stable = sorted(stats, key=lambda x: (-x["days"], x["avg_rank"]))[:4]
    leader = sorted(stats, key=lambda x: (-x["avg_alpha"], x["avg_rank"]))[:1]
    starters = sorted(stats, key=lambda x: -x["first_alpha"])[:2]
    decliners = sorted(stats, key=lambda x: (x["alpha_change"], x["rank_change"]))[:2]

    highlights = sorted(stats, key=lambda x: (-x["days"], -x["avg_alpha"]))[:3]
    movements = sorted(stats, key=lambda x: -abs(x["alpha_change"]))[:4]

    event_cards = []
    if stable:
        names = ", ".join(x["ticker"] for x in stable)
        event_cards.append({"n": 1, "title": "İstikrar öne çıktı", "text": f"{names} haftalık listelerde en yüksek devamlılığı gösteren grupta yer aldı."})
    if leader:
        x = leader[0]
        event_cards.append({"n": 2, "title": "Haftalık skor liderliği", "text": f"{x['ticker']} hafta boyunca ortalama {x['avg_alpha']:.1f} Alpha Score ile üst grupta kaldı."})
    if movements:
        up = max(movements, key=lambda x: x["alpha_change"])
        event_cards.append({"n": 3, "title": "Skor hareketi izlendi", "text": f"{up['ticker']} tarafında haftalık Alpha Score değişimi {up['alpha_change']:+.1f} puan olarak kaydedildi."})
    if decliners:
        d = decliners[0]
        event_cards.append({"n": 4, "title": "Göreli zayıflama görüldü", "text": f"{d['ticker']} haftayı başlangıca göre {d['alpha_change']:+.1f} puan Alpha değişimiyle tamamladı."})
    while len(event_cards) < 4:
        event_cards.append({"n": len(event_cards)+1, "title": "Veri akışı", "text": "Hafta içindeki sıralama ve skor değişimleri karşılaştırmalı olarak izlenmeye devam etti."})

    return {
        "date_start": start,
        "date_end": end,
        "stable_names": ", ".join(x["ticker"] for x in stable) if stable else "—",
        "leader_name": leader[0]["ticker"] if leader else "—",
        "starter_names": ", ".join(x["ticker"] for x in starters) if starters else "—",
        "decliner_names": ", ".join(x["ticker"] for x in decliners) if decliners else "—",
        "event_cards": event_cards[:4],
        "highlights": highlights,
        "movements": movements,
        "summary_text": summary_text or "Haftalık görünüm, yalnızca sistemin ürettiği sıralama, skor ve veri kalitesi değişimleri üzerinden değerlendirilmiştir.",
    }


def render_weekly_report(entries, agg, summary_text=None):
    ctx = build_weekly_context(entries, agg, summary_text)
    out = OUTPUT_DIR / f"beiq_weekly_{datetime.now(TR_TZ).strftime('%Y%m%d')}.png"
    return render_html_to_png("weekly_report.html", ctx, out, {"width": 1080, "height": 1380})
