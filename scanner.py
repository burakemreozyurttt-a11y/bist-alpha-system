"""
BIST FUNDAMENTAL ALPHA SYSTEM — Faz 1 (MVP) — v2
=============================================
Her hafta içi günü çalışır (GitHub Actions cron ile tetiklenir):
  1) KAP'tan tüm BIST hisse kodlarını çeker (olmazsa yedek listeyi kullanır)
  2) Round 1: İş Yatırım'dan fiyat/hacim verisiyle LİKİDİTE taraması yapar
     -> en likit ~20-25 hisse Round 2'ye aday gösterilir
  3) Round 2: sadece bu adaylar için yfinance'tan fundamental oranlar (best
     effort — bulunamazsa N/A) çekilir ve Gemini'ye gönderilip Alpha Score /
     Fair Value hesaplatılır
  4) TOP 10 + Reserve (11-15) tablosunu oluşturur, önceki günle kıyaslar
  5) Raporu Telegram'a yollar
  6) state.json dosyasını günceller (bir sonraki çalıştırmanın "hafızası")

v2 NOTU: İlk sürümde tüm evren (720 hisse) için yfinance kullanılıyordu, ama
Yahoo Finance GitHub Actions'ın paylaşımlı IP'lerini tamamen engelledi (boş
yanıt / 429). Bu yüzden Round 1 artık İş Yatırım'ın (isyatirimhisse
kütüphanesi) fiyat verisini kullanıyor, yfinance sadece Round 2'nin az
sayıdaki (~20) finalistine, best-effort olarak uygulanıyor.

Notlar / bilinen sınırlamalar (MVP aşaması):
  - Round 1'de henüz gerçek fundamental oranlar (F/K, ROE vb.) YOK, sadece
    likidite bazlı bir ön eleme var. Sektöre özel modeller ve tam fundamental
    Round 1 filtresi Faz 1.1 / Faz 3'te İş Yatırım'ın fetch_financials
    fonksiyonuyla eklenecek.
  - yfinance eksik/hatalı veri dönebilir; eksik veri N/A olarak işaretlenir,
    uydurulmaz.
  - Multi-Agent (Bull/Bear/CRO) katmanı henüz yok — Faz 3'te eklenecek.
"""

import io
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from isyatirimhisse import fetch_stock_data as isy_fetch_stock_data
from isyatirimhisse import fetch_financials as isy_fetch_financials
from google import genai
from google.genai import types

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

MODEL_NAME = "gemini-3.6-flash"
STATE_FILE = os.path.join(os.path.dirname(__file__), "state.json")
FALLBACK_TICKERS_FILE = os.path.join(os.path.dirname(__file__), "bist_tickers_fallback.csv")

MAX_DEEP_CANDIDATES = 22          # Round 2'ye (Gemini'ye) gidecek maksimum aday sayısı
MIN_AVG_DAILY_TURNOVER_TRY = 3_000_000   # yaklaşık günlük TL hacim eşiği (likidite filtresi)
SLEEP_BETWEEN_GEMINI_CALLS = 7    # saniye — ücretsiz tier RPM limitine takılmamak için

# İş Yatırım'ın fetch_financials fonksiyonu şirkete göre farklı raporlama
# formatı kullanabiliyor (THYAO testinde group='1' / XI_29 çalıştı, group='2'
# / UFRS boş döndü). Bu yüzden sırayla ikisini de deniyoruz.
FIN_GROUPS_TO_TRY = ["1", "2"]
FIN_REQUEST_DELAY = 1.2           # finansal veri istekleri arası bekleme (saniye)

# Bilanço/gelir tablosu kalem kodları (THYAO üzerinde doğrulandı, standart
# İş Yatırım XI_29 kod şeması olduğu için diğer şirketlerde de aynı olmalı)
ITEM_REVENUE = "3C"                # Satış Gelirleri
ITEM_NET_INCOME_PARENT = "3Z"      # Ana Ortaklık Payları (net kâr)
ITEM_EPS = "3ZD"                   # Hisse Başına Kazanç
ITEM_TOTAL_ASSETS = "1BL"          # TOPLAM VARLIKLAR
ITEM_PARENT_EQUITY = "2O"          # Ana Ortaklığa Ait Özkaynaklar
ITEM_SHORT_TERM_LIAB = "2A"        # Kısa Vadeli Yükümlülükler
ITEM_LONG_TERM_LIAB = "2B"         # Uzun Vadeli Yükümlülükler

TR_TZ = timezone(timedelta(hours=3))

client = genai.Client(api_key=GEMINI_API_KEY)


# --------------------------------------------------------------------------
# 1) TICKER LİSTESİ
# --------------------------------------------------------------------------
def get_bist_tickers():
    """KAP'ın herkese açık excel endpoint'inden BIST hisse kodlarını çekmeyi
    dener. Başarısız olursa (site yapısı değiştiyse vb.) yerel yedek listeyi
    kullanır."""
    url = "https://www.kap.org.tr/tr/api/company/generic/excel/IGS/A"
    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        df = pd.read_excel(io.BytesIO(resp.content), header=None)
        pattern = re.compile(r"^[A-Z]{4,6}$")
        found = set()
        for col in df.columns:
            for val in df[col].dropna().astype(str):
                val = val.strip()
                if pattern.match(val):
                    found.add(val)
        if len(found) >= 200:
            print(f"KAP'tan {len(found)} hisse kodu çekildi.")
            return sorted(found)
        print(f"KAP'tan sadece {len(found)} kod bulundu, yedek listeye geçiliyor.")
    except Exception as e:
        print(f"KAP listesi çekilemedi ({e}), yedek listeye geçiliyor.")

    df = pd.read_csv(FALLBACK_TICKERS_FILE)
    tickers = sorted(df["ticker"].dropna().astype(str).str.strip().unique().tolist())
    print(f"Yedek listeden {len(tickers)} hisse kodu kullanılıyor.")
    return tickers


# --------------------------------------------------------------------------
# 2) ROUND 1 — LİKİDİTE TARAMASI (İş Yatırım verisiyle, LLM'siz)
# --------------------------------------------------------------------------
IY_MIN_DELAY = 0.5                # istekler arası minimum bekleme (saniye)
IY_MAX_DELAY = 1.2                # istekler arası maksimum bekleme (saniye)
IY_MAX_RETRIES = 2
IY_LOOKBACK_DAYS = 20             # kaç günlük fiyat/hacim geçmişine bakılacak

_debug_columns_printed = False    # ilk başarılı çekimde sütun isimlerini bir kez loglamak için


def _pick_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def fetch_price_snapshot(ticker):
    """İş Yatırım'dan son ~20 günlük fiyat/hacim verisini çeker, güncel fiyatı
    ve ortalama günlük TL hacmini (likidite göstergesi) hesaplar."""
    global _debug_columns_printed

    end_date = datetime.now(TR_TZ).strftime("%d-%m-%Y")
    start_date = (datetime.now(TR_TZ) - timedelta(days=IY_LOOKBACK_DAYS * 2)).strftime("%d-%m-%Y")

    for attempt in range(1, IY_MAX_RETRIES + 1):
        try:
            df = isy_fetch_stock_data(symbols=ticker, start_date=start_date, end_date=end_date)
            if df is None or df.empty:
                raise ValueError("Boş veri döndü")

            if not _debug_columns_printed:
                print(f"[DEBUG] İş Yatırım'dan dönen örnek sütunlar ({ticker}): {df.columns.tolist()}")
                _debug_columns_printed = True

            close_col = _pick_column(df, ["HGDG_KAPANIS", "KAPANIS", "CLOSING_TL", "CLOSING"])
            vol_try_col = _pick_column(df, ["HGDG_HACIM_TL", "HACIM_TL", "VOLUME_TL"])
            vol_lot_col = _pick_column(df, ["HGDG_HACIM_LOT", "HACIM_LOT", "VOLUME_LOT", "HGDG_HACIM"])

            if close_col is None:
                raise ValueError(f"Kapanış fiyatı sütunu bulunamadı. Mevcut sütunlar: {df.columns.tolist()}")

            last_price = float(df[close_col].iloc[-1])

            if vol_try_col is not None:
                avg_turnover = float(df[vol_try_col].tail(IY_LOOKBACK_DAYS).mean())
            elif vol_lot_col is not None:
                avg_turnover = float((df[close_col] * df[vol_lot_col]).tail(IY_LOOKBACK_DAYS).mean())
            else:
                avg_turnover = None  # likidite hesaplanamıyor, filtrede elenecek

            return {
                "ticker": ticker,
                "price": last_price,
                "avg_daily_turnover_try": avg_turnover,
            }
        except Exception as e:
            if attempt < IY_MAX_RETRIES:
                time.sleep(3)
            else:
                print(f"  {ticker}: fiyat verisi alınamadı ({e})")
                return None
    return None


def round1_screen(tickers):
    """Tüm evreni İş Yatırım'dan seri şekilde çeker, likidite filtresi
    uygular ve en likit MAX_DEEP_CANDIDATES hisseyi Round 2'ye aday gösterir.
    (Not: F/K, ROE gibi fundamental oranlar burada YOK — onlar sadece
    Round 2'deki finalistler için yfinance'tan çekiliyor, bkz. round2_deep_analysis.)"""
    rows = []
    total = len(tickers)
    consecutive_failures = 0

    for i, tk in enumerate(tickers, 1):
        res = fetch_price_snapshot(tk)
        if res:
            rows.append(res)
            consecutive_failures = 0
        else:
            consecutive_failures += 1

        if i % 25 == 0 or i == total:
            print(f"  ...{i}/{total} hisse tarandı (şu ana kadar başarılı: {len(rows)})")

        if consecutive_failures >= 40:
            print("Üst üste çok fazla başarısız istek, İş Yatırım muhtemelen bu IP'yi geçici engelledi. Taramayı erken durduruyorum.")
            break

        time.sleep(random.uniform(IY_MIN_DELAY, IY_MAX_DELAY))

    df = pd.DataFrame(rows)
    print(f"Veri çekilebilen hisse sayısı: {len(df)} / {total}")

    if df.empty:
        print("Hiçbir hisse için veri çekilemedi.")
        return df

    # Likidite filtresi
    df = df[df["avg_daily_turnover_try"].fillna(0) >= MIN_AVG_DAILY_TURNOVER_TRY].copy()
    print(f"Likidite filtresinden geçen hisse sayısı: {len(df)}")

    if df.empty:
        return df

    # Fundamental veri henüz yok, en likit hisseleri aday gösteriyoruz
    # (likit hisseler genelde daha iyi kapsanan, daha güvenilir veriye sahip
    # orta/büyük ölçekli şirketlerdir).
    df = df.sort_values("avg_daily_turnover_try", ascending=False)
    return df.head(MAX_DEEP_CANDIDATES)


# --------------------------------------------------------------------------
# 3) ROUND 2 — SADECE FİNALİSTLER İÇİN FUNDAMENTAL VERİ + GEMINI ANALİZİ
# --------------------------------------------------------------------------
_QUARTER_COL_PATTERN = re.compile(r"^(\d{4})/(\d{1,2})$")


def _quarter_columns_sorted(df):
    """DataFrame'deki '2025/9' gibi çeyrek sütunlarını kronolojik sıraya dizer."""
    cols = []
    for c in df.columns:
        m = _QUARTER_COL_PATTERN.match(str(c))
        if m:
            cols.append((int(m.group(1)), int(m.group(2)), c))
    cols.sort()  # yıl, sonra ay bazlı artan sıralama
    return cols


def _get_item_row(df, item_code):
    match = df[df["FINANCIAL_ITEM_CODE"] == item_code]
    if match.empty:
        return None
    return match.iloc[0]


def _latest_and_prior_year(df, item_code, quarter_cols):
    """Bir kalemin en güncel (dolu) değerini ve bir önceki yılın aynı
    dönemindeki değerini döndürür. Ayrıca dönemin kaç aylık olduğunu (3/6/9/12)
    da döndürür (yıllıklandırma için)."""
    row = _get_item_row(df, item_code)
    if row is None:
        return None, None, None

    latest_val, latest_year, latest_period = None, None, None
    for year, period, col in reversed(quarter_cols):  # en güncelden geçmişe
        val = row.get(col)
        if pd.notna(val):
            try:
                latest_val = float(val)
                latest_year, latest_period = year, period
                break
            except (TypeError, ValueError):
                continue

    if latest_val is None:
        return None, None, None

    prior_col = f"{latest_year - 1}/{latest_period}"
    prior_val = None
    if prior_col in df.columns:
        raw = row.get(prior_col)
        if pd.notna(raw):
            try:
                prior_val = float(raw)
            except (TypeError, ValueError):
                prior_val = None

    return latest_val, prior_val, latest_period


def _annualize(value, period_months):
    if value is None or not period_months:
        return None
    return value * (12 / period_months)


def _safe_div(a, b):
    if a is None or b is None or b == 0:
        return None
    return a / b


def fetch_fundamentals_isyatirim(ticker):
    """İş Yatırım'ın bilanço/gelir tablosu verisinden F/K, ROE, PD/DD, borç/
    özsermaye ve büyüme oranlarını KENDİMİZ hesaplıyoruz. Veri gelmezse (ya
    da bir kalem eksikse) o alan N/A olarak işaretlenir, uydurulmaz."""
    fields = {
        "name": ticker, "sector": "N/A", "industry": "N/A", "market_cap": "N/A",
        "trailing_pe": "N/A", "forward_pe": "N/A", "price_to_book": "N/A",
        "roe": "N/A", "revenue_growth": "N/A", "earnings_growth": "N/A",
        "debt_to_equity": "N/A", "dividend_yield": "N/A",
    }

    this_year = datetime.now(TR_TZ).year
    df = None
    for group in FIN_GROUPS_TO_TRY:
        try:
            candidate_df = isy_fetch_financials(
                symbols=ticker, start_year=this_year - 2, end_year=this_year,
                financial_group=group,
            )
            if candidate_df is not None and not candidate_df.empty:
                df = candidate_df
                break
        except Exception:
            pass
        time.sleep(FIN_REQUEST_DELAY)

    if df is None:
        print(f"  {ticker}: bilanço verisi hiçbir formatta bulunamadı, N/A ile devam")
        return fields

    quarter_cols = _quarter_columns_sorted(df)
    if not quarter_cols:
        return fields

    try:
        revenue_latest, revenue_prior, period = _latest_and_prior_year(df, ITEM_REVENUE, quarter_cols)
        net_income_latest, net_income_prior, _ = _latest_and_prior_year(df, ITEM_NET_INCOME_PARENT, quarter_cols)
        eps_latest, _, _ = _latest_and_prior_year(df, ITEM_EPS, quarter_cols)
        equity_latest, _, _ = _latest_and_prior_year(df, ITEM_PARENT_EQUITY, quarter_cols)
        st_liab_latest, _, _ = _latest_and_prior_year(df, ITEM_SHORT_TERM_LIAB, quarter_cols)
        lt_liab_latest, _, _ = _latest_and_prior_year(df, ITEM_LONG_TERM_LIAB, quarter_cols)

        # Büyüme oranları: aynı dönemin bir önceki yıla göre değişimi (YoY)
        if revenue_latest is not None and revenue_prior:
            fields["revenue_growth"] = round((revenue_latest / revenue_prior - 1), 4)
        if net_income_latest is not None and net_income_prior:
            fields["earnings_growth"] = round((net_income_latest / net_income_prior - 1), 4)

        # Yıllıklandırılmış (TTM'e yakın) net kâr ve EPS
        net_income_annualized = _annualize(net_income_latest, period)
        eps_annualized = _annualize(eps_latest, period)

        # ROE = yıllıklandırılmış net kâr / son özkaynak
        roe = _safe_div(net_income_annualized, equity_latest)
        if roe is not None:
            fields["roe"] = round(roe, 4)

        # Borç / Özkaynak
        if st_liab_latest is not None and lt_liab_latest is not None:
            dte = _safe_div(st_liab_latest + lt_liab_latest, equity_latest)
            if dte is not None:
                fields["debt_to_equity"] = round(dte, 2)

        # F/K = fiyat / yıllıklandırılmış hisse başı kazanç
        # (fields["_price"] Round 2'de candidate içine ayrıca eklenecek)
        fields["_eps_annualized"] = eps_annualized
        fields["_net_income_annualized"] = net_income_annualized
        fields["_equity_latest"] = equity_latest

    except Exception as e:
        print(f"  {ticker}: oran hesaplanırken hata ({e}), mevcut alanlarla devam")

    return fields
ANALYSIS_PROMPT_TEMPLATE = """
Sen kurumsal düzeyde bir BIST temel analiz uzmanısın. Aşağıdaki şirket için
SADECE verilen sayısal verilere dayanarak bir değerlendirme yap. Bilmediğin
veya verilmeyen bilgiyi UYDURMA, "N/A" yaz.

ŞİRKET: {name} ({ticker})
Sektör: {sector} / {industry}
Güncel Fiyat: {price} TL
Piyasa Değeri: {market_cap}
Trailing F/K: {trailing_pe}
Forward F/K: {forward_pe}
PD/DD: {price_to_book}
ROE: {roe}
Ciro Büyümesi (YoY): {revenue_growth}
Kâr Büyümesi (YoY): {earnings_growth}
Borç/Özsermaye: {debt_to_equity}
Temettü Verimi: {dividend_yield}

Yalnızca aşağıdaki JSON formatında, başka hiçbir açıklama eklemeden yanıt ver:

{{
  "alpha_score": <0-100 arası tam sayı, ne kadar cazip bir fırsat>,
  "data_confidence": <0-100 arası tam sayı, verinin güvenilirliği/tamlığı>,
  "bear_fv": <TL cinsinden kötümser adil değer, sayı>,
  "base_fv": <TL cinsinden ana senaryo adil değer, sayı>,
  "bull_fv": <TL cinsinden iyimser adil değer, sayı>,
  "thesis_summary": "<2-3 cümlelik Türkçe yatırım tezi özeti>",
  "catalysts": ["<olası katalizör 1>", "<olası katalizör 2>"],
  "risks": ["<ana risk 1>", "<ana risk 2>"],
  "verdict": "<HIGH CONVICTION | ATTRACTIVE | WATCH | WEAKENING içinden biri>"
}}
"""


def analyze_with_gemini(candidate):
    prompt = ANALYSIS_PROMPT_TEMPLATE.format(**candidate)
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.3,
            ),
        )
        data = json.loads(response.text)
        data["ticker"] = candidate["ticker"]
        data["name"] = candidate["name"]
        data["price"] = candidate["price"]
        return data
    except Exception as e:
        print(f"  Gemini hatası ({candidate['ticker']}): {e}")
        return None


def _finalize_price_based_ratios(fundamentals, price):
    """fetch_fundamentals_isyatirim'in bıraktığı ara değerlerden (EPS, net
    kâr, özkaynak) fiyata bağlı oranları (F/K, PD/DD, piyasa değeri) tamamlar."""
    eps_ann = fundamentals.pop("_eps_annualized", None)
    net_income_ann = fundamentals.pop("_net_income_annualized", None)
    equity_latest = fundamentals.pop("_equity_latest", None)

    shares_outstanding = None
    if eps_ann and net_income_ann and abs(eps_ann) > 1e-9:
        shares_outstanding = net_income_ann / eps_ann

    if eps_ann and eps_ann > 0:
        fundamentals["trailing_pe"] = round(price / eps_ann, 2)

    if shares_outstanding and shares_outstanding > 0:
        market_cap = price * shares_outstanding
        fundamentals["market_cap"] = round(market_cap, 0)
        if equity_latest and equity_latest > 0:
            book_value_per_share = equity_latest / shares_outstanding
            if book_value_per_share > 0:
                fundamentals["price_to_book"] = round(price / book_value_per_share, 2)

    return fundamentals


def round2_deep_analysis(candidates_df):
    results = []
    records = candidates_df.to_dict("records")
    for i, row in enumerate(records, 1):
        ticker = row["ticker"]
        price = row["price"]
        print(f"  Derin analiz {i}/{len(records)}: {ticker}")
        fundamentals = fetch_fundamentals_isyatirim(ticker)
        fundamentals = _finalize_price_based_ratios(fundamentals, price)
        candidate = {**fundamentals, "ticker": ticker, "price": price}
        result = analyze_with_gemini(candidate)
        if result:
            results.append(result)
        time.sleep(SLEEP_BETWEEN_GEMINI_CALLS)
    return results


# --------------------------------------------------------------------------
# 4) STATE (HAFIZA) YÖNETİMİ
# --------------------------------------------------------------------------
def load_previous_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"date": None, "ranking": []}


def save_state(ranking):
    state = {
        "date": datetime.now(TR_TZ).strftime("%Y-%m-%d"),
        "ranking": ranking,
    }
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------
# 5) RAPOR OLUŞTURMA
# --------------------------------------------------------------------------
def fmt(v, suffix=""):
    if v is None or v == "N/A":
        return "N/A"
    try:
        return f"{float(v):,.2f}{suffix}"
    except (TypeError, ValueError):
        return str(v)


def build_report(ranked, previous_ranking, total_scanned, deep_count):
    today = datetime.now(TR_TZ).strftime("%d.%m.%Y")
    prev_map = {r["ticker"]: r for r in previous_ranking}

    lines = []
    lines.append("📊 BIST FUNDAMENTAL ALPHA — GÜNLÜK TARAMA")
    lines.append(f"Tarih: {today}")
    lines.append(f"Taranan hisse sayısı: {total_scanned}")
    lines.append(f"Derin analiz yapılan aday sayısı: {deep_count}")
    lines.append("")

    top10 = ranked[:10]
    reserve = ranked[10:15]

    if len(top10) < 10:
        lines.append(f"⚠️ Bugün yalnızca {len(top10)} şirket kalite eşiğini geçti.")
        lines.append("")

    medals = ["🥇", "🥈", "🥉"]
    for i, item in enumerate(top10[:3]):
        medal = medals[i] if i < 3 else ""
        prev = prev_map.get(item["ticker"])
        prev_rank_str = f"#{prev['rank']}" if prev else "Yeni"
        lines.append(f"{medal} {i+1}. {item['ticker']} — {item.get('name','')}")
        lines.append(f"Alpha Score: {item.get('alpha_score','N/A')}/100  |  Data Confidence: {item.get('data_confidence','N/A')}/100")
        lines.append(f"Fiyat: {fmt(item.get('price'))} TL")
        lines.append(f"Bear FV: {fmt(item.get('bear_fv'))} | Base FV: {fmt(item.get('base_fv'))} | Bull FV: {fmt(item.get('bull_fv'))}")
        try:
            upside = (float(item.get("base_fv")) / float(item.get("price")) - 1) * 100
            lines.append(f"Base Upside: %{upside:,.1f}")
        except Exception:
            lines.append("Base Upside: N/A")
        lines.append(f"Önceki Sıra: {prev_rank_str}")
        lines.append(f"Tez: {item.get('thesis_summary','N/A')}")
        cats = item.get("catalysts") or []
        if cats:
            lines.append("Katalizörler: " + "; ".join(cats))
        risks = item.get("risks") or []
        if risks:
            lines.append("Riskler: " + "; ".join(risks))
        lines.append(f"Verdict: {item.get('verdict','N/A')}")
        lines.append("")

    lines.append("📋 GÜNÜN TOP 10 FUNDAMENTAL FIRSATI")
    for i, item in enumerate(top10, 1):
        prev = prev_map.get(item["ticker"])
        delta = ""
        if prev:
            d = prev["rank"] - i
            delta = f"(+{d})" if d > 0 else (f"({d})" if d < 0 else "(=)")
        else:
            delta = "(YENİ)"
        lines.append(
            f"{i}. {item['ticker']} | Alpha {item.get('alpha_score','N/A')} | "
            f"Fiyat {fmt(item.get('price'))} | Base FV {fmt(item.get('base_fv'))} | {delta}"
        )
    lines.append("")

    # Değişimler
    lines.append("🔄 BUGÜN NE DEĞİŞTİ?")
    current_tickers = {item["ticker"] for item in top10}
    prev_tickers = set(prev_map.keys())
    new_entries = current_tickers - prev_tickers
    dropped = prev_tickers - current_tickers
    if new_entries:
        lines.append("Yeni Girenler: " + ", ".join(sorted(new_entries)))
    if dropped:
        lines.append("Listeden Çıkanlar: " + ", ".join(sorted(dropped)))
    if not new_entries and not dropped:
        lines.append("Listede önemli bir değişiklik yok.")
    lines.append("")

    if reserve:
        lines.append("🧾 RESERVE (11-15)")
        for i, item in enumerate(reserve, 11):
            lines.append(f"{i}. {item['ticker']} | Alpha {item.get('alpha_score','N/A')} | Base FV {fmt(item.get('base_fv'))}")
        lines.append("")

    lines.append("Not: Bu bir teknik analiz değildir. Teknik yapı (entry/stop/pozisyon) kullanıcı tarafından değerlendirilmelidir.")
    lines.append("Bu içerik yatırım tavsiyesi değildir.")

    return "\n".join(lines)


# --------------------------------------------------------------------------
# 6) TELEGRAM GÖNDERİMİ
# --------------------------------------------------------------------------
def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    max_len = 3900
    chunks = [text[i:i + max_len] for i in range(0, len(text), max_len)]
    for chunk in chunks:
        r = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": chunk})
        if not r.ok:
            print("Telegram gönderim hatası:", r.text)
        time.sleep(1)


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main():
    now = datetime.now(TR_TZ)
    is_manual_run = os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch"
    if now.weekday() >= 5 and not is_manual_run:  # 5=Cumartesi, 6=Pazar
        print("Hafta sonu, tarama yapılmıyor.")
        return

    print("BIST hisse listesi alınıyor...")
    tickers = get_bist_tickers()

    print("Round 1: sayısal ön eleme başlıyor...")
    candidates_df = round1_screen(tickers)
    if candidates_df.empty:
        send_telegram_message("⚠️ Bugün likidite/veri filtresinden geçen hisse bulunamadı.")
        return

    print(f"Round 2: {len(candidates_df)} aday için Gemini analizi başlıyor...")
    analyzed = round2_deep_analysis(candidates_df)
    if not analyzed:
        send_telegram_message("⚠️ Gemini analizinden sonuç alınamadı, lütfen logları kontrol et.")
        return

    ranked = sorted(analyzed, key=lambda x: x.get("alpha_score", 0), reverse=True)
    for i, item in enumerate(ranked, 1):
        item["rank"] = i

    prev_state = load_previous_state()

    report = build_report(
        ranked,
        prev_state.get("ranking", []),
        total_scanned=len(tickers),
        deep_count=len(candidates_df),
    )
    print(report)
    send_telegram_message(report)

    save_state([{"ticker": r["ticker"], "rank": r["rank"], "alpha_score": r.get("alpha_score")} for r in ranked[:15]])
    print("Tamamlandı.")


if __name__ == "__main__":
    main()
