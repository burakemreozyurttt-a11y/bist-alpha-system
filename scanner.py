"""
BIST FUNDAMENTAL ALPHA SYSTEM — Faz 1 (MVP)
=============================================
Her hafta içi günü çalışır (GitHub Actions cron ile tetiklenir):
  1) KAP'tan tüm BIST hisse kodlarını çeker (olmazsa yedek listeyi kullanır)
  2) Round 1: yfinance ile sayısal (LLM'siz) ön eleme yapar -> ~20-25 aday
  3) Round 2: her adayı Gemini'ye gönderip Alpha Score / Fair Value hesaplatır
  4) TOP 10 + Reserve (11-15) tablosunu oluşturur, önceki günle kıyaslar
  5) Raporu Telegram'a yollar
  6) state.json dosyasını günceller (bir sonraki çalıştırmanın "hafızası")

Notlar / bilinen sınırlamalar (MVP aşaması):
  - Round 1 filtresi basit bir sayısal skor kullanır (sektöre özel modeller
    Faz 3+ içinde eklenecek — attığın orijinal sistemdeki "Sector Router"
    henüz burada yok).
  - yfinance'ın BIST şirketleri için bazı alanları (özellikle ROE, borç
    oranları) eksik/hatalı dönebilir. Eksik veri N/A olarak işaretlenir,
    uydurulmaz.
  - Multi-Agent (Bull/Bear/CRO) katmanı henüz yok — Faz 3'te eklenecek.
"""

import io
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import yfinance as yf
from google import genai
from google.genai import types

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

MODEL_NAME = "gemini-2.5-flash"
STATE_FILE = os.path.join(os.path.dirname(__file__), "state.json")
FALLBACK_TICKERS_FILE = os.path.join(os.path.dirname(__file__), "bist_tickers_fallback.csv")

MAX_DEEP_CANDIDATES = 22          # Round 2'ye (Gemini'ye) gidecek maksimum aday sayısı
MIN_AVG_DAILY_TURNOVER_TRY = 3_000_000   # yaklaşık günlük TL hacim eşiği (likidite filtresi)
SLEEP_BETWEEN_GEMINI_CALLS = 7    # saniye — ücretsiz tier RPM limitine takılmamak için
YF_WORKERS = 6                    # yfinance için paralel iş sayısı

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
# 2) ROUND 1 — SAYISAL (LLM'SİZ) ÖN ELEME
# --------------------------------------------------------------------------
def fetch_yf_snapshot(ticker):
    """Tek bir hisse için yfinance'tan temel verileri çeker."""
    yf_ticker = f"{ticker}.IS"
    try:
        t = yf.Ticker(yf_ticker)
        info = t.info or {}
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        avg_volume = info.get("averageVolume") or info.get("averageDailyVolume10Day")
        if not price or not avg_volume:
            return None
        return {
            "ticker": ticker,
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "price": price,
            "market_cap": info.get("marketCap"),
            "avg_daily_turnover_try": price * avg_volume,
            "trailing_pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "price_to_book": info.get("priceToBook"),
            "roe": info.get("returnOnEquity"),
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            "debt_to_equity": info.get("debtToEquity"),
            "dividend_yield": info.get("dividendYield"),
        }
    except Exception:
        return None


def round1_screen(tickers):
    """Tüm evreni paralel şekilde çeker, likidite filtresi uygular ve basit
    bir sayısal kompozit skorla en iyi MAX_DEEP_CANDIDATES adayı seçer."""
    rows = []
    with ThreadPoolExecutor(max_workers=YF_WORKERS) as pool:
        futures = {pool.submit(fetch_yf_snapshot, tk): tk for tk in tickers}
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            if res:
                rows.append(res)
            if i % 50 == 0:
                print(f"  ...{i}/{len(tickers)} hisse tarandı")

    df = pd.DataFrame(rows)
    print(f"Veri çekilebilen hisse sayısı: {len(df)} / {len(tickers)}")

    # Likidite filtresi
    df = df[df["avg_daily_turnover_try"].fillna(0) >= MIN_AVG_DAILY_TURNOVER_TRY].copy()
    print(f"Likidite filtresinden geçen hisse sayısı: {len(df)}")

    if df.empty:
        return df

    # Basit kompozit skor (yüzdelik dilim bazlı, eksik veri nötr kabul edilir)
    def pct_rank(series, ascending=True):
        return series.rank(pct=True, ascending=ascending, na_option="keep").fillna(0.5)

    # Değerleme: düşük F/K ve düşük PD/DD daha iyi -> ters çevirip rank
    df["score_pe"] = pct_rank(df["trailing_pe"].where(df["trailing_pe"] > 0), ascending=False)
    df["score_pb"] = pct_rank(df["price_to_book"].where(df["price_to_book"] > 0), ascending=False)
    df["score_roe"] = pct_rank(df["roe"], ascending=True)
    df["score_rev_growth"] = pct_rank(df["revenue_growth"], ascending=True)
    df["score_earn_growth"] = pct_rank(df["earnings_growth"], ascending=True)
    df["score_leverage"] = pct_rank(df["debt_to_equity"], ascending=False)

    df["composite_score"] = (
        df["score_pe"] * 0.20
        + df["score_pb"] * 0.15
        + df["score_roe"] * 0.20
        + df["score_rev_growth"] * 0.20
        + df["score_earn_growth"] * 0.20
        + df["score_leverage"] * 0.05
    )

    df = df.sort_values("composite_score", ascending=False)
    return df.head(MAX_DEEP_CANDIDATES)


# --------------------------------------------------------------------------
# 3) ROUND 2 — GEMINI İLE DERİN ANALİZ
# --------------------------------------------------------------------------
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


def round2_deep_analysis(candidates_df):
    results = []
    for i, row in enumerate(candidates_df.to_dict("records"), 1):
        print(f"  Derin analiz {i}/{len(candidates_df)}: {row['ticker']}")
        result = analyze_with_gemini(row)
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
