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

# isyatirimhisse kütüphanesi requests.get(url, timeout=10, ...) çağrısını kod
# içine sabit yazmış, dışarıdan değiştirilemiyor. İş Yatırım'ın sunucusu bazen
# 10 saniyeden yavaş yanıt verdiği için (blok değil, gerçek yavaşlık da
# olabilir) bu süreyi global olarak uzatıyoruz. Artık zaman kısıtımız
# olmadığı için (saatlerce sürse bile sorun değil) sabırlı olabiliriz.
_original_requests_get = requests.get


def _patched_requests_get(*args, **kwargs):
    if kwargs.get("timeout") == 10:
        kwargs["timeout"] = 30
    return _original_requests_get(*args, **kwargs)


requests.get = _patched_requests_get

from isyatirimhisse import fetch_stock_data as isy_fetch_stock_data
from isyatirimhisse import fetch_financials as isy_fetch_financials
from google import genai
from google.genai import types

# Faz 5 (RAG): faaliyet raporu PDF'lerini KAP'tan çekmek için
try:
    import pykap
    from pypdf import PdfReader
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

# Tek bir modele bağımlı kalmıyoruz: gemini-3.6-flash'ın ücretsiz kotası
# sadece 20 istek/gün çıktı (çok düşük), gemini-2.5-flash-lite ise Google'ın
# resmi dokümantasyonuna göre 1000 istek/gün sunuyor. Bir model kotası
# dolarsa ya da erişilemez olursa (404/429) otomatik olarak sıradakine
# geçiyoruz — her modelin kotası ayrı olduğu için bu bize çok daha büyük
# birleşik bir günlük bütçe kazandırıyor.
# Google'ın ücretsiz kotaları şu sıralar sık ve habersiz değişiyor (bazı
# kullanıcı raporlarına göre lite modeller bile bir anda 20 istek/güne
# düşürülebiliyor). Bu yüzden birden fazla modeli sırayla deniyoruz ki bir
# tanesi kısıtlanırsa/kaldırılırsa sistem tamamen durmasın.
MODEL_CANDIDATES = [
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-3.6-flash",
]
_model_start_idx = 0   # bir model bu çalıştırmada arızalı/tükenmiş bulunursa ileri kaydırılır
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
ITEM_GROSS_PROFIT = "3D"           # BRÜT KAR (ZARAR)
ITEM_OPERATING_PROFIT = "3DF"      # FAALİYET KARI (ZARARI)
ITEM_NET_INCOME_PARENT = "3Z"      # Ana Ortaklık Payları (net kâr)
ITEM_EPS = "3ZD"                   # Hisse Başına Kazanç
ITEM_TOTAL_ASSETS = "1BL"          # TOPLAM VARLIKLAR
ITEM_PARENT_EQUITY = "2O"          # Ana Ortaklığa Ait Özkaynaklar
ITEM_SHORT_TERM_LIAB = "2A"        # Kısa Vadeli Yükümlülükler
ITEM_LONG_TERM_LIAB = "2B"         # Uzun Vadeli Yükümlülükler
ITEM_CASH = "1AA"                  # Nakit ve Nakit Benzerleri
ITEM_ST_FIN_DEBT = "2AA"           # Kısa Vadeli Finansal Borçlar
ITEM_LT_FIN_DEBT = "2BA"           # Uzun Vadeli Finansal Borçlar
ITEM_CFO = "4C"                    # İşletme Faaliyetlerinden Kaynaklanan Net Nakit
ITEM_DEPRECIATION = "4B"           # Amortisman Giderleri

# --- Bilanço önbelleği ------------------------------------------------------
# Bilançolar sadece çeyrekte bir değişir, her gün 720 şirketin bilançosunu
# yeniden çekmek anlamsız (ve saatler sürer). Bu yüzden diskte önbellekliyoruz
# ve sadece CACHE_MAX_AGE_DAYS'ten eski kayıtları yeniliyoruz.
FIN_CACHE_FILE = os.path.join(os.path.dirname(__file__), "financials_cache.json")
CACHE_MAX_AGE_DAYS = 7
CACHE_SAVE_EVERY = 25              # her N şirkette bir önbelleği diske yaz (kesinti olursa ilerleme kaybolmasın)

# --- Faz 5: RAG (Faaliyet Raporu Analizi) ----------------------------------
# Faaliyet raporları çeyrekte bir yayınlanır, bilanço gibi bunu da önbelleğe
# alıyoruz ama çok daha uzun süre taze sayıyoruz (rapor zaten aylarca aynı).
REPORT_CACHE_FILE = os.path.join(os.path.dirname(__file__), "report_excerpts_cache.json")
REPORT_CACHE_MAX_AGE_DAYS = 60
REPORT_MAX_EXCERPT_CHARS = 6000     # ajanlara giden özet metnin karakter sınırı
REPORT_MAX_PDF_CHARS_TO_SCAN = 400_000   # PDF'ten okunacak maksimum karakter (çok büyük dosyalarda zaman aşımını önler)

# Yönetim beklentisi/guidance, CAPEX ve sipariş backlog'u gibi nitel
# sinyalleri içeren paragrafları PDF'ten "arayıp bulmak" için kullanılan
# anahtar kelimeler (basit, embedding'siz bir retrieval).
REPORT_KEYWORDS = [
    "beklenti", "hedef", "öngör", "yatırım", "capex", "yatırım harcaması",
    "sipariş", "backlog", "kapasite", "genişleme", "büyüme stratejisi",
    "yeni tesis", "yeni fabrika", "pazar payı", "ihracat", "talep",
]

# --- Round 1 tarama skoru ağırlıkları --------------------------------------
# Orijinal sistemin 25. bölümündeki Alpha Score ağırlıklarının, LLM'siz
# hesaplanabilen (tamamen sayısal) alt kümesi. Katalizör, guidance ve
# beklenti revizyonu gibi nitel bileşenler Round 2'de Bear/Bull/CRO
# ajanlarına bırakılıyor.
W_VALUATION = 0.20                 # ucuzluk (F/K, PD/DD, earnings yield)
W_MOMENTUM = 0.25                  # ciro + kâr büyümesi (YoY)
W_ACCELERATION = 0.15              # büyümenin hızlanması (Bölüm 14)
W_MARGIN = 0.10                    # marj trendi
W_BALANCE = 0.15                   # borçluluk / deleveraging (Bölüm 16)
W_DIVERGENCE = 0.15                # temel iyileşme var ama fiyat tepki vermemiş (Bölüm 21)

# Likidite: orijinal sistemin 5. bölümü "likidite problemi olan şirketi
# otomatik elemek zorunda değilsin ama açıkça işaretle" diyor. Bu yüzden
# eşiği sert bir filtre olarak değil, bir BAYRAK olarak kullanıyoruz.
LIQUIDITY_FLAG_THRESHOLD_TRY = 3_000_000   # bunun altı "DÜŞÜK LİKİDİTE" damgası alır
ABSOLUTE_MIN_TURNOVER_TRY = 250_000        # bunun altı gerçekten alınıp satılamaz, elenir
MIN_DATA_COMPLETENESS = 0.5                # bu orandan az veri varsa sıralamaya sokma

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
IY_MIN_DELAY = 2.0                # istekler arası minimum bekleme (saniye) — sabırlı ve nazik
IY_MAX_DELAY = 4.0                # istekler arası maksimum bekleme (saniye)
IY_MAX_RETRIES = 2                # zaman kısıtımız yok, tekrar denemeye değer
IY_RETRY_BACKOFF = 8              # denemeler arası bekleme (saniye)
IY_LOOKBACK_DAYS = 20             # kaç günlük fiyat/hacim geçmişine bakılacak

_debug_columns_printed = False    # ilk başarılı çekimde sütun isimlerini bir kez loglamak için


def _pick_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


PRICE_HISTORY_DAYS = 130          # 60 işlem günü getirisi için ~130 takvim günü gerekir


def fetch_price_snapshot(ticker):
    """İş Yatırım'dan son ~130 takvim günlük fiyat/hacim verisini çeker.
    Güncel fiyat, ortalama günlük TL hacmi (likidite) ve 5/20/60 işlem
    günlük getirileri (orijinal sistemin 21. bölümündeki Fundamental-Price
    Divergence hesabı için) döndürür."""
    global _debug_columns_printed

    end_date = datetime.now(TR_TZ).strftime("%d-%m-%Y")
    start_date = (datetime.now(TR_TZ) - timedelta(days=PRICE_HISTORY_DAYS)).strftime("%d-%m-%Y")

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

            closes = pd.to_numeric(df[close_col], errors="coerce").dropna()
            if closes.empty:
                raise ValueError("Geçerli kapanış fiyatı yok")

            last_price = float(closes.iloc[-1])

            def _return_over(n_days):
                """n işlem günü önceki fiyata göre yüzde getiri."""
                if len(closes) > n_days:
                    past = float(closes.iloc[-(n_days + 1)])
                    if past > 0:
                        return (last_price / past) - 1
                return None

            if vol_try_col is not None:
                turnover_series = pd.to_numeric(df[vol_try_col], errors="coerce")
            elif vol_lot_col is not None:
                turnover_series = pd.to_numeric(df[vol_lot_col], errors="coerce") * pd.to_numeric(df[close_col], errors="coerce")
            else:
                turnover_series = None

            avg_turnover = None
            if turnover_series is not None:
                tail = turnover_series.dropna().tail(20)
                if not tail.empty:
                    avg_turnover = float(tail.mean())

            return {
                "ticker": ticker,
                "price": last_price,
                "avg_daily_turnover_try": avg_turnover,
                "return_5d": _return_over(5),
                "return_20d": _return_over(20),
                "return_60d": _return_over(60),
            }
        except Exception as e:
            if attempt < IY_MAX_RETRIES:
                time.sleep(IY_RETRY_BACKOFF)
            else:
                print(f"  {ticker}: fiyat verisi alınamadı ({e})")
                return None
    return None


ROUND1_TIME_BUDGET_SECONDS = 5 * 3600   # Round 1 en fazla ~5 saat sürsün (18:30-10:00 arası bolca pay var)
CONSECUTIVE_FAILURE_CIRCUIT_BREAKER = 150   # bu kadar üst üste başarısızlık = gerçek bir engelleme, dur


def _pct_rank(series, ascending=True):
    """Yüzdelik dilim sıralaması; eksik veri nötr (0.5) kabul edilir."""
    return series.rank(pct=True, ascending=ascending, na_option="keep").fillna(0.5)


def _compute_screening_scores(df):
    """Orijinal sistemin 25. bölümündeki Alpha Score mantığının sayısal
    (LLM'siz) karşılığı. Her bileşen kendi içinde evrene göre yüzdelik
    dilime çevrilir, sonra ağırlıklandırılır."""

    # --- A) DEĞERLEME (ucuzluk) ---------------------------------------------
    # Earnings yield = 1/(F/K); yüksek olan daha ucuz demek
    df["earnings_yield"] = df["trailing_pe"].apply(
        lambda x: (1.0 / x) if (isinstance(x, (int, float)) and x and x > 0) else None
    )
    score_ey = _pct_rank(df["earnings_yield"], ascending=True)
    score_pb = _pct_rank(df["price_to_book"].where(df["price_to_book"] > 0), ascending=False)
    valuation_score = (score_ey * 0.6) + (score_pb * 0.4)

    # --- B) FINANCIAL MOMENTUM (Bölüm 13) -----------------------------------
    score_rev_g = _pct_rank(df["revenue_growth"], ascending=True)
    score_earn_g = _pct_rank(df["earnings_growth"], ascending=True)
    momentum_score = (score_rev_g * 0.4) + (score_earn_g * 0.6)

    # --- C) EARNINGS ACCELERATION (Bölüm 14) --------------------------------
    score_rev_acc = _pct_rank(df["revenue_acceleration"], ascending=True)
    score_earn_acc = _pct_rank(df["earnings_acceleration"], ascending=True)
    acceleration_score = (score_rev_acc * 0.4) + (score_earn_acc * 0.6)

    # --- D) MARJ TRENDİ -----------------------------------------------------
    score_gm = _pct_rank(df["gross_margin_trend"], ascending=True)
    score_om = _pct_rank(df["operating_margin_trend"], ascending=True)
    margin_score = (score_gm * 0.5) + (score_om * 0.5)

    # --- E) BİLANÇO (Bölüm 16) ----------------------------------------------
    # Düşük borç iyi (ascending=False), deleveraging (borç azalması) iyi
    score_nd = _pct_rank(df["net_debt_to_equity"], ascending=False)
    score_delev = _pct_rank(df["deleveraging"], ascending=True)
    score_roe = _pct_rank(df["roe"], ascending=True)
    balance_score = (score_nd * 0.4) + (score_delev * 0.25) + (score_roe * 0.35)

    # --- F) FUNDAMENTAL-PRICE DIVERGENCE (Bölüm 21) -------------------------
    # Aradığımız yapı: Temeller ↑↑ ama fiyat ↔ veya ↓ (henüz fiyatlanmamış)
    # "fundamental güç" ile "fiyat tepkisi" arasındaki farkı ölçüyoruz.
    fundamental_strength = (momentum_score * 0.5) + (acceleration_score * 0.3) + (margin_score * 0.2)
    price_response = (
        _pct_rank(df["return_20d"], ascending=True) * 0.5
        + _pct_rank(df["return_60d"], ascending=True) * 0.5
    )
    # Fark pozitifse: temel güçlü ama fiyat tepki vermemiş -> fırsat
    divergence_raw = fundamental_strength - price_response
    divergence_score = _pct_rank(divergence_raw, ascending=True)

    df["score_valuation"] = valuation_score
    df["score_momentum"] = momentum_score
    df["score_acceleration"] = acceleration_score
    df["score_margin"] = margin_score
    df["score_balance"] = balance_score
    df["score_divergence"] = divergence_score

    df["screening_score"] = (
        valuation_score * W_VALUATION
        + momentum_score * W_MOMENTUM
        + acceleration_score * W_ACCELERATION
        + margin_score * W_MARGIN
        + balance_score * W_BALANCE
        + divergence_score * W_DIVERGENCE
    ) * 100

    return df


# Büyük/orta/küçük ölçek dengesi: kullanıcı talebi üzerine eklendi. Ham
# skora göre saf sıralama küçük şirketleri (düşük baz etkisiyle şişen
# büyüme yüzdeleri sayesinde) sistematik olarak domine ediyordu. Bu yüzden
# MAX_DEEP_CANDIDATES kontenjanını üç piyasa değeri katmanına bölüştürüyoruz
# — her katmanda kendi içindeki en iyi skorlu şirketler seçiliyor.
CAP_BUCKET_SLOTS = {"Büyük": 8, "Orta": 7, "Küçük": 7}   # toplam = MAX_DEEP_CANDIDATES


def _stratified_candidate_selection(df):
    """df zaten screening_score'a göre sıralı. Piyasa değerine göre üç
    katmana ayırıp her katmandan CAP_BUCKET_SLOTS kadar en iyi skorluyu
    seçer. Bir katmanda yeterli şirket yoksa, boşta kalan kontenjan genel
    sıralamadan (katman gözetmeksizin) tamamlanır."""
    known_mcap = df[df["market_cap"].notna() & (df["market_cap"] > 0)]

    if len(known_mcap) >= 10:
        q1 = known_mcap["market_cap"].quantile(1 / 3)
        q2 = known_mcap["market_cap"].quantile(2 / 3)

        def bucket_for(mcap):
            if pd.isna(mcap) or mcap <= 0:
                return "Küçük"   # piyasa değeri hesaplanamayanlar genelde çok küçük/veri zayıf şirketler
            if mcap >= q2:
                return "Büyük"
            if mcap >= q1:
                return "Orta"
            return "Küçük"

        df = df.copy()
        df["cap_bucket"] = df["market_cap"].apply(bucket_for)
    else:
        # Yeterli piyasa değeri verisi yoksa katmanlama anlamsız, düz sıralamaya dön
        df = df.copy()
        df["cap_bucket"] = "Bilinmiyor"

    selected_parts = []
    selected_tickers = set()
    for bucket, slots in CAP_BUCKET_SLOTS.items():
        bucket_df = df[df["cap_bucket"] == bucket].head(slots)
        selected_parts.append(bucket_df)
        selected_tickers.update(bucket_df["ticker"])

    selected = pd.concat(selected_parts) if selected_parts else df.head(0)

    # Katmanlardan biri dolduramadıysa (örn. az şirket varsa) boşta kalan
    # kontenjanı genel sıralamadan (katman gözetmeksizin) tamamla.
    remaining_slots = MAX_DEEP_CANDIDATES - len(selected)
    if remaining_slots > 0:
        rest = df[~df["ticker"].isin(selected_tickers)].head(remaining_slots)
        selected = pd.concat([selected, rest])

    return selected.sort_values("screening_score", ascending=False).head(MAX_DEEP_CANDIDATES)


def round1_screen(tickers):
    """TAM FUNDAMENTAL TARAMA (orijinal sistemin 49. bölümündeki ROUND 1).

    Her hisse için:
      - fiyat/hacim geçmişi (5/20/60 günlük getiriler dahil)
      - bilanço/gelir tablosu (önbellekli — çeyrekte bir yenilenir)
    çekilir; ardından değerleme, momentum, ivmelenme, marj, bilanço ve
    fundamental-price divergence bileşenlerinden bir tarama skoru hesaplanır.

    Likidite artık sert bir filtre DEĞİL (orijinal sistem 5. bölüm: "likidite
    problemi olan şirketi otomatik elemek zorunda değilsin ama açıkça
    işaretle") — sadece gerçekten alınıp satılamayacak kadar sığ olanlar
    elenir, geri kalanı DÜŞÜK LİKİDİTE bayrağıyla işaretlenir.
    """
    rows = []
    total = len(tickers)
    consecutive_failures = 0
    start_time = time.time()
    cache = load_fin_cache()
    fetched_count = 0

    print(f"Önbellekte {len(cache)} şirketin bilanço verisi var.")

    for i, tk in enumerate(tickers, 1):
        elapsed = time.time() - start_time
        if elapsed > ROUND1_TIME_BUDGET_SECONDS:
            print(f"Round 1 zaman bütçesini ({ROUND1_TIME_BUDGET_SECONDS}s) aştı, {i-1}/{total} hisseyle devam ediliyor.")
            break

        price_data = fetch_price_snapshot(tk)
        if not price_data:
            consecutive_failures += 1
            if consecutive_failures >= CONSECUTIVE_FAILURE_CIRCUIT_BREAKER:
                print(f"Üst üste {CONSECUTIVE_FAILURE_CIRCUIT_BREAKER} başarısız istek, veri kaynağı muhtemelen bu IP'yi engelledi. Duruyorum.")
                break
            time.sleep(random.uniform(IY_MIN_DELAY, IY_MAX_DELAY))
            continue

        consecutive_failures = 0

        # Gerçekten alınıp satılamayacak kadar sığ olanları erkenden ele —
        # bunlar için bilanço çekmeye değmez (zaman tasarrufu)
        turnover = price_data.get("avg_daily_turnover_try") or 0
        if turnover < ABSOLUTE_MIN_TURNOVER_TRY:
            time.sleep(random.uniform(IY_MIN_DELAY, IY_MAX_DELAY))
            continue

        fundamentals, was_fetched = get_fundamentals_cached(tk, cache)
        if was_fetched:
            fetched_count += 1
            if fetched_count % CACHE_SAVE_EVERY == 0:
                save_fin_cache(cache)   # kesinti olursa ilerleme kaybolmasın

        row = {**price_data, **fundamentals}
        row["low_liquidity_flag"] = turnover < LIQUIDITY_FLAG_THRESHOLD_TRY
        rows.append(row)

        if i % 25 == 0 or i == total:
            print(f"  ...{i}/{total} hisse tarandı (veri toplanan: {len(rows)}, "
                  f"bilanço ağdan çekilen: {fetched_count}, geçen süre: {elapsed:.0f}s)")

        time.sleep(random.uniform(IY_MIN_DELAY, IY_MAX_DELAY))

    save_fin_cache(cache)

    df = pd.DataFrame(rows)
    print(f"Veri toplanabilen hisse sayısı: {len(df)} / {total}")
    if df.empty:
        print("Hiçbir hisse için veri toplanamadı.")
        return df

    # --- Fiyata bağlı çarpanları hesapla (F/K, PD/DD, Piyasa Değeri) --------
    pe_list, pb_list, mcap_list = [], [], []
    for _, r in df.iterrows():
        eps_ann = r.get("_eps_annualized")
        ni_ann = r.get("_net_income_annualized")
        equity = r.get("_equity_latest")
        price = r.get("price")

        pe = None
        if eps_ann and eps_ann > 0 and price:
            candidate_pe = price / eps_ann
            if 1.0 <= candidate_pe <= 500:
                pe = round(candidate_pe, 2)

        pb = None
        market_cap = None
        if eps_ann and ni_ann and abs(eps_ann) > 1e-9 and equity and equity > 0 and price:
            shares = ni_ann / eps_ann
            if shares > 0:
                market_cap = round(price * shares, 0)
                bvps = equity / shares
                if bvps > 0:
                    candidate_pb = price / bvps
                    if 0.1 <= candidate_pb <= 50:
                        pb = round(candidate_pb, 2)

        pe_list.append(pe)
        pb_list.append(pb)
        mcap_list.append(market_cap)

    df["trailing_pe"] = pe_list
    df["price_to_book"] = pb_list
    df["market_cap"] = mcap_list

    # --- Veri tamlığı kontrolü ----------------------------------------------
    # Verisi çok eksik olan şirketler nötr (0.5) skorlarla yapay olarak
    # yukarı çıkabilir; bunları sıralamaya sokmuyoruz.
    key_fields = ["trailing_pe", "price_to_book", "revenue_growth", "earnings_growth",
                  "roe", "net_debt_to_equity", "return_20d"]
    df["data_completeness"] = df[key_fields].notna().sum(axis=1) / len(key_fields)

    before = len(df)
    df = df[df["data_completeness"] >= MIN_DATA_COMPLETENESS].copy()
    print(f"Yeterli veriye sahip hisse sayısı: {len(df)} / {before}")
    if df.empty:
        return df

    # --- Tarama skorunu hesapla ---------------------------------------------
    df = _compute_screening_scores(df)
    df = df.sort_values("screening_score", ascending=False)

    top = _stratified_candidate_selection(df)
    print("\nRound 1 tarama sonucu — Round 2'ye giden adaylar:")
    for _, r in top.iterrows():
        flag = " [DÜŞÜK LİKİDİTE]" if r.get("low_liquidity_flag") else ""
        cap = f" [{r.get('cap_bucket', '?')}]"
        print(f"  {r['ticker']}: skor {r['screening_score']:.1f}{cap} "
              f"(değerleme {r['score_valuation']:.2f}, momentum {r['score_momentum']:.2f}, "
              f"ivme {r['score_acceleration']:.2f}, bilanço {r['score_balance']:.2f}, "
              f"divergence {r['score_divergence']:.2f}){flag}")
    print()

    return top


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


def _series_for_item(df, item_code, quarter_cols):
    """Bir kalemin tüm çeyrek değerlerini [(yıl, dönem_ay, değer), ...] olarak
    kronolojik sırada döndürür."""
    row = _get_item_row(df, item_code)
    if row is None:
        return []
    out = []
    for year, period, col in quarter_cols:
        val = row.get(col)
        if pd.notna(val):
            try:
                out.append((year, period, float(val)))
            except (TypeError, ValueError):
                continue
    return out


def _decumulate(series):
    """İş Yatırım gelir tablosu kalemleri KÜMÜLATİFtir (3/6/9/12 aylık birikimli).
    Bunları tek tek çeyreklere ayırır: Q2 = 6ay - 3ay, Q3 = 9ay - 6ay ...
    Böylece çeyreklik büyüme ve ivmelenme doğru hesaplanabilir (Bölüm 13/14)."""
    by_year = {}
    for year, period, val in series:
        by_year.setdefault(year, {})[period] = val

    quarterly = []
    for year in sorted(by_year):
        periods = by_year[year]
        for period in sorted(periods):
            q_index = period // 3          # 3->1, 6->2, 9->3, 12->4
            prev_period = period - 3
            if prev_period >= 3 and prev_period in periods:
                value = periods[period] - periods[prev_period]
            else:
                value = periods[period]     # yılın ilk çeyreği, zaten kümülatif değil
            quarterly.append((year, q_index, value))
    return quarterly


def _yoy_growth_for_quarter(quarterly, offset=0):
    """Sondan `offset` çeyrek geriden başlayarak, o çeyreğin bir önceki yılın
    aynı çeyreğine göre büyümesini hesaplar."""
    if len(quarterly) < offset + 1:
        return None
    year, q_index, value = quarterly[-(offset + 1)]
    for y, q, v in quarterly:
        if y == year - 1 and q == q_index:
            if v and v != 0:
                return (value / abs(v)) - 1 if v > 0 else None
            return None
    return None


def _margin_trend(revenue_q, profit_q):
    """Son çeyrek marjı ile bir önceki yılın aynı çeyreğindeki marjı
    karşılaştırır (marj genişlemesi pozitif sinyaldir)."""
    if not revenue_q or not profit_q:
        return None
    rev_map = {(y, q): v for y, q, v in revenue_q}
    prof_map = {(y, q): v for y, q, v in profit_q}

    latest_key = revenue_q[-1][:2]
    prior_key = (latest_key[0] - 1, latest_key[1])

    def margin(key):
        rev = rev_map.get(key)
        prof = prof_map.get(key)
        if rev and rev > 0 and prof is not None:
            return prof / rev
        return None

    m_now, m_prior = margin(latest_key), margin(prior_key)
    if m_now is None or m_prior is None:
        return None
    return m_now - m_prior


def _fetch_financials_df(ticker):
    """İş Yatırım'dan ham bilanço DataFrame'ini çeker (format denemeleriyle)."""
    this_year = datetime.now(TR_TZ).year
    for group in FIN_GROUPS_TO_TRY:
        try:
            candidate_df = isy_fetch_financials(
                symbols=ticker, start_year=this_year - 2, end_year=this_year,
                financial_group=group,
            )
            if candidate_df is not None and not candidate_df.empty:
                return candidate_df
        except Exception:
            pass
        time.sleep(FIN_REQUEST_DELAY)
    return None


def compute_fundamentals(ticker):
    """Bir şirketin tüm sayısal temel göstergelerini hesaplar.
    Orijinal sistemin 13 (momentum), 14 (ivmelenme), 16 (bilanço) ve
    kısmen 15 (nakit akış kalitesi) bölümlerine karşılık gelir.
    Veri yoksa ilgili alan None kalır — ASLA uydurulmaz."""
    out = {
        "name": ticker, "sector": "N/A", "industry": "N/A",
        "revenue_growth": None, "earnings_growth": None,
        "revenue_growth_prev_q": None, "earnings_growth_prev_q": None,
        "revenue_acceleration": None, "earnings_acceleration": None,
        "gross_margin_trend": None, "operating_margin_trend": None,
        "roe": None, "debt_to_equity": None, "net_debt_to_equity": None,
        "deleveraging": None, "cfo_to_net_income": None,
        "_eps_annualized": None, "_net_income_annualized": None,
        "_equity_latest": None,
    }

    df = _fetch_financials_df(ticker)
    if df is None:
        return out

    quarter_cols = _quarter_columns_sorted(df)
    if not quarter_cols:
        return out

    try:
        # --- Kümülatif (yıl-başından-bugüne) değerler -----------------------
        revenue_latest, revenue_prior, period = _latest_and_prior_year(df, ITEM_REVENUE, quarter_cols)
        net_income_latest, net_income_prior, _ = _latest_and_prior_year(df, ITEM_NET_INCOME_PARENT, quarter_cols)
        eps_latest, _, _ = _latest_and_prior_year(df, ITEM_EPS, quarter_cols)
        equity_latest, _, _ = _latest_and_prior_year(df, ITEM_PARENT_EQUITY, quarter_cols)
        st_liab, _, _ = _latest_and_prior_year(df, ITEM_SHORT_TERM_LIAB, quarter_cols)
        lt_liab, _, _ = _latest_and_prior_year(df, ITEM_LONG_TERM_LIAB, quarter_cols)
        cash, _, _ = _latest_and_prior_year(df, ITEM_CASH, quarter_cols)
        st_fin_debt, _, _ = _latest_and_prior_year(df, ITEM_ST_FIN_DEBT, quarter_cols)
        lt_fin_debt, _, _ = _latest_and_prior_year(df, ITEM_LT_FIN_DEBT, quarter_cols)
        cfo_latest, _, _ = _latest_and_prior_year(df, ITEM_CFO, quarter_cols)

        # --- Bölüm 13: Financial Momentum (YoY, kümülatif bazda) ------------
        if revenue_latest is not None and revenue_prior and revenue_prior > 0:
            g = (revenue_latest / revenue_prior) - 1
            if -3.0 <= g <= 20.0:
                out["revenue_growth"] = round(g, 4)
        if net_income_latest is not None and net_income_prior and net_income_prior > 0:
            g = (net_income_latest / net_income_prior) - 1
            if -3.0 <= g <= 20.0:
                out["earnings_growth"] = round(g, 4)

        # --- Bölüm 14: Earnings Acceleration (çeyreklik bazda) --------------
        rev_series = _series_for_item(df, ITEM_REVENUE, quarter_cols)
        ni_series = _series_for_item(df, ITEM_NET_INCOME_PARENT, quarter_cols)
        gp_series = _series_for_item(df, ITEM_GROSS_PROFIT, quarter_cols)
        op_series = _series_for_item(df, ITEM_OPERATING_PROFIT, quarter_cols)

        rev_q = _decumulate(rev_series)
        ni_q = _decumulate(ni_series)
        gp_q = _decumulate(gp_series)
        op_q = _decumulate(op_series)

        rev_g_now = _yoy_growth_for_quarter(rev_q, 0)
        rev_g_prev = _yoy_growth_for_quarter(rev_q, 1)
        ni_g_now = _yoy_growth_for_quarter(ni_q, 0)
        ni_g_prev = _yoy_growth_for_quarter(ni_q, 1)

        # Makul aralık kelepçesi: bir çeyreğin bazı sıfıra çok yakınsa (veya
        # işaret değiştiriyorsa) YoY büyüme oranı matematiksel olarak
        # patlayabilir (örn. %-2527 gibi anlamsız değerler). Bunları N/A'ya
        # çeviriyoruz — F/K ve PD/DD'de yaptığımız korumanın aynısı.
        def _clamp_growth(v, lower=-3.0, upper=20.0):
            if v is None or v < lower or v > upper:
                return None
            return v

        def _clamp_accel(v, lower=-15.0, upper=15.0):
            if v is None or v < lower or v > upper:
                return None
            return v

        rev_g_now = _clamp_growth(rev_g_now)
        rev_g_prev = _clamp_growth(rev_g_prev)
        ni_g_now = _clamp_growth(ni_g_now)
        ni_g_prev = _clamp_growth(ni_g_prev)

        if rev_g_now is not None:
            out["revenue_growth_prev_q"] = round(rev_g_prev, 4) if rev_g_prev is not None else None
            if rev_g_prev is not None:
                accel = _clamp_accel(rev_g_now - rev_g_prev)
                out["revenue_acceleration"] = round(accel, 4) if accel is not None else None
        if ni_g_now is not None:
            out["earnings_growth_prev_q"] = round(ni_g_prev, 4) if ni_g_prev is not None else None
            if ni_g_prev is not None:
                accel = _clamp_accel(ni_g_now - ni_g_prev)
                out["earnings_acceleration"] = round(accel, 4) if accel is not None else None

        # --- Marj trendi ---------------------------------------------------
        gm_trend = _margin_trend(rev_q, gp_q)
        if gm_trend is not None:
            out["gross_margin_trend"] = round(gm_trend, 4)
        om_trend = _margin_trend(rev_q, op_q)
        if om_trend is not None:
            out["operating_margin_trend"] = round(om_trend, 4)

        # --- Yıllıklandırma ------------------------------------------------
        net_income_annualized = _annualize(net_income_latest, period)
        eps_annualized = _annualize(eps_latest, period)
        out["_eps_annualized"] = eps_annualized
        out["_net_income_annualized"] = net_income_annualized
        out["_equity_latest"] = equity_latest

        roe = _safe_div(net_income_annualized, equity_latest)
        if roe is not None:
            out["roe"] = round(roe, 4)

        # --- Bölüm 16: Balance Sheet Engine --------------------------------
        if st_liab is not None and lt_liab is not None:
            dte = _safe_div(st_liab + lt_liab, equity_latest)
            if dte is not None:
                out["debt_to_equity"] = round(dte, 2)

        if st_fin_debt is not None or lt_fin_debt is not None:
            gross_debt = (st_fin_debt or 0) + (lt_fin_debt or 0)
            net_debt = gross_debt - (cash or 0)
            nd_e = _safe_div(net_debt, equity_latest)
            if nd_e is not None:
                out["net_debt_to_equity"] = round(nd_e, 2)

            # Deleveraging: net borç/özkaynak bir yıl öncesine göre düşüyor mu?
            prev_st, prev_lt, prev_cash, prev_eq = None, None, None, None
            _, prev_st, _ = _latest_and_prior_year(df, ITEM_ST_FIN_DEBT, quarter_cols)
            _, prev_lt, _ = _latest_and_prior_year(df, ITEM_LT_FIN_DEBT, quarter_cols)
            _, prev_cash, _ = _latest_and_prior_year(df, ITEM_CASH, quarter_cols)
            _, prev_eq, _ = _latest_and_prior_year(df, ITEM_PARENT_EQUITY, quarter_cols)
            if prev_eq and prev_eq > 0:
                prev_net_debt = (prev_st or 0) + (prev_lt or 0) - (prev_cash or 0)
                prev_nd_e = prev_net_debt / prev_eq
                if nd_e is not None:
                    out["deleveraging"] = round(prev_nd_e - nd_e, 2)  # pozitif = borç azalıyor

        # --- Bölüm 15: Earnings Quality (CFO / Net Kâr) --------------------
        if cfo_latest is not None and net_income_latest and net_income_latest > 0:
            out["cfo_to_net_income"] = round(cfo_latest / net_income_latest, 2)

    except Exception as e:
        print(f"  {ticker}: oran hesaplanırken hata ({e}), mevcut alanlarla devam")

    return out


# --- Bilanço önbelleği ------------------------------------------------------
def load_fin_cache():
    if os.path.exists(FIN_CACHE_FILE):
        try:
            with open(FIN_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_fin_cache(cache):
    try:
        with open(FIN_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
    except Exception as e:
        print(f"Önbellek yazılamadı: {e}")


def get_fundamentals_cached(ticker, cache):
    """Önbellekte taze (CACHE_MAX_AGE_DAYS'ten yeni) kayıt varsa onu kullanır,
    yoksa İş Yatırım'dan çeker ve önbelleğe yazar.
    Bilançolar çeyrekte bir değiştiği için bu, her gün 720 şirketin
    bilançosunu yeniden çekmeyi (saatler) gereksiz kılıyor."""
    today = datetime.now(TR_TZ).date()
    entry = cache.get(ticker)
    if entry and entry.get("fetched_at"):
        try:
            fetched = datetime.strptime(entry["fetched_at"], "%Y-%m-%d").date()
            if (today - fetched).days < CACHE_MAX_AGE_DAYS:
                return entry["data"], False   # False = ağdan çekilmedi
        except Exception:
            pass

    data = compute_fundamentals(ticker)
    cache[ticker] = {"fetched_at": today.strftime("%Y-%m-%d"), "data": data}
    return data, True


# --------------------------------------------------------------------------
# FAZ 5: RAG — FAALİYET RAPORU ANALİZİ
# --------------------------------------------------------------------------
def _find_latest_far_pdf_bytes(ticker):
    """KAP'tan bir şirketin en güncel Faaliyet Raporu PDF'ini bulur ve
    indirir. pykap kütüphanesinin BISTCompany sınıfını kullanır. Herhangi bir
    adımda hata olursa (rapor yok, PDF linki bulunamadı, ağ hatası vb.)
    sessizce None döner — bu modül olmadan da sistem çalışmaya devam eder."""
    if not RAG_AVAILABLE:
        return None, None

    try:
        comp = pykap.BISTCompany(ticker)
        reports = comp.get_disclosures("FAR")
        if not reports:
            return None, None

        reports_sorted = sorted(reports, key=lambda r: r.get("publishDate", ""), reverse=True)
        latest = reports_sorted[0]
        disc_index = latest.get("disclosureIndex")
        if not disc_index:
            return None, None

        announcement_url = f"https://www.kap.org.tr/tr/Bildirim/{disc_index}"
        resp = requests.get(announcement_url, timeout=30)
        resp.raise_for_status()

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html5lib")
        pdf_link_tag = soup.select("a.modal-attachment.type-xsmall.bi-sky-black.maximize")
        if not pdf_link_tag or not pdf_link_tag[0].get("href"):
            return None, latest

        pdf_url = "https://www.kap.org.tr" + pdf_link_tag[0]["href"]
        pdf_resp = requests.get(pdf_url, timeout=60)
        content_type = pdf_resp.headers.get("Content-Type", "").lower()
        if pdf_resp.status_code == 200 and "pdf" in content_type:
            return pdf_resp.content, latest
        return None, latest
    except Exception as e:
        print(f"  {ticker}: faaliyet raporu PDF'i bulunamadı/indirilemedi ({e})")
        return None, None


def _extract_pdf_text(pdf_bytes):
    """PDF byte içeriğinden düz metin çıkarır. Taranmış (image-only) PDF'lerde
    ya da bozuk dosyalarda boş/kısmi metin dönebilir — bu normaldir, hata
    fırlatmaz."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        chunks = []
        total_chars = 0
        for page in reader.pages:
            text = page.extract_text() or ""
            chunks.append(text)
            total_chars += len(text)
            if total_chars >= REPORT_MAX_PDF_CHARS_TO_SCAN:
                break
        return "\n".join(chunks)
    except Exception as e:
        print(f"  PDF metni çıkarılamadı: {e}")
        return ""


def _extract_keyword_excerpt(full_text):
    """Basit, embedding'siz bir 'retrieval': REPORT_KEYWORDS'ten en az birini
    içeren paragrafları toplayıp REPORT_MAX_EXCERPT_CHARS'a kadar birleştirir.
    Gerçek bir vektör DB değil ama tek bir belge için (aynı şirketin kendi
    raporu) pratikte aynı işi görüyor: yönetim beklentisi/CAPEX/sipariş gibi
    nitel sinyalleri içeren bölümleri buluyor."""
    if not full_text:
        return None

    # Paragrafları basit şekilde ayır (boş satırlar, ya da uzun tek bloklarda
    # cümle bazlı ayır)
    raw_paragraphs = re.split(r"\n\s*\n", full_text)
    if len(raw_paragraphs) < 5:
        raw_paragraphs = re.split(r"(?<=[.!?])\s+", full_text)

    keywords_lower = [k.lower() for k in REPORT_KEYWORDS]
    matched = []
    seen = set()
    for para in raw_paragraphs:
        para = para.strip()
        if len(para) < 30 or para in seen:
            continue
        low = para.lower()
        if any(kw in low for kw in keywords_lower):
            matched.append(para)
            seen.add(para)

    if not matched:
        return None

    excerpt = "\n---\n".join(matched)
    if len(excerpt) > REPORT_MAX_EXCERPT_CHARS:
        excerpt = excerpt[:REPORT_MAX_EXCERPT_CHARS] + "…"
    return excerpt


def load_report_cache():
    if os.path.exists(REPORT_CACHE_FILE):
        try:
            with open(REPORT_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_report_cache(cache):
    try:
        with open(REPORT_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
    except Exception as e:
        print(f"Rapor önbelleği yazılamadı: {e}")


def get_report_excerpt_cached(ticker, cache):
    """Önbellekte taze (REPORT_CACHE_MAX_AGE_DAYS'ten yeni) bir alıntı varsa
    onu döner, yoksa KAP'tan yeni faaliyet raporunu çeker, anahtar kelime
    filtrelemesi yapar ve önbelleğe yazar. Rapor bulunamazsa None döner —
    bu durumda Bull/Bear ajanları bu bölümü N/A olarak görür, uydurmaz."""
    today = datetime.now(TR_TZ).date()
    entry = cache.get(ticker)
    if entry and entry.get("fetched_at"):
        try:
            fetched = datetime.strptime(entry["fetched_at"], "%Y-%m-%d").date()
            if (today - fetched).days < REPORT_CACHE_MAX_AGE_DAYS:
                return entry.get("excerpt"), entry.get("report_period")
        except Exception:
            pass

    if not RAG_AVAILABLE:
        return None, None

    pdf_bytes, meta = _find_latest_far_pdf_bytes(ticker)
    excerpt, period = None, None
    if pdf_bytes:
        full_text = _extract_pdf_text(pdf_bytes)
        excerpt = _extract_keyword_excerpt(full_text)
        if meta:
            period = f"{meta.get('year', '')} {meta.get('period', '')}".strip()

    cache[ticker] = {
        "fetched_at": today.strftime("%Y-%m-%d"),
        "excerpt": excerpt,
        "report_period": period,
    }
    return excerpt, period


def fetch_fundamentals_isyatirim(ticker):
    """Geriye dönük uyumluluk için ince sarmalayıcı (Round 2 hâlâ bunu
    çağırabiliyor). Round 1 artık get_fundamentals_cached kullanıyor."""
    return compute_fundamentals(ticker)


DATA_BLOCK_TEMPLATE = """ŞİRKET: {name} ({ticker}){liquidity_note}
Ölçek Kategorisi: {cap_bucket} (BIST piyasa değerine göre)
Güncel Fiyat: {price} TL

--- DEĞERLEME ---
Trailing F/K: {trailing_pe}
PD/DD: {price_to_book}
ROE: {roe}

--- FINANCIAL MOMENTUM (yıl-başından-bugüne, YoY) ---
Ciro Büyümesi: {revenue_growth}
Net Kâr Büyümesi: {earnings_growth}

--- EARNINGS ACCELERATION (çeyreklik YoY büyümenin değişimi) ---
Ciro Büyümesi İvmesi: {revenue_acceleration}
Kâr Büyümesi İvmesi: {earnings_acceleration}
(pozitif = büyüme hızlanıyor, negatif = yavaşlıyor)

--- MARJ TRENDİ (geçen yılın aynı çeyreğine göre değişim) ---
Brüt Marj Değişimi: {gross_margin_trend}
Faaliyet Marjı Değişimi: {operating_margin_trend}

--- BİLANÇO ---
Borç/Özsermaye: {debt_to_equity}
Net Finansal Borç/Özsermaye: {net_debt_to_equity}
Deleveraging (yıllık net borç/özkaynak azalışı): {deleveraging}
(pozitif = borçluluk azalıyor)

--- KÂR KALİTESİ ---
İşletme Nakit Akışı / Net Kâr: {cfo_to_net_income}
(1'in altı veya negatif = kâr nakde dönüşmüyor, uyarı işareti)

--- FİYAT TEPKİSİ (Fundamental-Price Divergence için) ---
5 Günlük Getiri: {return_5d}
20 Günlük Getiri: {return_20d}
60 Günlük Getiri: {return_60d}

--- ROUND 1 TARAMA SKORLARI (0-1 arası, BIST evrenine göre yüzdelik dilim) ---
Değerleme: {score_valuation} | Momentum: {score_momentum} | İvmelenme: {score_acceleration}
Marj: {score_margin} | Bilanço: {score_balance} | Divergence: {score_divergence}
Veri Tamlığı: {data_completeness}

--- FAALİYET RAPORUNDAN ÖNE ÇIKAN BÖLÜMLER (varsa, {report_period}) ---
(Yönetim beklentisi, CAPEX/yatırım, sipariş/backlog, kapasite gibi anahtar
kelimeleri içeren paragraflar — rapor yoksa veya bu bölümler bulunamadıysa
N/A yazar, bu durumda bu kısımla ilgili UYDURMA YAPMA.)
{report_excerpt}"""

BEAR_PROMPT_TEMPLATE = """
Sen bir BIST Ayı (Bear) Analistisin. Görevin SADECE şu şirketteki riskleri,
"Value Trap" (değer tuzağı) olasılığını, borç/kârlılık/kâr kalitesi
sorunlarını ve en kötü senaryoyu olabildiğince güçlü savunmak. Şirketin
olumlu yanlarını bu analizde ELE ALMA — görevin kötümser tarafı zorlamak.

Özellikle şunlara dikkat et: büyümede yavaşlama (negatif ivmelenme), marj
daralması, artan borçluluk, işletme nakit akışının net kârın gerisinde
kalması (kâr kalitesi sorunu), düşük likidite ve döngüsel zirve kârı
olasılığı. Faaliyet raporundan alıntılar varsa (guidance, CAPEX, sipariş
bilgisi), bunlarda da riskli/belirsiz ifadeleri (örn. "zorlu piyasa
koşulları", "belirsizlik") yakalamaya çalış.

Bilmediğin/verilmeyen bilgiyi UYDURMA, "N/A" ise o konuda yorum yapma.

{data_block}

Yalnızca aşağıdaki JSON formatında yanıt ver, başka açıklama ekleme:
{{
  "bear_case": "<2-3 cümlelik Türkçe, en kötü senaryo argümanı>",
  "key_risks": ["<risk 1>", "<risk 2>"],
  "value_trap_risk": <0-100 arası tam sayı, bu şirketin bir değer tuzağı olma ihtimali>,
  "bear_fv": <TL cinsinden kötümser adil değer, sayı>
}}
"""

BULL_PROMPT_TEMPLATE = """
Sen bir BIST Boğa (Bull) Analistisin. Görevin SADECE şu şirketin büyüme
potansiyelini, olası katalizörlerini ve en iyi senaryoyu olabildiğince güçlü
savunmak. Şirketin risklerini bu analizde ELE ALMA — görevin iyimser tarafı
zorlamak.

Özellikle şunlara dikkat et: büyümede hızlanma (pozitif ivmelenme), marj
genişlemesi, borç azalması (deleveraging), güçlü kâr kalitesi ve
"temeller güçlü ama fiyat henüz tepki vermemiş" durumu (yani yüksek
divergence skoru = piyasa bu iyileşmeyi henüz fiyatlamamış olabilir).
Faaliyet raporundan alıntılar varsa, yönetimin büyüme hedeflerini, yeni
yatırım/kapasite planlarını veya sipariş/backlog bilgilerini katalizör
olarak kullanabilirsin.

Bilmediğin/verilmeyen bilgiyi UYDURMA, "N/A" ise o konuda yorum yapma.

{data_block}

Yalnızca aşağıdaki JSON formatında yanıt ver, başka açıklama ekleme:
{{
  "bull_case": "<2-3 cümlelik Türkçe, en iyi senaryo argümanı>",
  "catalysts": ["<katalizör 1>", "<katalizör 2>"],
  "growth_conviction": <0-100 arası tam sayı, büyüme tezine olan kanaat gücü>,
  "bull_fv": <TL cinsinden iyimser adil değer, sayı>
}}
"""

CRO_PROMPT_TEMPLATE = """
Sen bir Chief Risk Officer'sın (Yönetici Risk Sorumlusu). Aynı şirket için
Ayı ve Boğa analistlerinin ürettiği ZIT argümanlar aşağıda veriliyor.
Görevin: ikisini tarafsızca çarpıştırmak, hangi argümanın veriyle daha
tutarlı olduğuna karar vermek, çelişkileri tespit etmek ve dengeli, nihai
bir karar vermek. Tek bir tarafı kayırma; ikna edici olan neyse ona göre
karar ver.

{data_block}

--- AYI ANALİSTİNİN GÖRÜŞÜ ---
{bear_case}
Riskler: {key_risks}
Value Trap Riski: {value_trap_risk}/100
Bear FV: {bear_fv}

--- BOĞA ANALİSTİNİN GÖRÜŞÜ ---
{bull_case}
Katalizörler: {catalysts}
Büyüme Kanaati: {growth_conviction}/100
Bull FV: {bull_fv}

Yalnızca aşağıdaki JSON formatında yanıt ver, başka açıklama ekleme:
{{
  "alpha_score": <0-100 arası tam sayı, iki tarafı tarttıktan sonra nihai cazibe puanı>,
  "data_confidence": <0-100 arası tam sayı, verinin güvenilirliği/tamlığı>,
  "bear_fv": <TL, ayı analistinin değerine katılıyorsan aynen, katılmıyorsan düzeltilmiş hali>,
  "base_fv": <TL, senin ana senaryo (dengelenmiş) adil değerin>,
  "bull_fv": <TL, boğa analistinin değerine katılıyorsan aynen, katılmıyorsan düzeltilmiş hali>,
  "thesis_summary": "<2-3 cümlelik Türkçe, iki tarafı da yansıtan dengeli sentez>",
  "catalysts": ["<en inandırıcı katalizör(ler)>"],
  "risks": ["<en inandırıcı risk(ler)>"],
  "verdict": "<HIGH CONVICTION | ATTRACTIVE | WATCH | WEAKENING içinden biri>"
}}
"""


def _call_gemini_json(prompt, label):
    """Model yedekleme zincirini kullanarak Gemini'den JSON yanıt alır.
    Bear/Bull/CRO ajanlarının üçü de bu ortak fonksiyonu kullanır."""
    global _model_start_idx
    last_error = None
    for idx in range(_model_start_idx, len(MODEL_CANDIDATES)):
        model_name = MODEL_CANDIDATES[idx]
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.3,
                ),
            )
            data = json.loads(response.text)
            if idx != _model_start_idx:
                print(f"  [Model geçişi] {label}: bu istek için {model_name} kullanıldı")
            return data
        except Exception as e:
            last_error = e
            err_str = str(e)
            if "NOT_FOUND" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                print(f"  {model_name} kalıcı olarak kullanılamıyor ({label}): {e}")
                if idx == _model_start_idx:
                    _model_start_idx = idx + 1
                continue
            elif "UNAVAILABLE" in err_str:
                print(f"  {model_name} geçici olarak meşgul ({label}), sıradaki model deneniyor")
                continue
            else:
                print(f"  Gemini hatası ({label}): {e}")
                return None

    print(f"  Tüm modeller tükendi/erişilemedi ({label}): {last_error}")
    return None


def analyze_with_gemini(candidate):
    """Bear -> Bull -> CRO üç aşamalı analiz zinciri. Herhangi bir aşama
    başarısız olursa (tüm modeller tükenirse) None döner, o hisse atlanır."""
    data_block = DATA_BLOCK_TEMPLATE.format(**candidate)
    ticker = candidate["ticker"]

    bear_prompt = BEAR_PROMPT_TEMPLATE.format(data_block=data_block)
    bear = _call_gemini_json(bear_prompt, f"{ticker}-BEAR")
    if bear is None:
        return None
    time.sleep(2)

    bull_prompt = BULL_PROMPT_TEMPLATE.format(data_block=data_block)
    bull = _call_gemini_json(bull_prompt, f"{ticker}-BULL")
    if bull is None:
        return None
    time.sleep(2)

    cro_prompt = CRO_PROMPT_TEMPLATE.format(
        data_block=data_block,
        bear_case=bear.get("bear_case", "N/A"),
        key_risks=", ".join(bear.get("key_risks", []) or []) or "N/A",
        value_trap_risk=bear.get("value_trap_risk", "N/A"),
        bear_fv=bear.get("bear_fv", "N/A"),
        bull_case=bull.get("bull_case", "N/A"),
        catalysts=", ".join(bull.get("catalysts", []) or []) or "N/A",
        growth_conviction=bull.get("growth_conviction", "N/A"),
        bull_fv=bull.get("bull_fv", "N/A"),
    )
    cro = _call_gemini_json(cro_prompt, f"{ticker}-CRO")
    if cro is None:
        return None

    cro["ticker"] = candidate["ticker"]
    cro["name"] = candidate["name"]
    cro["price"] = candidate["price"]
    # Bear/Bull ham argümanlarını da saklıyoruz — ileride (ör. Telegram
    # botu /analiz komutu) detay göstermek istersek elimizde olsun.
    cro["_bear_case"] = bear.get("bear_case")
    cro["_bull_case"] = bull.get("bull_case")
    cro["_value_trap_risk"] = bear.get("value_trap_risk")
    return cro


def _finalize_price_based_ratios(fundamentals, price, ticker):
    """fetch_fundamentals_isyatirim'in bıraktığı ara değerlerden (EPS, net
    kâr, özkaynak) fiyata bağlı oranları (F/K, PD/DD, piyasa değeri) tamamlar.
    Ayrıca fiziksel olarak imkansız/anlamsız çıkan (veri hatasından
    kaynaklanan) çarpanları N/A'ya çevirerek Gemini'ye yanlış veri gitmesini
    engeller."""
    eps_ann = fundamentals.pop("_eps_annualized", None)
    net_income_ann = fundamentals.pop("_net_income_annualized", None)
    equity_latest = fundamentals.pop("_equity_latest", None)

    print(f"  [DEBUG {ticker}] eps_ann={eps_ann}, net_income_ann={net_income_ann}, equity_latest={equity_latest}, price={price}")

    shares_outstanding = None
    if eps_ann and net_income_ann and abs(eps_ann) > 1e-9:
        shares_outstanding = net_income_ann / eps_ann

    if eps_ann and eps_ann > 0:
        pe = price / eps_ann
        if 1.0 <= pe <= 500:
            fundamentals["trailing_pe"] = round(pe, 2)
        else:
            print(f"  [UYARI {ticker}] Hesaplanan F/K ({pe:.2f}) makul aralık dışında (1-500), N/A yapılıyor.")

    if shares_outstanding and shares_outstanding > 0:
        market_cap = price * shares_outstanding
        fundamentals["market_cap"] = round(market_cap, 0)
        if equity_latest and equity_latest > 0:
            book_value_per_share = equity_latest / shares_outstanding
            if book_value_per_share > 0:
                pb = price / book_value_per_share
                if 0.1 <= pb <= 50:
                    fundamentals["price_to_book"] = round(pb, 2)
                else:
                    print(f"  [UYARI {ticker}] Hesaplanan PD/DD ({pb:.2f}) makul aralık dışında (0.1-50), N/A yapılıyor.")

    return fundamentals


def _fmt_for_prompt(value, as_pct=False):
    """Prompt'a gidecek değerleri okunur hale getirir; eksikse N/A yazar."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    if isinstance(value, (int, float)):
        if as_pct:
            return f"%{value * 100:.2f}"
        return f"{value:.2f}"
    return str(value)


def _build_candidate_for_agents(row, report_excerpt=None, report_period=None):
    """Round 1'de zaten hesaplanmış veriden ajanlar için prompt sözlüğü üretir.
    Round 2 artık fundamental veriyi YENİDEN ÇEKMİYOR — Round 1'de toplanan
    veriyi kullanıyor. report_excerpt/report_period ise Faz 5 (RAG) tarafından
    ayrıca (bilanço dışı, KAP faaliyet raporundan) sağlanıyor."""
    pct_fields = {
        "revenue_growth", "earnings_growth", "revenue_acceleration",
        "earnings_acceleration", "gross_margin_trend", "operating_margin_trend",
        "roe", "return_5d", "return_20d", "return_60d",
    }
    plain_fields = {
        "trailing_pe", "price_to_book", "debt_to_equity", "net_debt_to_equity",
        "deleveraging", "cfo_to_net_income", "score_valuation", "score_momentum",
        "score_acceleration", "score_margin", "score_balance", "score_divergence",
        "data_completeness", "cap_bucket",
    }

    out = {
        "ticker": row["ticker"],
        "name": row.get("name") or row["ticker"],
        "price": round(float(row["price"]), 2),
        "liquidity_note": "  [UYARI: DÜŞÜK LİKİDİTE — işlem hacmi sığ]" if row.get("low_liquidity_flag") else "",
        "report_excerpt": report_excerpt if report_excerpt else "N/A",
        "report_period": report_period if report_period else "bilinmiyor",
    }
    for f in pct_fields:
        out[f] = _fmt_for_prompt(row.get(f), as_pct=True)
    for f in plain_fields:
        out[f] = _fmt_for_prompt(row.get(f))
    return out


def round2_deep_analysis(candidates_df):
    results = []
    records = candidates_df.to_dict("records")
    report_cache = load_report_cache()
    report_fetched_count = 0

    for i, row in enumerate(records, 1):
        ticker = row["ticker"]
        print(f"  Derin analiz {i}/{len(records)}: {ticker} (Bear -> Bull -> CRO)")

        report_excerpt, report_period = get_report_excerpt_cached(ticker, report_cache)
        report_fetched_count += 1
        if report_fetched_count % 5 == 0:
            save_report_cache(report_cache)   # kesinti olursa ilerleme kaybolmasın
        time.sleep(1.0)   # KAP'a da nazik davranalım

        candidate = _build_candidate_for_agents(row, report_excerpt, report_period)
        result = analyze_with_gemini(candidate)
        if result:
            # Round 1'den gelen sayısal verileri de sakla (rapor/hafıza için)
            result["low_liquidity_flag"] = bool(row.get("low_liquidity_flag"))
            result["screening_score"] = round(float(row.get("screening_score", 0)), 1)
            result["cap_bucket"] = row.get("cap_bucket")
            result["had_report_excerpt"] = bool(report_excerpt)
            results.append(result)
        time.sleep(SLEEP_BETWEEN_GEMINI_CALLS)

    save_report_cache(report_cache)
    return results


# --------------------------------------------------------------------------
# 4) STATE (HAFIZA) YÖNETİMİ
# --------------------------------------------------------------------------
def load_previous_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"date": None, "ranking": []}


HISTORY_FILE = os.path.join(os.path.dirname(__file__), "history.json")
HISTORY_MAX_DAYS = 20   # Faz 4 (thesis tracking) için birkaç haftalık hafıza tutuyoruz


def load_full_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def build_last_seen_map(history, exclude_date):
    """Her hisse için, bugün (exclude_date) HARİÇ en son hangi günde görüldüğünü
    ve o günkü tam tez/FV/verdict bilgisini bulur. Faz 4 (Thesis Tracking):
    'bu hissenin tezi son gördüğümüzden beri nasıl değişti' sorusunun temeli."""
    last_seen = {}
    for entry in sorted(history, key=lambda h: h.get("date", "")):
        date = entry.get("date")
        if date == exclude_date:
            continue
        for item in entry.get("ranking", []):
            tk = item.get("ticker")
            if not tk:
                continue
            last_seen[tk] = {**item, "date": date}   # kronolojik sırayla üzerine yazılır, sonuçta en yenisi kalır
    return last_seen


def save_state(ranking):
    today_str = datetime.now(TR_TZ).strftime("%Y-%m-%d")
    state = {"date": today_str, "ranking": ranking}
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

    # Faz 2 (haftalık özet) + Faz 4 (thesis tracking) için günlük geçmişi
    # biriktiriyoruz
    history = load_full_history()
    history = [h for h in history if h.get("date") != today_str]  # aynı gün varsa üzerine yaz
    history.append(state)
    history = history[-HISTORY_MAX_DAYS:]  # sadece son N günü tut

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


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


def build_report(ranked, last_seen_map, total_scanned, deep_count):
    today = datetime.now(TR_TZ).strftime("%d.%m.%Y")
    prev_map = last_seen_map   # geriye dönük isim uyumu için

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
        prev_rank_str = f"#{prev['rank']} ({prev['date']})" if prev else "Yeni"
        liq = " ⚠️ DÜŞÜK LİKİDİTE" if item.get("low_liquidity_flag") else ""
        cap = f" [{item.get('cap_bucket')}]" if item.get("cap_bucket") else ""
        lines.append(f"{medal} {i+1}. {item['ticker']} — {item.get('name','')}{cap}{liq}")
        lines.append(f"Alpha Score: {item.get('alpha_score','N/A')}/100  |  Data Confidence: {item.get('data_confidence','N/A')}/100")
        if item.get("screening_score") is not None:
            lines.append(f"Round 1 Tarama Skoru: {item.get('screening_score')}/100")
        lines.append(f"Fiyat: {fmt(item.get('price'))} TL")
        if item.get("had_report_excerpt"):
            lines.append("📄 Faaliyet raporu bulundu ve analize dahil edildi")
        lines.append(f"Bear FV: {fmt(item.get('bear_fv'))} | Base FV: {fmt(item.get('base_fv'))} | Bull FV: {fmt(item.get('bull_fv'))}")
        try:
            upside = (float(item.get("base_fv")) / float(item.get("price")) - 1) * 100
            lines.append(f"Base Upside: %{upside:,.1f}")
        except Exception:
            lines.append("Base Upside: N/A")
        lines.append(f"Önceki Sıra: {prev_rank_str}")

        # --- Faz 4: Thesis Tracking ----------------------------------------
        # Bu hisse daha önce (herhangi bir günde) taranmışsa, o zamanki
        # Base FV ve Verdict ile bugünküyle karşılaştırıp neyin değiştiğini
        # gösteriyoruz.
        if prev and prev.get("base_fv") is not None:
            try:
                old_fv = float(prev["base_fv"])
                new_fv = float(item.get("base_fv"))
                fv_change_pct = (new_fv / old_fv - 1) * 100 if old_fv else None
            except (TypeError, ValueError):
                fv_change_pct = None

            old_verdict = prev.get("verdict", "N/A")
            new_verdict = item.get("verdict", "N/A")

            change_bits = []
            if fv_change_pct is not None:
                arrow = "📈" if fv_change_pct > 0 else ("📉" if fv_change_pct < 0 else "➡️")
                change_bits.append(f"Base FV {fmt(prev['base_fv'])} → {fmt(item.get('base_fv'))} ({arrow} %{fv_change_pct:+.1f})")
            if old_verdict != new_verdict:
                change_bits.append(f"Verdict {old_verdict} → {new_verdict}")

            if change_bits:
                lines.append(f"📜 Tez Değişimi ({prev['date']}'den beri): " + "; ".join(change_bits))

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
        liq = " ⚠️" if item.get("low_liquidity_flag") else ""
        lines.append(
            f"{i}. {item['ticker']}{liq} | Alpha {item.get('alpha_score','N/A')} | "
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

    today_str = datetime.now(TR_TZ).strftime("%Y-%m-%d")
    full_history = load_full_history()
    last_seen_map = build_last_seen_map(full_history, exclude_date=today_str)

    report = build_report(
        ranked,
        last_seen_map,
        total_scanned=len(tickers),
        deep_count=len(candidates_df),
    )
    print(report)
    send_telegram_message(report)

    # Faz 4 (Thesis Tracking) için zenginleştirilmiş kayıt: artık sadece
    # sıra/skor değil, Base/Bear/Bull FV ve Verdict de saklanıyor ki
    # ileride "bu hissenin tezi nasıl değişti" karşılaştırması yapılabilsin.
    save_state([
        {
            "ticker": r["ticker"], "rank": r["rank"],
            "alpha_score": r.get("alpha_score"),
            "bear_fv": r.get("bear_fv"), "base_fv": r.get("base_fv"), "bull_fv": r.get("bull_fv"),
            "verdict": r.get("verdict"),
        }
        for r in ranked[:15]
    ])
    print("Tamamlandı.")


if __name__ == "__main__":
    main()
