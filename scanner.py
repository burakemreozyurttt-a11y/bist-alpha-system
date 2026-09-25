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
import numpy as np
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
from isyatirimhisse import fetch_index_data as isy_fetch_index_data
from google import genai
from google.genai import types
from visual_report import render_daily_report, render_followup_report, confidence_label, profile_label

# Faz 5 (RAG): faaliyet raporu PDF'lerini KAP'tan çekmek için
try:
    import pykap
    from pypdf import PdfReader
    RAG_AVAILABLE = True
except ImportError as _rag_import_error:
    RAG_AVAILABLE = False
    print(f"[UYARI] RAG modülü (Faz 5) yüklenemedi, bu çalıştırmada devre dışı: {_rag_import_error}")
    print("[UYARI] requirements.txt içinde pykap, pypdf, beautifulsoup4, html5lib satırlarının olduğundan emin ol.")

# Teknik Analiz Katmanı: intraday (gün içi) veri için borsapy (TradingView
# WebSocket tabanlı). Sadece Round 1'in 22 finalistine uygulanıyor.
try:
    import borsapy as bpy
    TECHNICAL_LAYER_AVAILABLE = True
except ImportError as _tech_import_error:
    TECHNICAL_LAYER_AVAILABLE = False
    print(f"[UYARI] Teknik analiz katmanı yüklenemedi, bu çalıştırmada devre dışı: {_tech_import_error}")
    print("[UYARI] requirements.txt içinde borsapy satırının olduğundan emin ol.")

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
REPORT_NOT_FOUND_RETRY_DAYS = 3     # "rapor bulunamadı" sonucu çok daha kısa süre önbelleklenir

# Faz 5 (RAG) şimdilik RAFA KALDIRILDI: KAP'ın bildirim sayfası JavaScript ile
# dolduruluyor, düz bir HTTP isteği PDF linkini göremiyor; resmi indirme API'si
# ise ücretli bir Borsa İstanbul sözleşmesi gerektiriyor. Kod duruyor, ileride
# (örn. headless tarayıcı ile) tekrar denenebilir — şimdilik her taramada 22
# kez boşuna KAP'a istek atıp zaman kaybetmemek için kapatıyoruz.
RAG_ENABLED = False
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
# VERSIONING — istatistiklerde metodoloji değişimini ayırmak için
# --------------------------------------------------------------------------
APP_VERSION = "V5.17"
SCORING_VERSION = "FINAL_ALPHA_70_30_OPPORTUNITY_GATE_V3"
VALUATION_VERSION = "BASE_UPSIDE_DISCIPLINE_V2"
CASE_ENGINE_VERSION = "CASE_LIFECYCLE_V3"


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
            }, False
        except Exception as e:
            # "Boş veri döndü" bizim kendi ValueError'ımız; "No data was
            # fetched" ise isyatirimhisse kütüphanesinin kendi hata mesajı —
            # ikisi de "bu kod gerçek bir hisse değil" anlamına geliyor.
            err_str = str(e)
            last_exception_was_empty_data = (
                "Boş veri döndü" in err_str or "No data was fetched" in err_str
            )
            if attempt < IY_MAX_RETRIES:
                time.sleep(IY_RETRY_BACKOFF)
            else:
                print(f"  {ticker}: fiyat verisi alınamadı ({e})")
                return None, last_exception_was_empty_data
    return None, False


ROUND1_TIME_BUDGET_SECONDS = 5 * 3600   # Round 1 en fazla ~5 saat sürsün (18:30-10:00 arası bolca pay var)
CONSECUTIVE_FAILURE_CIRCUIT_BREAKER = 150   # bu kadar üst üste başarısızlık = gerçek bir engelleme, dur

# --- "Ölü kod" (gerçek hisse olmayan) önbelleği --------------------------
# KAP'ın tam listesinde gerçek hisse olmayan çok sayıda kod var (il isimleri,
# varlık kiralama/faktoring/finansman şirketleri, test kodları vb.) — İş
# Yatırım bunlar için hep "veri yok" döner ve her denemede zaman kaybettirir.
# Bu kodları bir kez öğrenip önbelleğe alıyoruz, sonraki taramalarda baştan
# atlıyoruz (30 günde bir yeniden deneyip yeni IPO/kod değişikliklerini
# kaçırmıyoruz).
DEAD_TICKERS_FILE = os.path.join(os.path.dirname(__file__), "dead_tickers_cache.json")
DEAD_TICKER_RECHECK_DAYS = 30


def load_dead_tickers():
    if os.path.exists(DEAD_TICKERS_FILE):
        try:
            with open(DEAD_TICKERS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_dead_tickers(dead_map):
    try:
        with open(DEAD_TICKERS_FILE, "w", encoding="utf-8") as f:
            json.dump(dead_map, f, ensure_ascii=False)
    except Exception as e:
        print(f"Ölü kod önbelleği yazılamadı: {e}")


# ÖNEMLİ: isyatirimhisse kütüphanesi, bağlantı zaman aşımı (İş Yatırım
# sunucusuna hiç ulaşılamadığı geçici durumlar) ile "bu kod gerçekten
# hisse değil" durumunu AYNI genel "No data was fetched" mesajıyla
# bildiriyor — ikisini tek bir başarısızlıktan ayırt edemiyoruz. Bu yüzden
# bir kodu ancak FARKLI GÜNLERDE üst üste STRIKES_TO_BLACKLIST kez
# başarısız olursa "ölü" sayıyoruz; tek bir kötü gün (örn. İş Yatırım'ın
# genel kesintisi) hiçbir gerçek hisseyi yanlışlıkla dışlamaz.
STRIKES_TO_BLACKLIST = 3


def register_ticker_failure(ticker, dead_map):
    """Bir kodun bugün başarısız olduğunu kaydeder. Aynı gün içindeki
    tekrar denemeler tek strike sayılır (gün bazlı sayaç)."""
    today_str = datetime.now(TR_TZ).strftime("%Y-%m-%d")
    entry = dead_map.get(ticker, {"strikes": 0, "last_failed_date": None})
    if entry.get("last_failed_date") != today_str:
        entry["strikes"] = entry.get("strikes", 0) + 1
        entry["last_failed_date"] = today_str
        dead_map[ticker] = entry


def register_ticker_success(ticker, dead_map):
    """Bir kod başarılı veri döndürdüyse, önceki başarısızlık geçmişini sıfırla."""
    if ticker in dead_map:
        del dead_map[ticker]


def is_dead_ticker(ticker, dead_map):
    entry = dead_map.get(ticker)
    if not entry or entry.get("strikes", 0) < STRIKES_TO_BLACKLIST:
        return False
    try:
        marked = datetime.strptime(entry["last_failed_date"], "%Y-%m-%d").date()
        return (datetime.now(TR_TZ).date() - marked).days < DEAD_TICKER_RECHECK_DAYS
    except Exception:
        return False


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


LAST_ROUND1_SCORED_DF = None


def round1_screen(tickers):
    global LAST_ROUND1_SCORED_DF
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
    dead_map = load_dead_tickers()
    skipped_dead_count = 0

    print(f"Önbellekte {len(cache)} şirketin bilanço verisi var.")
    print(f"Ölü kod listesinde {len(dead_map)} kod var (gerçek hisse olmayan, atlanacak).")

    for i, tk in enumerate(tickers, 1):
        elapsed = time.time() - start_time
        if elapsed > ROUND1_TIME_BUDGET_SECONDS:
            print(f"Round 1 zaman bütçesini ({ROUND1_TIME_BUDGET_SECONDS}s) aştı, {i-1}/{total} hisseyle devam ediliyor.")
            break

        if is_dead_ticker(tk, dead_map):
            skipped_dead_count += 1
            continue

        price_data, definitely_no_data = fetch_price_snapshot(tk)
        if not price_data:
            register_ticker_failure(tk, dead_map)   # gün bazlı strike — tek kötü gün asla dışlamaya yetmez
            consecutive_failures += 1
            if consecutive_failures >= CONSECUTIVE_FAILURE_CIRCUIT_BREAKER:
                print(f"Üst üste {CONSECUTIVE_FAILURE_CIRCUIT_BREAKER} başarısız istek, veri kaynağı muhtemelen bu IP'yi engelledi. Duruyorum.")
                break
            time.sleep(random.uniform(IY_MIN_DELAY, IY_MAX_DELAY))
            continue

        consecutive_failures = 0
        register_ticker_success(tk, dead_map)   # başarılı oldu, önceki strike geçmişi varsa temizle

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
                  f"bilanço ağdan çekilen: {fetched_count}, ölü kod atlanan: {skipped_dead_count}, geçen süre: {elapsed:.0f}s)")

        time.sleep(random.uniform(IY_MIN_DELAY, IY_MAX_DELAY))

    save_fin_cache(cache)
    save_dead_tickers(dead_map)
    actually_dead = sum(1 for e in dead_map.values() if e.get("strikes", 0) >= STRIKES_TO_BLACKLIST)
    print(f"Bu çalıştırmada atlanan bilinen ölü kod sayısı: {skipped_dead_count}, "
          f"toplam kalıcı olarak ölü ilan edilen kod sayısı: {actually_dead} "
          f"(bir kod ancak {STRIKES_TO_BLACKLIST} farklı günde üst üste başarısız olursa kalıcı sayılır)")

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

    LAST_ROUND1_SCORED_DF = df.copy()
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
        print(f"  [RAG-DEBUG {ticker}] company_id={comp.company_id}")
        reports = comp.get_disclosures("FAR")
        print(f"  [RAG-DEBUG {ticker}] get_disclosures('FAR') -> {len(reports) if reports else 0} kayıt")
        if not reports:
            return None, None

        reports_sorted = sorted(reports, key=lambda r: r.get("publishDate", ""), reverse=True)
        latest = reports_sorted[0]
        disc_index = latest.get("disclosureIndex")
        print(f"  [RAG-DEBUG {ticker}] en güncel rapor: disclosureIndex={disc_index}, publishDate={latest.get('publishDate')}")
        if not disc_index:
            return None, None

        announcement_url = f"https://www.kap.org.tr/tr/Bildirim/{disc_index}"
        browser_ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        resp = requests.get(announcement_url, timeout=30, headers={"User-Agent": browser_ua})
        resp.raise_for_status()

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html5lib")
        pdf_link_tag = soup.select("a.modal-attachment.type-xsmall.bi-sky-black.maximize")
        href = None
        if pdf_link_tag and pdf_link_tag[0].get("href"):
            href = pdf_link_tag[0]["href"]
        else:
            # Yedek: spesifik CSS class bulunamadıysa, sayfadaki .pdf ile
            # biten herhangi bir bağlantıyı ara (KAP'ın sayfa yapısı
            # bildirim tarihine/tipine göre değişebiliyor).
            fallback = soup.find("a", href=re.compile(r"\.pdf($|\?)", re.IGNORECASE))
            if fallback and fallback.get("href"):
                href = fallback["href"]
                print(f"  [RAG-DEBUG {ticker}] Ana seçici başarısız, yedek .pdf linki bulundu")

        print(f"  [RAG-DEBUG {ticker}] PDF link bulundu mu: {bool(href)}")
        if not href:
            has_modal_class = "modal-attachment" in resp.text
            pdf_mentions = resp.text.lower().count(".pdf")
            print(f"  [RAG-DEBUG {ticker}] Sayfa uzunluğu={len(resp.text)}, "
                  f"'modal-attachment' geçiyor mu={has_modal_class}, sayfada '.pdf' geçme sayısı={pdf_mentions}")
            return None, latest

        pdf_url = href if href.startswith("http") else ("https://www.kap.org.tr" + href)
        pdf_resp = requests.get(pdf_url, timeout=60)
        content_type = pdf_resp.headers.get("Content-Type", "").lower()
        print(f"  [RAG-DEBUG {ticker}] PDF indirme: status={pdf_resp.status_code}, content-type={content_type}, boyut={len(pdf_resp.content)} byte")
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
    """Önbellekte taze bir alıntı varsa onu döner, yoksa KAP'tan yeni
    faaliyet raporunu çeker, anahtar kelime filtrelemesi yapar ve önbelleğe
    yazar. Rapor bulunamazsa None döner — bu durumda Bull/Bear ajanları bu
    bölümü N/A olarak görür, uydurmaz.

    Not: Başarılı sonuçlar REPORT_CACHE_MAX_AGE_DAYS (60 gün) önbelleklenir,
    ama 'bulunamadı' sonuçları sadece REPORT_NOT_FOUND_RETRY_DAYS (3 gün)
    önbelleklenir — bu, geçici bir hata/bug varsa sistemin uzun süre
    "poison" olmasını engelliyor, gerçek RAG hit-rate'i hâlâ öğreniyoruz."""
    today = datetime.now(TR_TZ).date()
    entry = cache.get(ticker)
    if entry and entry.get("fetched_at"):
        try:
            fetched = datetime.strptime(entry["fetched_at"], "%Y-%m-%d").date()
            age_days = (today - fetched).days
            max_age = REPORT_CACHE_MAX_AGE_DAYS if entry.get("excerpt") else REPORT_NOT_FOUND_RETRY_DAYS
            if age_days < max_age:
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

KART METNİ KURALI:
- key_risks içindeki her madde TEK BAŞINA okunduğunda anlaşılabilen TAM bir cümle olsun.
- "...", yarım ifade, başlık dili veya devamı varmış hissi veren cümle kullanma.
- Her madde yaklaşık 90-150 karakter olsun; mümkünse somut oran/veri + bunun neden risk olduğu birlikte yazılsın.
- Aynı bilgiyi iki maddede tekrar etme.

{data_block}

Yalnızca aşağıdaki JSON formatında yanıt ver, başka açıklama ekleme:
{{
  "bear_case": "<2-3 cümlelik Türkçe, en kötü senaryo argümanı>",
  "key_risks": ["<90-150 karakterlik, tek başına anlamlı TAM Türkçe cümle; mümkünse somut oran/veri içersin>", "<90-150 karakterlik ikinci TAM cümle>"],
  "value_trap_risk": <0-100 arası tam sayı, bu şirketin bir değer tuzağı olma ihtimali>,
  "bear_fv": <TL cinsinden kötümser adil değer, sayı>
}}
"""

BULL_PROMPT_TEMPLATE = """
Sen bir BIST Boğa (Bull) Analistisin. Görevin şirket için EN İYİ MAKUL
senaryoyu kurmaktır; ancak iyimser olmak serbest tahmin yapmak anlamına gelmez.
Bull-case mutlaka verilen finansal veriye ve ölçülebilir/izlenebilir katalizörlere
dayanmalıdır.

ÖNEMLİ DEĞERLEME KURALI:
- Sırf hisse fiyatı yükseldi diye Bull FV yükseltme.
- Bir haber/katalizörün yalnızca var olması Bull FV üretmek için yeterli değildir.
- Bull FV ancak katalizörün gelir, kâr, marj, nakit akışı, bilanço, kapasite,
  sipariş/backlog veya değerleme çarpanına etkisi makul şekilde modellenebiliyorsa
  verilebilir.
- Destek zayıfsa veya yalnızca hikâye düzeyindeyse bull_fv=null döndür.
- Bull FV, "fiyatın gidebileceği yer" değil; şartları açıkça tanımlanmış destekli
  bir iyimser adil değer senaryosudur.

Bilmediğin/verilmeyen bilgiyi UYDURMA, "N/A" ise o konuda yorum yapma.

KART METNİ KURALI:
- catalysts içindeki her madde TEK BAŞINA okunduğunda anlaşılabilen TAM bir cümle olsun.
- "...", yarım ifade veya başlık dili kullanma.
- Her madde yaklaşık 90-150 karakter olsun; mümkünse somut veri ile bunun şirket açısından neden önemli olduğunu aynı cümlede ver.
- Aynı bilgiyi iki maddede tekrar etme.

{data_block}

Yalnızca aşağıdaki JSON formatında yanıt ver, başka açıklama ekleme:
{{
  "bull_case": "<2-3 cümlelik Türkçe, destekli iyimser senaryo>",
  "catalysts": ["<90-150 karakterlik, tek başına anlamlı TAM Türkçe cümle; veri + analitik anlam birlikte>", "<90-150 karakterlik ikinci TAM cümle>"],
  "growth_conviction": <0-100 arası tam sayı>,
  "bull_fv_supported": <true | false>,
  "bull_fv_evidence": ["<ölçülebilir destek 1>", "<ölçülebilir destek 2>"],
  "bull_fv": <destek varsa TL sayı, destek yoksa null>
}}
"""

CRO_PROMPT_TEMPLATE = """
Sen bir Chief Risk Officer'sın (Yönetici Risk Sorumlusu). Aynı şirket için
Ayı ve Boğa analistlerinin zıt argümanlarını tarafsız biçimde tart.

SİSTEMİN ANA AMACI:
İyi şirketleri değil, FUNDAMENTAL OLARAK HÂLÂ FİYATLANMAMIŞ fırsatları bulmak.
Teknik görünüm daha sonra sıralamaya yardımcı olur; değerleme açığı yaratamaz.

DEĞERLEME DİSİPLİNİ — KESİN KURALLAR:
1) Fiyatın eski Base FV'ye yaklaşması/ulaşması, Base FV'yi yükseltmek için ASLA
   gerekçe değildir.
2) Base FV yalnızca yeni ve maddi fundamental kanıtla yukarı revize edilebilir:
   ör. raporlanan gelir/kâr gücü, marj genişlemesi, nakit akışı kalitesi,
   borç azalması, kapasite/siparişin finansallara ölçülebilir katkısı veya
   güvenilir ve nicel yeni guidance.
3) Katalizörün duyurulmuş olması tek başına Base revizyonu değildir. Etki henüz
   gerçekleşmemiş ama ölçülebilir ise yalnızca Upside/Bull senaryosunda kalmalıdır.
4) Önceki taramadaki aynı fundamental veri setiyle yeni günde Base FV'yi tekrar
   yukarı taşıma. Base revizyonu için önceki analize göre YENİ kanıt gerekir.
5) Upside/Bull FV ancak açık, ölçülebilir ve veriye dayanan bull-case desteği varsa
   gösterilebilir. Destek yoksa bull_fv=null olmalıdır.
6) Bull FV gelecekteki otomatik yeni Base değildir. Bull senaryosunun şartları
   gerçekleşmeye başlarsa ancak o zaman yeni analizde Base'e kısmen/tamamen
   taşınabilir.
7) Güncel fiyat Base FV'yi fiyatlamışsa ve yeni fundamental kanıt yoksa Base'i
   koru; sistem şirketi fırsat listesinden çıkaracaktır.
8) Alpha puanında şirket kalitesi ile mevcut fiyatta kalan değerleme alanını
   ayır. İyi şirket ile iyi fırsat aynı şey değildir.

ÖNCEKİ ANALİZ HAFIZASI:
{previous_valuation_block}

GÜNCEL VERİ:
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
Bull FV desteği: {bull_fv_supported}
Bull FV kanıtı: {bull_fv_evidence}
Bull FV: {bull_fv}

KAMUYA AÇIK KART DİLİ:
- Final catalysts ve risks maddelerini yeniden yaz; Bear/Bull çıktısını ham haliyle kopyalamak zorunda değilsin.
- Her madde tam cümle olsun ve tek başına okunduğunda ne anlatıldığı anlaşılsın.
- Bir cümlede mümkünse önce somut bulguyu, sonra analitik anlamını ver.
- 90-150 karakter hedefle; kesinlikle üç nokta (...) veya yarım bırakılmış ifade üretme.
- Yatırım eylemi önerme; "al/sat" dili kullanma.

Yalnızca aşağıdaki JSON formatında yanıt ver, başka açıklama ekleme:
{{
  "alpha_score": <0-100 arası tam sayı>,
  "data_confidence": <0-100 arası tam sayı>,
  "bear_fv": <TL sayı>,
  "base_fv": <TL sayı>,
  "bull_fv": <destekli ise TL sayı, aksi halde null>,
  "base_revision_status": "UNCHANGED | SUPPORTED_BY_NEW_FUNDAMENTALS | LOWERED",
  "base_revision_evidence": ["<önceki analize göre yeni ve maddi kanıtlar>"],
  "upside_case_status": "SUPPORTED | UNSUPPORTED",
  "upside_evidence": ["<Bull FV'yi destekleyen ölçülebilir kanıtlar>"],
  "thesis_summary": "<2-3 cümlelik dengeli sentez>",
  "catalysts": ["<90-150 karakterlik TAM ve bağımsız analiz cümlesi>", "<gerekirse ikinci TAM cümle>"],
  "risks": ["<90-150 karakterlik TAM ve bağımsız risk cümlesi>", "<gerekirse ikinci TAM cümle>"],
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


def _previous_valuation_block(prev):
    if not prev:
        return "Önceki analiz yok; ilk değerleme. Fiyatı referans alarak FV üretme."
    fields = [
        ("Önceki fiyat", "price"), ("Önceki Base FV", "base_fv"),
        ("Önceki Bull FV", "bull_fv"), ("Önceki Alpha", "alpha_score"),
        ("Ciro büyümesi", "revenue_growth"), ("Kâr büyümesi", "earnings_growth"),
        ("Ciro ivmesi", "revenue_acceleration"), ("Kâr ivmesi", "earnings_acceleration"),
        ("Brüt marj trendi", "gross_margin_trend"), ("Faaliyet marj trendi", "operating_margin_trend"),
        ("ROE", "roe"), ("Net borç/özkaynak", "net_debt_to_equity"),
        ("CFO/Net kâr", "cfo_to_net_income"),
    ]
    return "\n".join(f"- {label}: {prev.get(key, 'N/A')}" for label, key in fields)


def _material_fundamental_change(candidate, prev):
    """Aynı veriyle her gün Base FV yukarı ratchet edilmesini engeller.
    Önceki state'teki temel metriklerle güncel aday arasında anlamlı fark arar.
    İlk analizde (hafıza yoksa) True döner; sonraki analizlerde yeni kanıt gerekir.
    """
    if not prev:
        return True, ["ilk değerleme"]

    # V5.12'ye geçişte eski state kayıtlarında aşağıdaki temel metrikler yoksa
    # bir defaya mahsus baz çizgisi kurulmasına izin ver. Sonraki taramalarda
    # bu metrikler state'e kaydedildiği için aynı veriyle ratchet yapılamaz.
    tracked_keys = [
        "revenue_growth", "earnings_growth", "revenue_acceleration",
        "earnings_acceleration", "gross_margin_trend", "operating_margin_trend",
        "roe", "net_debt_to_equity", "cfo_to_net_income",
    ]
    if not any(_safe_float(prev.get(k)) is not None for k in tracked_keys):
        return True, ["önceki state'te fundamental baz çizgisi yok; V5.12 geçiş kaydı"]

    checks = [
        ("revenue_growth", 0.05, "ciro büyümesi"),
        ("earnings_growth", 0.08, "kâr büyümesi"),
        ("revenue_acceleration", 0.05, "ciro ivmesi"),
        ("earnings_acceleration", 0.08, "kâr ivmesi"),
        ("gross_margin_trend", 0.01, "brüt marj trendi"),
        ("operating_margin_trend", 0.01, "faaliyet marj trendi"),
        ("roe", 0.02, "ROE"),
        ("net_debt_to_equity", 0.10, "net borç/özkaynak"),
        ("cfo_to_net_income", 0.20, "nakit akışı kalitesi"),
    ]
    changes = []
    for key, threshold, label in checks:
        cur = _safe_float(candidate.get(key))
        old = _safe_float(prev.get(key))
        if cur is not None and old is not None and abs(cur - old) >= threshold:
            changes.append(f"{label}: {old:.3f} -> {cur:.3f}")
    return bool(changes), changes


def _validate_valuation_output(cro, candidate, prev, ticker, bull=None):
    """LLM çıktısına deterministik değerleme disiplini uygular."""
    prev_base = _safe_float((prev or {}).get("base_fv"))
    new_base = _safe_float(cro.get("base_fv"))
    status = str(cro.get("base_revision_status", "UNCHANGED")).upper()
    material_change, changes = _material_fundamental_change(candidate, prev)

    if prev_base and new_base and new_base > prev_base * 1.03:
        supported = status == "SUPPORTED_BY_NEW_FUNDAMENTALS" and material_change
        if not supported:
            print(
                f"  [FV-DİSİPLİN {ticker}] Base artışı reddedildi: "
                f"{prev_base:.2f} -> {new_base:.2f}. Yeni maddi fundamental kanıt yok."
            )
            cro["base_fv"] = prev_base
            cro["base_revision_status"] = "UNCHANGED"
            cro["base_revision_evidence"] = []
        else:
            print(
                f"  [FV-DİSİPLİN {ticker}] Base revizyonu kabul: "
                f"{prev_base:.2f} -> {new_base:.2f} | yeni kanıt: {'; '.join(changes)}"
            )

    # Upside sadece destekli bull-case varsa görünür.
    upside_status = str(cro.get("upside_case_status", "UNSUPPORTED")).upper()
    evidence = cro.get("upside_evidence") or []
    bull_agent_supported = bool((bull or {}).get("bull_fv_supported"))
    bull_agent_evidence = (bull or {}).get("bull_fv_evidence") or []
    bull_fv = _safe_float(cro.get("bull_fv"))
    base_fv = _safe_float(cro.get("base_fv"))
    if (
        upside_status != "SUPPORTED"
        or not evidence
        or not bull_agent_supported
        or not bull_agent_evidence
        or bull_fv is None
        or (base_fv and bull_fv <= base_fv)
    ):
        if bull_fv is not None:
            print(f"  [FV-DİSİPLİN {ticker}] Upside FV kaldırıldı: destekli/ölçülebilir bull-case doğrulanmadı.")
        cro["bull_fv"] = None
        cro["upside_case_status"] = "UNSUPPORTED"
        cro["upside_evidence"] = []

    cro["fundamental_change_detected"] = material_change
    cro["fundamental_change_evidence"] = changes
    return cro


def analyze_with_gemini(candidate, prev_snapshot=None):
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
        previous_valuation_block=_previous_valuation_block(prev_snapshot),
        bull_case=bull.get("bull_case", "N/A"),
        catalysts=", ".join(bull.get("catalysts", []) or []) or "N/A",
        growth_conviction=bull.get("growth_conviction", "N/A"),
        bull_fv_supported=bull.get("bull_fv_supported", False),
        bull_fv_evidence="; ".join(bull.get("bull_fv_evidence", []) or []) or "N/A",
        bull_fv=bull.get("bull_fv", "N/A"),
    )
    cro = _call_gemini_json(cro_prompt, f"{ticker}-CRO")
    if cro is None:
        return None

    cro = _validate_valuation_output(cro, candidate, prev_snapshot, ticker, bull=bull)

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


# --------------------------------------------------------------------------
# TEKNİK ANALİZ KATMANI (kullanıcı fikri, kurumsal düzey Volume Profile
# tabanlı sinyaller) — SADECE Round 1'in 22 finalistine uygulanır.
# Fundamental skor (Bear/Bull/CRO'dan gelen alpha_score) %70, bu katman %30
# ağırlıkla birleştirilip TOP 10 seçimi/sıralaması bu birleşik skora göre
# yapılır. Round 1'in 22 aday seçimini VE Round 2'nin analizini ETKİLEMEZ.
# --------------------------------------------------------------------------
TECH_FUNDAMENTAL_WEIGHT = 0.70
TECH_TECHNICAL_WEIGHT = 0.30

# --------------------------------------------------------------------------
# FUNDAMENTAL QUALITY GUARDRAIL (v3)
# --------------------------------------------------------------------------
# Teknik katman bir "aday yaratma" motoru değildir; yalnızca fundamental
# olarak yeterli kaliteyi geçen adayların kendi aralarındaki sırasını etkiler.
# Böylece çok yüksek teknik skor, düşük fundamental alpha / çok düşük kalan
# fundamental upside / WEAKENING gibi durumları maskeleyemez.
MIN_ACTIVE_ALPHA = 55.0
MIN_ACTIVE_BASE_UPSIDE_PCT = 5.0
MIN_ACTIVE_DATA_CONFIDENCE = 60.0
BLOCKED_ACTIVE_VERDICTS = {"WEAKENING"}

# TOP 3 için daha yüksek standart: güçlü fundamental görüş + anlamlı kalan
# upside + yeterli veri güveni gerekir. WATCH bir şirket, teknik skoru çok
# yüksek olsa bile TOP 3'e taşınmaz.
TOP3_MIN_ALPHA = 65.0
TOP3_MIN_BASE_UPSIDE_PCT = 8.0
TOP3_MIN_DATA_CONFIDENCE = 70.0
TOP3_ALLOWED_VERDICTS = {"HIGH CONVICTION", "ATTRACTIVE"}

# %30'luk teknik payın kendi içindeki dağılımı (toplam = 1.0)
# İlk 6'sı kullanıcının fikirleri, son 3'ü (sector_rs, foreign_interest,
# avwap_52wk_high) Claude'un önerdiği ek kurumsal-düzey sinyaller.
TECH_SUB_WEIGHTS = {
    "weekly_vp_trend": 0.18,      # Haftalık Volume Profile dizilimi (POC/VA yukarı mı kayıyor)
    "weekly_vp_level": 0.18,      # Fiyatın haftalık POC/VAH/VAL seviyelerine göre konumu
    "sector_rs": 0.15,            # XU100'e göre göreceli güç (20/60 gün)
    "anchored_vp": 0.13,          # Bilanço gününden itibaren Anchored VWAP'a göre konum
    "foreign_interest": 0.12,     # Yabancı oranı (evrene göre yüzdelik dilim + kendi geçmişimize göre değişim)
    "daily_pbd_shape": 0.10,      # Son 3 günün P/b/D hacim dağılım yapısı
    "avwap_52wk_high": 0.08,      # 52 haftanın zirvesinden Anchored VWAP'a göre konum
    "monthly_open": 0.04,         # Fiyat aylık açılışın üzerinde mi altında mı
    "delta_obv": 0.02,            # Fiyat yönüne göre hacim (OBV mantığı) — gerçek bid/ask yok
}
# Her bileşenin teorik maksimum mutlak puanı (normalize etmek için)
TECH_SUB_MAX_ABS = {
    "weekly_vp_trend": 2.0,
    "weekly_vp_level": 2.0,
    "sector_rs": 2.0,
    "anchored_vp": 1.0,
    "foreign_interest": 1.0,
    "daily_pbd_shape": 1.0,
    "avwap_52wk_high": 1.0,
    "monthly_open": 1.0,
    "delta_obv": 1.0,
}


def _pick_col_ci(df, candidates):
    """Sütun adını büyük/küçük harf duyarsız arar (borsapy'nin sütun isimleri
    sürüm/varlık tipine göre değişebilir: Close/close, Volume/volume vb.)."""
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


# --- Bölüm A: mevcut günlük İş Yatırım verisiyle hesaplanabilenler --------
def fetch_daily_series_isy(ticker, lookback_days=400):
    """Teknik katman için uzunca bir günlük fiyat/ciro serisi çeker (Anchored
    VWAP, aylık açılış ve OBV-tarzı Delta hesaplamaları için). Round 1'deki
    kısa vadeli seriden bağımsız, ayrı bir çekim (finalistler için ek maliyet
    küçük, ~22 istek)."""
    end_date = datetime.now(TR_TZ).strftime("%d-%m-%Y")
    start_date = (datetime.now(TR_TZ) - timedelta(days=lookback_days)).strftime("%d-%m-%Y")
    try:
        df = isy_fetch_stock_data(symbols=ticker, start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            return None
        close_col = _pick_column(df, ["HGDG_KAPANIS", "KAPANIS", "CLOSING_TL", "CLOSING"])
        date_col = _pick_column(df, ["HGDG_TARIH", "TARIH"])
        vol_try_col = _pick_column(df, ["HGDG_HACIM_TL", "HACIM_TL", "VOLUME_TL"])
        vol_lot_col = _pick_column(df, ["HGDG_HACIM_LOT", "HACIM_LOT", "VOLUME_LOT", "HGDG_HACIM"])
        if close_col is None or date_col is None:
            return None

        out = pd.DataFrame({
            "date": pd.to_datetime(df[date_col], errors="coerce"),
            "close": pd.to_numeric(df[close_col], errors="coerce"),
        })
        if vol_try_col is not None:
            out["turnover"] = pd.to_numeric(df[vol_try_col], errors="coerce")
        elif vol_lot_col is not None:
            out["turnover"] = pd.to_numeric(df[vol_lot_col], errors="coerce") * out["close"]
        else:
            out["turnover"] = None

        out = out.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
        return out if not out.empty else None
    except Exception as e:
        print(f"  [TEKNİK] {ticker}: günlük seri alınamadı ({e})")
        return None


def _most_recent_quarter_end(today):
    """En yakın (bugünden önceki/eşit) çeyrek sonu tarihi — gerçek KAP bilanço
    açıklama tarihi yerine kullanılan YAKLAŞIK bir 'bilanço günü' referansı.
    (Faz 5/RAG kapalı olduğu için gerçek açıklama tarihine erişimimiz yok.)"""
    quarter_ends = [(3, 31), (6, 30), (9, 30), (12, 31)]
    candidates = []
    for y in (today.year, today.year - 1):
        for m, d in quarter_ends:
            candidates.append(datetime(y, m, d).date())
    candidates = [c for c in candidates if c <= today]
    return max(candidates) if candidates else None


def compute_anchored_vwap_signal(daily_df, current_price, today):
    """Bilanço gününden (yaklaşık) itibaren Anchored VWAP hesaplar, fiyat
    üzerindeyse +1, altındaysa -1 döner. Veri yetersizse None (nötr)."""
    if daily_df is None or daily_df.empty or not current_price:
        return None
    anchor = _most_recent_quarter_end(today)
    if anchor is None:
        return None
    sub = daily_df[(daily_df["date"].dt.date >= anchor) & daily_df["turnover"].notna() & (daily_df["close"] > 0)]
    if len(sub) < 3:
        return None
    volume_approx = sub["turnover"] / sub["close"]
    total_vol = volume_approx.sum()
    if total_vol <= 0:
        return None
    avwap = sub["turnover"].sum() / total_vol
    if avwap <= 0:
        return None
    return 1.0 if current_price >= avwap else -1.0


def compute_monthly_open_signal(daily_df, current_price, today):
    """Fiyat, bu ayın ilk işlem gününün kapanışına göre üstte mi altta mı."""
    if daily_df is None or daily_df.empty or not current_price:
        return None
    month_start = today.replace(day=1)
    sub = daily_df[daily_df["date"].dt.date >= month_start]
    if sub.empty:
        return None
    monthly_ref = sub.iloc[0]["close"]
    if not monthly_ref or monthly_ref <= 0:
        return None
    return 1.0 if current_price >= monthly_ref else -1.0


def compute_obv_delta_signal(daily_df, lookback=20):
    """OBV mantığıyla (gerçek bid/ask yok, fiyat yönüne göre hacmi +/- sayarak)
    son `lookback` günün net akışını hesaplar."""
    if daily_df is None or len(daily_df) < 6:
        return None
    sub = daily_df.tail(lookback + 1).copy()
    sub["price_change"] = sub["close"].diff()
    sub = sub.dropna(subset=["price_change", "turnover"])
    if sub.empty:
        return None
    sub["volume_approx"] = sub["turnover"] / sub["close"]
    sub["signed_vol"] = np.where(
        sub["price_change"] > 0, sub["volume_approx"],
        np.where(sub["price_change"] < 0, -sub["volume_approx"], 0.0)
    )
    net = sub["signed_vol"].sum()
    if net == 0:
        return 0.0
    return 1.0 if net > 0 else -1.0


def _generic_return(series_dates_values, n_days):
    """Bir (tarih, değer) serisinde n_days işlem günü öncesine göre getiri."""
    if series_dates_values is None or len(series_dates_values) <= n_days:
        return None
    last = series_dates_values.iloc[-1]
    past = series_dates_values.iloc[-(n_days + 1)]
    if past and past > 0:
        return (last / past) - 1
    return None


_xu100_cache = None


def _get_xu100_series():
    """XU100 endeksinin günlük seviyesini bir kez çekip run boyunca önbellekte
    tutar (22 finalist için 22 kez çekmeye gerek yok)."""
    global _xu100_cache
    if _xu100_cache is not None:
        return _xu100_cache
    end_date = datetime.now(TR_TZ).strftime("%d-%m-%Y")
    start_date = (datetime.now(TR_TZ) - timedelta(days=120)).strftime("%d-%m-%Y")
    try:
        df = isy_fetch_index_data(indices="XU100", start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            _xu100_cache = pd.DataFrame()
            return _xu100_cache
        df = df.sort_values("DATE").reset_index(drop=True)
        print(f"  [TEKNİK-DEBUG XU100] {len(df)} günlük endeks verisi çekildi")
        _xu100_cache = df
        return df
    except Exception as e:
        print(f"  [TEKNİK] XU100 endeks verisi alınamadı (Sektör RS sinyali devre dışı): {e}")
        _xu100_cache = pd.DataFrame()
        return _xu100_cache


def compute_sector_rs_signal(return_20d, return_60d):
    """Hissenin kendi 20/60 günlük getirisini XU100'ün aynı dönemdeki
    getirisiyle karşılaştırır. Her iki dönemde de endeksi geçerse +2,
    ikisinde de geride kalırsa -2. (Sektör endeksleri yerine XU100
    kullanıyoruz — daha basit ve güvenilir bir referans noktası.)"""
    xu100 = _get_xu100_series()
    if xu100 is None or xu100.empty or return_20d is None or return_60d is None:
        return None
    bench_20d = _generic_return(xu100["VALUE"], 20)
    bench_60d = _generic_return(xu100["VALUE"], 60)
    if bench_20d is None or bench_60d is None:
        return None
    score = 0.0
    score += 1.0 if return_20d > bench_20d else -1.0
    score += 1.0 if return_60d > bench_60d else -1.0
    return score


FOREIGN_RATIO_HISTORY_FILE = os.path.join(os.path.dirname(__file__), "foreign_ratio_history.json")
_screener_cache = None
_foreign_ratio_history_cache = None


def load_foreign_ratio_history():
    global _foreign_ratio_history_cache
    if _foreign_ratio_history_cache is not None:
        return _foreign_ratio_history_cache
    if os.path.exists(FOREIGN_RATIO_HISTORY_FILE):
        try:
            with open(FOREIGN_RATIO_HISTORY_FILE, "r", encoding="utf-8") as f:
                _foreign_ratio_history_cache = json.load(f)
                return _foreign_ratio_history_cache
        except Exception:
            pass
    _foreign_ratio_history_cache = {}
    return _foreign_ratio_history_cache


def save_foreign_ratio_history():
    if _foreign_ratio_history_cache is None:
        return
    try:
        with open(FOREIGN_RATIO_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(_foreign_ratio_history_cache, f, ensure_ascii=False)
    except Exception as e:
        print(f"Yabancı oranı geçmişi yazılamadı: {e}")


def _get_full_screener_df():
    """borsapy'nin İş Yatırım tabanlı screener'ından TÜM BIST evrenini bir
    kez çeker (yabancı oranı dahil), run boyunca önbellekte tutar."""
    global _screener_cache
    if _screener_cache is not None:
        return _screener_cache
    if not TECHNICAL_LAYER_AVAILABLE:
        _screener_cache = pd.DataFrame()
        return _screener_cache
    try:
        df = bpy.screen_stocks(market_cap_min=0)
        if df is None or df.empty:
            print("  [TEKNİK-DEBUG] Screener boş sonuç döndü")
            _screener_cache = pd.DataFrame()
        else:
            print(f"  [TEKNİK-DEBUG] Screener {len(df)} şirket döndürdü, sütunlar: {df.columns.tolist()}")
            _screener_cache = df
        return _screener_cache
    except Exception as e:
        print(f"  [TEKNİK] Screener (yabancı ilgisi sinyali) alınamadı: {e}")
        _screener_cache = pd.DataFrame()
        return _screener_cache


def compute_foreign_interest_signal(ticker):
    """Yabancı ilgisi sinyalini doğrudan borsapy Ticker.fast_info içindeki
    ``foreign_ratio`` alanından okur ve kendi günlük geçmişimizle kıyaslar.

    ``screen_stocks(market_cap_min=0)`` yalnızca kullanılan screener kriterinin
    kolonunu döndürebildiği için yabancı oranı kolonu garanti değildi. İlk
    başarılı gün oran sadece kaydedilir ve sinyal N/A olur. Sonraki günlerden
    itibaren oran artıyorsa +1, düşüyorsa -1, anlamlı değişmediyse 0 üretir.
    """
    if not TECHNICAL_LAYER_AVAILABLE:
        return None
    try:
        stock = bpy.Ticker(ticker)
        info = stock.fast_info
        current_ratio = info.get("foreign_ratio") if hasattr(info, "get") else info["foreign_ratio"]
        current_ratio = float(current_ratio)
        if pd.isna(current_ratio):
            return None
    except Exception as e:
        print(f"  [TEKNİK-DEBUG {ticker}] yabancı oranı alınamadı ({e})")
        return None

    today_str = datetime.now(TR_TZ).strftime("%Y-%m-%d")
    history = load_foreign_ratio_history()
    entries = history.get(ticker, [])
    previous_entries = [e for e in entries if e.get("date") != today_str]
    prev = previous_entries[-1] if previous_entries else None
    history[ticker] = (previous_entries + [{"date": today_str, "ratio": current_ratio}])[-40:]

    if prev is None:
        print(f"  [TEKNİK-DEBUG {ticker}] yabancı oranı={current_ratio:.4f}; geçmiş yok, bugün sadece kaydedildi")
        return None
    try:
        prev_ratio = float(prev["ratio"])
    except Exception:
        return None

    eps = max(abs(prev_ratio) * 0.001, 0.01 if max(abs(prev_ratio), abs(current_ratio)) > 1 else 0.0001)
    delta = current_ratio - prev_ratio
    score = 0.0 if abs(delta) <= eps else (1.0 if delta > 0 else -1.0)
    print(f"  [TEKNİK-DEBUG {ticker}] yabancı oranı {prev_ratio:.4f} → {current_ratio:.4f} (Δ={delta:+.4f}), sinyal={score:+.0f}")
    return score

def compute_avwap_52wk_high_signal(daily_df, current_price):
    """Son 52 haftanın en yüksek kapanışından itibaren Anchored VWAP
    hesaplar. Fiyat bunun üzerindeyse o zirvede alım yapanlar artık kârda
    demektir (+1); altındaysa hâlâ 'hapsolmuş arz' baskısı sürüyor (-1)."""
    if daily_df is None or len(daily_df) < 30 or not current_price:
        return None
    lookback = daily_df.tail(252)   # ~52 hafta işlem günü
    if lookback.empty:
        return None
    peak_idx = lookback["close"].idxmax()
    peak_date = lookback.loc[peak_idx, "date"]

    sub = daily_df[(daily_df["date"] >= peak_date) & daily_df["turnover"].notna() & (daily_df["close"] > 0)]
    if len(sub) < 3:
        return None
    volume_approx = sub["turnover"] / sub["close"]
    total_vol = volume_approx.sum()
    if total_vol <= 0:
        return None
    avwap = sub["turnover"].sum() / total_vol
    if avwap <= 0:
        return None
    return 1.0 if current_price >= avwap else -1.0


# --- Bölüm B: borsapy (TradingView) intraday veri gerektirenler -----------
def fetch_intraday_bars(ticker, period="1mo", interval="15m"):
    """borsapy ile intraday mum verisi çeker. CANLI TEST EDİLEMEDİ (bu
    sandbox'ın TradingView'a ağ erişimi yok) — bu yüzden bol teşhis logu
    içeriyor ve her adımda hata olursa None döner, sistem çökmez."""
    if not TECHNICAL_LAYER_AVAILABLE:
        return None
    try:
        stock = bpy.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        if df is None or df.empty:
            print(f"  [TEKNİK-DEBUG {ticker}] intraday veri boş döndü (period={period}, interval={interval})")
            return None

        high_col = _pick_col_ci(df, ["High", "high"])
        low_col = _pick_col_ci(df, ["Low", "low"])
        close_col = _pick_col_ci(df, ["Close", "close"])
        vol_col = _pick_col_ci(df, ["Volume", "volume"])
        if not all([high_col, low_col, close_col, vol_col]):
            print(f"  [TEKNİK-DEBUG {ticker}] beklenen sütunlar bulunamadı: {df.columns.tolist()}")
            return None

        out = df.rename(columns={high_col: "high", low_col: "low", close_col: "close", vol_col: "volume"})
        # Tarih/saat index'ini yakalamaya çalış (farklı sürümlerde index ya da sütun olabilir)
        if isinstance(out.index, pd.DatetimeIndex):
            out = out.reset_index().rename(columns={out.index.name or "index": "datetime"})
        else:
            dt_col = _pick_col_ci(out, ["Date", "Datetime", "date", "datetime"])
            if dt_col:
                out = out.rename(columns={dt_col: "datetime"})
                out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
            else:
                print(f"  [TEKNİK-DEBUG {ticker}] tarih/saat sütunu bulunamadı: {df.columns.tolist()}")
                return None

        out = out.dropna(subset=["datetime", "high", "low", "close", "volume"])
        return out if not out.empty else None
    except Exception as e:
        print(f"  [TEKNİK] {ticker}: intraday veri çekilemedi ({e})")
        return None


def _compute_volume_profile(bars_df, n_bins=40, value_area_pct=0.70):
    """Intraday barlardan basit bir Volume Profile hesaplar: her barın hacmini
    tipik fiyatına ((H+L+C)/3) göre bir fiyat aralığına (bin) atar. Gerçek
    tick-level POC kadar hassas değil ama standart bir yaklaşıklıktır."""
    if bars_df is None or bars_df.empty:
        return None
    price_min = bars_df["low"].min()
    price_max = bars_df["high"].max()
    if pd.isna(price_min) or pd.isna(price_max) or price_max <= price_min:
        return None

    bins = np.linspace(price_min, price_max, n_bins + 1)
    typical = (bars_df["high"] + bars_df["low"] + bars_df["close"]) / 3
    bin_idx = np.clip(np.digitize(typical, bins) - 1, 0, n_bins - 1)

    vol_by_bin = np.zeros(n_bins)
    for b, v in zip(bin_idx, bars_df["volume"]):
        if pd.notna(v):
            vol_by_bin[b] += v

    total_vol = vol_by_bin.sum()
    if total_vol <= 0:
        return None

    poc_idx = int(np.argmax(vol_by_bin))
    poc_price = (bins[poc_idx] + bins[poc_idx + 1]) / 2

    target = total_vol * value_area_pct
    lo, hi = poc_idx, poc_idx
    cur_vol = vol_by_bin[poc_idx]
    while cur_vol < target and (lo > 0 or hi < n_bins - 1):
        left_vol = vol_by_bin[lo - 1] if lo > 0 else -1
        right_vol = vol_by_bin[hi + 1] if hi < n_bins - 1 else -1
        if right_vol >= left_vol:
            hi += 1
            cur_vol += vol_by_bin[hi]
        else:
            lo -= 1
            cur_vol += vol_by_bin[lo]

    return {"poc": poc_price, "vah": bins[hi + 1], "val": bins[lo]}


def compute_weekly_vp_signals(ticker, current_price):
    """Son 3 TAMAMLANMIŞ hafta için haftalık Volume Profile hesaplar.
    Döndürür: (weekly_vp_trend_score, weekly_vp_level_score) — ikisi de None
    olabilir (veri yoksa/yetersizse)."""
    bars = fetch_intraday_bars(ticker, period="1mo", interval="15m")
    if bars is None or len(bars) < 30:
        return None, None

    bars = bars.copy()
    bars["iso_year"] = bars["datetime"].dt.isocalendar().year
    bars["iso_week"] = bars["datetime"].dt.isocalendar().week
    bars["week_key"] = list(zip(bars["iso_year"], bars["iso_week"]))

    today = datetime.now(TR_TZ).date()
    current_iso = today.isocalendar()
    current_week_key = (current_iso[0], current_iso[1])

    week_keys_sorted = sorted(set(bars["week_key"]) - {current_week_key})
    if len(week_keys_sorted) < 2:
        print(f"  [TEKNİK-DEBUG {ticker}] yeterli tamamlanmış hafta yok (bulunan: {len(week_keys_sorted)})")
        return None, None

    last_weeks = week_keys_sorted[-3:]  # en eski -> en yeni, en fazla 3 hafta
    profiles = []
    for wk in last_weeks:
        week_bars = bars[bars["week_key"] == wk]
        prof = _compute_volume_profile(week_bars)
        if prof:
            profiles.append(prof)

    if len(profiles) < 2:
        print(f"  [TEKNİK-DEBUG {ticker}] {len(profiles)} haftalık profil hesaplanabildi (en az 2 lazım)")
        return None, None

    # --- Trend skoru: POC ve VA (orta nokta) art arda yükseliyor mu? -------
    pocs = [p["poc"] for p in profiles]
    va_mids = [(p["vah"] + p["val"]) / 2 for p in profiles]

    if len(profiles) == 3:
        poc_up_1 = pocs[1] > pocs[0]
        poc_up_2 = pocs[2] > pocs[1]
        va_up_1 = va_mids[1] > va_mids[0]
        va_up_2 = va_mids[2] > va_mids[1]
        if poc_up_1 and poc_up_2 and va_up_1 and va_up_2:
            trend_score = 2.0    # 3 hafta boyunca kesintisiz güçlenme
        elif poc_up_2 and va_up_2:
            trend_score = 1.0    # son hafta öncekini yenerek yukarı kaydı
        elif not poc_up_2 and not va_up_2:
            trend_score = -1.0   # son hafta geriledi
        else:
            trend_score = 0.0
    else:  # 2 hafta
        trend_score = 1.0 if (pocs[1] > pocs[0] and va_mids[1] > va_mids[0]) else (
            -1.0 if (pocs[1] < pocs[0] and va_mids[1] < va_mids[0]) else 0.0
        )

    # --- Seviye skoru: bugünkü fiyat, geçmiş haftaların seviyelerine göre nerede? ---
    latest = profiles[-1]
    all_pocs = pocs
    all_val = [p["val"] for p in profiles]
    all_vah = [p["vah"] for p in profiles]

    if current_price is None:
        level_score = None
    elif current_price < min(all_val):
        level_score = -1.0    # en alt VP seviyesinin bile altında
    elif current_price > max(all_pocs) and len(profiles) >= 2 and current_price > max(all_vah[:-1] or [0]):
        level_score = 2.0     # geçmiş haftaların value area'larının üzerinde
    elif latest["val"] <= current_price <= latest["vah"]:
        level_score = 1.0     # son haftanın value area'sı içinde
    elif current_price > max(all_pocs):
        level_score = 1.0     # tüm POC'ların üzerinde
    else:
        level_score = 0.0

    print(f"  [TEKNİK-DEBUG {ticker}] {len(profiles)} hafta VP hesaplandı, "
          f"POC dizilimi={[round(p,2) for p in pocs]}, trend={trend_score}, seviye={level_score}")
    return trend_score, level_score


def compute_daily_pbd_signal(ticker):
    """Son 3 işlem gününün hacim dağılım şeklini (P/b/D) sınıflandırır.
    Basitleştirilmiş yaklaşım: günün VWAP'ının, günün High-Low aralığındaki
    GÖRECELİ konumunu kullanır (0=Low, 1=High). >=0.6 -> P (üstte
    yoğunlaşma), <=0.4 -> b (altta yoğunlaşma), arası -> D (dengeli)."""
    bars = fetch_intraday_bars(ticker, period="5d", interval="1h")
    if bars is None or bars.empty:
        return None

    bars = bars.copy()
    bars["date_only"] = bars["datetime"].dt.date
    days = sorted(bars["date_only"].unique())[-3:]
    if len(days) < 2:
        return None

    shapes = []
    for d in days:
        day_bars = bars[bars["date_only"] == d]
        if day_bars.empty:
            continue
        day_high = day_bars["high"].max()
        day_low = day_bars["low"].min()
        if day_high <= day_low:
            continue
        typical = (day_bars["high"] + day_bars["low"] + day_bars["close"]) / 3
        vwap = (typical * day_bars["volume"]).sum() / day_bars["volume"].sum() if day_bars["volume"].sum() > 0 else None
        if vwap is None:
            continue
        position = (vwap - day_low) / (day_high - day_low)
        if position >= 0.6:
            shapes.append("P")
        elif position <= 0.4:
            shapes.append("b")
        else:
            shapes.append("D")

    if not shapes:
        return None

    print(f"  [TEKNİK-DEBUG {ticker}] son {len(shapes)} günün P/b/D yapısı: {shapes}")
    p_or_d = sum(1 for s in shapes if s in ("P", "D"))
    b_count = sum(1 for s in shapes if s == "b")
    if b_count > p_or_d:
        return -1.0
    elif p_or_d > b_count:
        return 1.0
    return 0.0


def compute_technical_score(ticker, current_price, return_20d=None, return_60d=None):
    """Bir hisse için 9 teknik bileşeni hesaplar ve 0-100 teknik skor üretir.

    ÖNEMLİ: Eksik sinyaller artık kalan ağırlıkları yapay biçimde %100'e
    büyütmez. Önce mevcut sinyaller kendi içinde yön skoru üretir, ardından
    bu skor toplam *veri kapsaması* kadar 50 (nötr) seviyesine doğru küçülür.
    Böylece örneğin 9 sinyalden yalnızca 3'ü mevcutken üçü de +1 diye skorun
    doğrudan 100'e yapışması engellenir.
    """
    daily_df = fetch_daily_series_isy(ticker)
    today = datetime.now(TR_TZ).date()

    raw = {
        "anchored_vp": compute_anchored_vwap_signal(daily_df, current_price, today),
        "monthly_open": compute_monthly_open_signal(daily_df, current_price, today),
        "delta_obv": compute_obv_delta_signal(daily_df),
        "avwap_52wk_high": compute_avwap_52wk_high_signal(daily_df, current_price),
        "sector_rs": compute_sector_rs_signal(return_20d, return_60d),
        "foreign_interest": compute_foreign_interest_signal(ticker),
    }

    if TECHNICAL_LAYER_AVAILABLE:
        trend_score, level_score = compute_weekly_vp_signals(ticker, current_price)
        raw["weekly_vp_trend"] = trend_score
        raw["weekly_vp_level"] = level_score
        raw["daily_pbd_shape"] = compute_daily_pbd_signal(ticker)
    else:
        raw["weekly_vp_trend"] = None
        raw["weekly_vp_level"] = None
        raw["daily_pbd_shape"] = None

    available = {k: v for k, v in raw.items() if v is not None}
    total_possible_weight = sum(TECH_SUB_WEIGHTS.values())
    available_weight = sum(TECH_SUB_WEIGHTS[k] for k in available)
    coverage = (available_weight / total_possible_weight) if total_possible_weight > 0 else 0.0

    if not available:
        technical_score = 50.0
        raw_score = 50.0
    else:
        # Mevcut sinyallerin kendi içindeki yön skoru (-1..+1).
        combined_z_available = sum(
            TECH_SUB_WEIGHTS[k] * (v / TECH_SUB_MAX_ABS[k]) for k, v in available.items()
        ) / available_weight

        raw_score = 50.0 + 50.0 * combined_z_available

        # Eksik veri varsa aşırı skoru nötre doğru küçült.
        # coverage=1.00 -> ham skor aynen kalır
        # coverage=0.40 -> ham sapmanın yalnızca %40'ı skora yansır
        shrunk_z = combined_z_available * coverage
        technical_score = 50.0 + 50.0 * shrunk_z
        technical_score = max(0.0, min(100.0, technical_score))

    # Claude ile kaldığımız teşhis adımı: 9 sinyalin tamamını tek satırda gör.
    def _dbg(v):
        return "N/A" if v is None else f"{float(v):+.2f}"

    ordered_debug = " | ".join(f"{k}={_dbg(raw.get(k))}" for k in TECH_SUB_WEIGHTS)
    print(
        f"  [TEKNİK-ÖZET {ticker}] {ordered_debug} | "
        f"mevcut={len(available)}/9 | coverage=%{coverage * 100:.0f} | "
        f"ham={raw_score:.1f} | düzeltilmiş={technical_score:.1f}"
    )

    meta = {
        "available_signals": len(available),
        "total_signals": len(TECH_SUB_WEIGHTS),
        "coverage": round(coverage, 4),
        "raw_score_before_coverage": round(raw_score, 2),
    }
    return technical_score, raw, meta


def round2_deep_analysis(candidates_df, last_seen_map=None):
    results = []
    records = candidates_df.to_dict("records")
    report_cache = load_report_cache()
    report_fetched_count = 0

    for i, row in enumerate(records, 1):
        ticker = row["ticker"]
        print(f"  Derin analiz {i}/{len(records)}: {ticker} (Bear -> Bull -> CRO)")

        if RAG_ENABLED:
            report_excerpt, report_period = get_report_excerpt_cached(ticker, report_cache)
            report_fetched_count += 1
            if report_fetched_count % 5 == 0:
                save_report_cache(report_cache)   # kesinti olursa ilerleme kaybolmasın
            time.sleep(1.0)   # KAP'a da nazik davranalım
        else:
            report_excerpt, report_period = None, None

        candidate = _build_candidate_for_agents(row, report_excerpt, report_period)
        result = analyze_with_gemini(candidate, (last_seen_map or {}).get(ticker))
        if result:
            # Data Confidence LLM'in tek başına 100 vermesine bırakılmıyor.
            # Sayısal veri tamlığı ve kullanılan kaynak katmanı ile tavan uygulanır.
            try:
                numeric_completeness_pct = float(row.get("data_completeness", 0)) * 100.0
            except (TypeError, ValueError):
                numeric_completeness_pct = 0.0
            try:
                llm_conf = float(result.get("data_confidence", 0))
            except (TypeError, ValueError):
                llm_conf = 0.0
            source_cap = 95.0 if report_excerpt else 85.0
            adjusted_conf = min(llm_conf, numeric_completeness_pct, source_cap)
            result["data_confidence_raw"] = result.get("data_confidence")
            result["data_confidence"] = int(round(max(0.0, adjusted_conf)))

            # Round 1'den gelen sayısal verileri de sakla (rapor/hafıza için)
            result["low_liquidity_flag"] = bool(row.get("low_liquidity_flag"))
            result["screening_score"] = round(float(row.get("screening_score", 0)), 1)
            result["cap_bucket"] = row.get("cap_bucket")
            # V5 görsel radar ve açıklanabilirlik için Round-1 bileşenlerini sakla.
            for _field in [
                "score_valuation", "score_momentum", "score_acceleration",
                "score_margin", "score_balance", "score_divergence",
                "cfo_to_net_income", "roe", "revenue_growth",
                "earnings_growth", "revenue_acceleration", "earnings_acceleration",
                "net_debt_to_equity", "debt_to_equity"
            ]:
                _v = row.get(_field)
                try:
                    result[_field] = None if pd.isna(_v) else float(_v)
                except Exception:
                    result[_field] = _v
            result["had_report_excerpt"] = bool(report_excerpt)
            result["_return_20d"] = row.get("return_20d")
            result["_return_60d"] = row.get("return_60d")
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



def build_previous_day_top10_map(history, exclude_date):
    """En yakın önceki taramanın gerçek TOP10'unu döndürür.
    Thesis için kullanılan 'last seen' haritasından ayrı tutulur.
    """
    prior = [h for h in history if h.get("date") and h.get("date") < exclude_date]
    if not prior:
        return {}
    latest = max(prior, key=lambda h: h.get("date", ""))
    ranked = sorted(latest.get("ranking", []), key=lambda x: x.get("rank", 9999))[:10]
    return {
        item.get("ticker"): {**item, "date": latest.get("date")}
        for item in ranked if item.get("ticker")
    }

def save_state(ranking):
    today_str = datetime.now(TR_TZ).strftime("%Y-%m-%d")
    state = {
        "date": today_str,
        "app_version": APP_VERSION,
        "scoring_version": SCORING_VERSION,
        "valuation_version": VALUATION_VERSION,
        "case_engine_version": CASE_ENGINE_VERSION,
        "ranking": ranking,
    }
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
# V5.14 — CASE-BASED THESIS LIFECYCLE + HISTORICAL DATA
# --------------------------------------------------------------------------
# Ana ilke:
# - TOP10'a her YENİ giriş yeni bir case açar.
# - TOP10'dan doğal çıkış, aktif case'i kapatır ve history'ye taşır.
# - Aynı ticker tekrar TOP10'a girerse eski case devam ETMEZ; yeni case açılır.
# - %8 Base eşiği TOP10'dan atma kuralı değildir. Sadece APPROACHING_BASE etiketi.
# - Base'e ulaşan ve destekli bull-case'i olan case aynı yaşam döngüsünde
#   BULL_CASE_WATCH'a geçebilir. Bull-case bittiğinde case kapanır.
# - İstatistik için her aktif case'in günlük snapshot'ı ayrı dosyada tutulur.
ACTIVE_CASES_FILE = os.path.join(os.path.dirname(__file__), "active_cases.json")
CASE_HISTORY_FILE = os.path.join(os.path.dirname(__file__), "case_history.json")
CASE_DAILY_HISTORY_FILE = os.path.join(os.path.dirname(__file__), "case_daily_history.json")

APPROACHING_ENTRY_PCT = 8.0
UPSIDE_WATCH_MIN_GAP_PCT = 8.0
FOLLOWUP_MAX_PER_BLOCK = 3
FOLLOWUP_MIN_FUNDAMENTAL = 55.0
FOLLOWUP_MIN_FINAL_ALPHA = 45.0


def _load_json_file(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"[CASE] {os.path.basename(path)} okunamadı: {e}")
        return default


def load_case_store():
    active = _load_json_file(ACTIVE_CASES_FILE, {"updated_at": None, "cases": {}})
    history = _load_json_file(CASE_HISTORY_FILE, {"updated_at": None, "closed_cases": []})
    daily = _load_json_file(CASE_DAILY_HISTORY_FILE, {"updated_at": None, "cases": {}})
    active.setdefault("cases", {})
    history.setdefault("closed_cases", [])
    daily.setdefault("cases", {})
    return active, history, daily


def save_case_store(active, history, daily):
    today = datetime.now(TR_TZ).strftime("%Y-%m-%d")
    active["updated_at"] = today
    history["updated_at"] = today
    daily["updated_at"] = today
    for path, payload in [
        (ACTIVE_CASES_FILE, active),
        (CASE_HISTORY_FILE, history),
        (CASE_DAILY_HISTORY_FILE, daily),
    ]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


def _case_sequence_for_ticker(ticker, active, history):
    seqs = []
    for rec in active.get("cases", {}).values():
        if rec.get("ticker") == ticker:
            seqs.append(int(rec.get("case_seq", 0) or 0))
    for rec in history.get("closed_cases", []):
        if rec.get("ticker") == ticker:
            seqs.append(int(rec.get("case_seq", 0) or 0))
    return (max(seqs) if seqs else 0) + 1


def _make_case_id(ticker, today, seq):
    return f"{ticker}-{today.replace('-', '')}-{seq:02d}"


def _bull_watch_supported(item):
    bull = _safe_float(item.get("bull_fv"))
    price = _safe_float(item.get("price"))
    if not bull or not price or price <= 0:
        return False, None
    gap = (bull / price - 1.0) * 100.0
    supported = (
        str(item.get("upside_case_status", "")).upper() == "SUPPORTED"
        and bool(item.get("upside_evidence"))
        and gap >= UPSIDE_WATCH_MIN_GAP_PCT
    )
    return supported, gap


def _copy_case_metrics(rec, item):
    keys = [
        "alpha_score", "fundamental_score", "technical_score", "data_confidence",
        "price", "bear_fv", "base_fv", "bull_fv", "base_upside_pct",
        "upside_case_status", "upside_evidence", "base_revision_status",
        "base_revision_evidence", "verdict", "cap_bucket", "catalysts", "risks",
        "revenue_growth", "earnings_growth", "revenue_acceleration",
        "earnings_acceleration", "gross_margin_trend", "operating_margin_trend",
        "roe", "net_debt_to_equity", "cfo_to_net_income",
        "score_valuation", "score_momentum", "score_acceleration", "score_margin",
        "score_balance", "score_divergence", "data_completeness", "screening_score",
        "trailing_pe", "price_to_book", "debt_to_equity", "deleveraging",
        "return_5d", "return_20d", "return_60d", "low_liquidity_flag", "name",
    ]
    for k in keys:
        if k in item:
            v = item.get(k)
            if isinstance(v, (np.floating, np.integer)):
                v = v.item()
            rec[k] = v
    return rec


def _new_case(ticker, item, rank, active, history, today):
    seq = _case_sequence_for_ticker(ticker, active, history)
    case_id = _make_case_id(ticker, today, seq)
    gap = _safe_float(item.get("base_upside_pct"))
    status = "APPROACHING_BASE" if gap is not None and 0 < gap < APPROACHING_ENTRY_PCT else "ACTIVE_OPPORTUNITY"
    rec = {
        "case_id": case_id,
        "case_seq": seq,
        "ticker": ticker,
        "status": status,
        "entry_date": today,
        "entry_price": _safe_float(item.get("price")),
        "entry_rank": rank,
        "entry_alpha": _safe_float(item.get("alpha_score")),
        "entry_fundamental": _safe_float(item.get("fundamental_score")),
        "entry_technical": _safe_float(item.get("technical_score")),
        "entry_bear_fv": _safe_float(item.get("bear_fv")),
        "entry_base_fv": _safe_float(item.get("base_fv")),
        "entry_bull_fv": _safe_float(item.get("bull_fv")),
        "last_seen_date": today,
        "last_rank": rank,
        "days_tracked": 0,
        "base_reached_date": None,
        "base_reached_price": None,
        "max_price": _safe_float(item.get("price")),
        "min_price": _safe_float(item.get("price")),
        "max_alpha": _safe_float(item.get("alpha_score")),
        "min_base_gap_pct": gap,
        "top10_exit_date": None,
        "close_date": None,
        "close_reason": None,
        "app_version": APP_VERSION,
        "scoring_version": SCORING_VERSION,
        "valuation_version": VALUATION_VERSION,
        "case_engine_version": CASE_ENGINE_VERSION,
    }
    _copy_case_metrics(rec, item)
    active["cases"][ticker] = rec
    print(f"  [CASE-AÇILDI {ticker}] {case_id} | giriş fiyat={rec.get('entry_price')} | rank=#{rank} | status={status}")
    return rec


def _close_case(ticker, active, history, today, reason, item=None):
    rec = active.get("cases", {}).get(ticker)
    if not rec:
        return
    if item:
        _copy_case_metrics(rec, item)
    rec["close_date"] = today
    rec["close_reason"] = reason
    exit_price = _safe_float((item or {}).get("price"))
    if exit_price is None:
        exit_price = _safe_float(rec.get("price"))
    rec["exit_price"] = exit_price
    rec["exit_rank"] = (item or {}).get("rank")
    entry_price = _safe_float(rec.get("entry_price"))
    exit_price = _safe_float(rec.get("exit_price"))
    rec["case_return_pct"] = round((exit_price / entry_price - 1.0) * 100.0, 2) if entry_price and exit_price else None
    rec["outcome"] = (
        "BASE_REACHED" if rec.get("base_reached_date") else
        "POSITIVE_EXIT" if (rec.get("case_return_pct") or 0) > 0 else
        "NEGATIVE_EXIT" if rec.get("case_return_pct") is not None else
        "CLOSED"
    )
    history.setdefault("closed_cases", []).append(dict(rec))
    active["cases"].pop(ticker, None)
    print(f"  [CASE-KAPANDI {ticker}] {rec.get('case_id')} | neden={reason} | getiri=%{rec.get('case_return_pct')}")


def _append_daily_snapshot(daily, rec, item, rank, today):
    case_id = rec["case_id"]
    price = _safe_float(item.get("price"))
    entry_price = _safe_float(rec.get("entry_price"))
    snapshot = {
        "date": today,
        "ticker": rec.get("ticker"),
        "case_id": case_id,
        "case_seq": rec.get("case_seq"),
        "status": rec.get("status"),
        "rank": rank,
        "price": price,
        "entry_price": entry_price,
        "case_return_pct": round((price / entry_price - 1.0) * 100.0, 2) if price and entry_price else None,
        "alpha_score": _safe_float(item.get("alpha_score")),
        "fundamental_score": _safe_float(item.get("fundamental_score")),
        "technical_score": _safe_float(item.get("technical_score")),
        "bear_fv": _safe_float(item.get("bear_fv")),
        "base_fv": _safe_float(item.get("base_fv")),
        "bull_fv": _safe_float(item.get("bull_fv")),
        "base_upside_pct": _safe_float(item.get("base_upside_pct")),
        "upside_case_status": item.get("upside_case_status"),
        "base_revision_status": item.get("base_revision_status"),
        "data_confidence": item.get("data_confidence"),
        "revenue_growth": _safe_float(item.get("revenue_growth")),
        "earnings_growth": _safe_float(item.get("earnings_growth")),
        "revenue_acceleration": _safe_float(item.get("revenue_acceleration")),
        "earnings_acceleration": _safe_float(item.get("earnings_acceleration")),
        "gross_margin_trend": _safe_float(item.get("gross_margin_trend")),
        "operating_margin_trend": _safe_float(item.get("operating_margin_trend")),
        "roe": _safe_float(item.get("roe")),
        "net_debt_to_equity": _safe_float(item.get("net_debt_to_equity")),
        "cfo_to_net_income": _safe_float(item.get("cfo_to_net_income")),
        "app_version": APP_VERSION,
        "scoring_version": SCORING_VERSION,
        "valuation_version": VALUATION_VERSION,
        "case_engine_version": CASE_ENGINE_VERSION,
    }
    arr = daily.setdefault("cases", {}).setdefault(case_id, [])
    # Manuel aynı-gün testleri istatistiği şişirmesin: aynı tarih snapshot'ı güncellenir.
    if arr and arr[-1].get("date") == today:
        arr[-1] = snapshot
    else:
        arr.append(snapshot)


def _restore_case_from_daily_history(ticker, active, history, daily, previous_item, previous_date):
    """Önceki günün final TOP10'unda olan ticker için aktif case kaybolmuşsa onarır.

    Bu, workflow/state geçişlerinde oluşabilecek yapay günlük case resetlerini engeller.
    Gerçek bir TOP10 çıkışı varsa ticker zaten önceki günün final TOP10'unda olmayacağı
    için bu fonksiyon devreye girmez.
    """
    if ticker in active.get("cases", {}):
        return active["cases"][ticker]

    # Önce günlük snapshot'lardan gerçek case kimliğini geri bul.
    best = None
    for case_id, arr in daily.get("cases", {}).items():
        if not arr:
            continue
        for snap in reversed(arr):
            if snap.get("ticker") == ticker and snap.get("date") == previous_date:
                best = (case_id, arr, snap)
                break
        if best:
            break

    if best:
        case_id, arr, snap = best
        # Aynı case yanlışlıkla history'ye kapanmışsa history'den çıkarıp yeniden aktive et.
        closed = None
        closed_cases = history.get("closed_cases", [])
        for i in range(len(closed_cases)-1, -1, -1):
            c = closed_cases[i]
            if c.get("case_id") == case_id:
                closed = closed_cases.pop(i)
                break
        if closed:
            rec = dict(closed)
            rec["close_date"] = None
            rec["close_reason"] = None
            rec.pop("exit_price", None)
            rec.pop("exit_rank", None)
            rec.pop("case_return_pct", None)
            rec.pop("outcome", None)
        else:
            rec = {
                "case_id": case_id,
                "case_seq": snap.get("case_seq") or 1,
                "ticker": ticker,
                "status": snap.get("status") or "ACTIVE_OPPORTUNITY",
                "entry_date": arr[0].get("date") or previous_date,
                "entry_price": snap.get("entry_price") or previous_item.get("price"),
                "entry_rank": previous_item.get("rank"),
                "entry_alpha": previous_item.get("alpha_score"),
                "entry_fundamental": previous_item.get("fundamental_score"),
                "entry_technical": previous_item.get("technical_score"),
                "entry_bear_fv": previous_item.get("bear_fv"),
                "entry_base_fv": previous_item.get("base_fv"),
                "entry_bull_fv": previous_item.get("bull_fv"),
                "last_seen_date": previous_date,
                "last_rank": previous_item.get("rank"),
                "days_tracked": len({x.get("date") for x in arr if x.get("date")}),
                "base_reached_date": None,
                "base_reached_price": None,
                "max_price": snap.get("price"),
                "min_price": snap.get("price"),
                "max_alpha": snap.get("alpha_score"),
                "min_base_gap_pct": snap.get("base_upside_pct"),
                "top10_exit_date": None,
                "close_date": None,
                "close_reason": None,
            }
            _copy_case_metrics(rec, previous_item)
        rec["app_version"] = APP_VERSION
        rec["scoring_version"] = SCORING_VERSION
        rec["valuation_version"] = VALUATION_VERSION
        rec["case_engine_version"] = CASE_ENGINE_VERSION
        active.setdefault("cases", {})[ticker] = rec
        print(f"  [CASE-ONARILDI {ticker}] {case_id} | önceki final TOP10 ile devam")
        return rec

    # Snapshot yoksa (eski sürümden geçiş) önceki günkü final TOP10 kaydından
    # tek seferlik continuity case oluştur.
    seq = _case_sequence_for_ticker(ticker, active, history)
    case_id = _make_case_id(ticker, previous_date, seq)
    gap = _safe_float(previous_item.get("base_upside_pct"))
    rec = {
        "case_id": case_id, "case_seq": seq, "ticker": ticker,
        "status": "APPROACHING_BASE" if gap is not None and 0 < gap < APPROACHING_ENTRY_PCT else "ACTIVE_OPPORTUNITY",
        "entry_date": previous_date, "entry_price": _safe_float(previous_item.get("price")),
        "entry_rank": previous_item.get("rank"), "entry_alpha": _safe_float(previous_item.get("alpha_score")),
        "entry_fundamental": _safe_float(previous_item.get("fundamental_score")),
        "entry_technical": _safe_float(previous_item.get("technical_score")),
        "entry_bear_fv": _safe_float(previous_item.get("bear_fv")),
        "entry_base_fv": _safe_float(previous_item.get("base_fv")),
        "entry_bull_fv": _safe_float(previous_item.get("bull_fv")),
        "last_seen_date": previous_date, "last_rank": previous_item.get("rank"),
        "days_tracked": 1, "base_reached_date": None, "base_reached_price": None,
        "max_price": _safe_float(previous_item.get("price")), "min_price": _safe_float(previous_item.get("price")),
        "max_alpha": _safe_float(previous_item.get("alpha_score")), "min_base_gap_pct": gap,
        "top10_exit_date": None, "close_date": None, "close_reason": None,
        "app_version": APP_VERSION, "scoring_version": SCORING_VERSION,
        "valuation_version": VALUATION_VERSION, "case_engine_version": CASE_ENGINE_VERSION,
    }
    _copy_case_metrics(rec, previous_item)
    active.setdefault("cases", {})[ticker] = rec
    print(f"  [CASE-MIGRASYON {ticker}] {case_id} | önceki TOP10'dan continuity case oluşturuldu")
    return rec


def reconcile_case_continuity(previous_day_top10_map, active, history, daily):
    """Önceki taramanın final TOP10'unda bulunan hisselerin aktif case'ini garanti eder.

    V5.14/V5.16 geçişinde oluşmuş olabilecek sahte günlük case resetlerini de
    tek seferlik onarır: ticker dün final TOP10'da ise ve bugün yeni case açılmış
    görünürken eski case bugün TOP10_EXIT ile kapanmışsa, eski case yeniden aktive
    edilir; bugünkü yapay case snapshot'ı silinir.
    """
    if not previous_day_top10_map:
        return
    today = datetime.now(TR_TZ).strftime("%Y-%m-%d")
    previous_dates = [x.get("date") or x.get("_history_date") for x in previous_day_top10_map.values()]
    previous_date = max([d for d in previous_dates if d] or [None])
    if not previous_date:
        previous_date = max([h.get("date") for h in load_full_history() if h.get("date") and h.get("date") < today], default=None)
    if not previous_date:
        return

    for ticker, previous_item in previous_day_top10_map.items():
        current = active.get("cases", {}).get(ticker)

        # Bugünkü yapay reset'i yakala: yeni case bugün açılmış ama ticker dün de
        # final TOP10'daydı. Eski kapanmış case'i geri getir.
        if current and current.get("entry_date") == today:
            closed_cases = history.get("closed_cases", [])
            restore_idx = None
            for i in range(len(closed_cases)-1, -1, -1):
                c = closed_cases[i]
                if (
                    c.get("ticker") == ticker
                    and c.get("close_date") == today
                    and c.get("close_reason") == "TOP10_EXIT"
                    and (c.get("entry_date") or "") <= previous_date
                ):
                    restore_idx = i
                    break
            if restore_idx is not None:
                old = closed_cases.pop(restore_idx)
                bogus_case_id = current.get("case_id")
                if bogus_case_id:
                    daily.get("cases", {}).pop(bogus_case_id, None)
                old["close_date"] = None
                old["close_reason"] = None
                old.pop("exit_price", None)
                old.pop("exit_rank", None)
                old.pop("case_return_pct", None)
                old.pop("outcome", None)
                old["app_version"] = APP_VERSION
                old["scoring_version"] = SCORING_VERSION
                old["valuation_version"] = VALUATION_VERSION
                old["case_engine_version"] = CASE_ENGINE_VERSION
                active["cases"][ticker] = old
                print(f"  [CASE-BİRLEŞTİR {ticker}] {bogus_case_id} iptal -> {old.get('case_id')} devam")
                continue

        if ticker not in active.get("cases", {}):
            _restore_case_from_daily_history(ticker, active, history, daily, previous_item, previous_date)


def augment_candidates_with_active_cases(candidates_df, active_store):
    """Aktif case'ler Round1 shortlist'inden düşse de günlük yeniden analiz edilir.
    Özellikle APPROACHING_BASE ve BULL_CASE_WATCH case'lerinin unutulmasını önler.
    """
    global LAST_ROUND1_SCORED_DF
    if LAST_ROUND1_SCORED_DF is None or LAST_ROUND1_SCORED_DF.empty:
        return candidates_df
    tracked = set(active_store.get("cases", {}).keys())
    if not tracked:
        return candidates_df
    existing = set(candidates_df["ticker"].astype(str))
    add = LAST_ROUND1_SCORED_DF[
        LAST_ROUND1_SCORED_DF["ticker"].astype(str).isin(tracked - existing)
    ].copy()
    if add.empty:
        return candidates_df
    print("[CASE] Round2'ye aktif case takip adayları eklendi: " + ", ".join(add["ticker"].astype(str).tolist()))
    return pd.concat([candidates_df, add], ignore_index=True).drop_duplicates("ticker", keep="first")


def active_main_case_tickers(active_store):
    return {
        tk for tk, rec in active_store.get("cases", {}).items()
        if rec.get("status") in {"ACTIVE_OPPORTUNITY", "APPROACHING_BASE"}
    }


def update_case_lifecycle(ranked, analyzed, active, history, daily):
    """TOP10 giriş/çıkışlarını case bazında işler.

    Önemli:
    - %8'in altına inen hisse TOP10'dan çıkarılmaz; yalnızca APPROACHING_BASE olur.
    - TOP10'dan doğal sıralama nedeniyle çıkan ACTIVE/APPROACHING case kapanır.
    - Aynı ticker daha sonra tekrar TOP10'a girerse yeni case açılır.
    - Base'e ulaşan case destekli bull-case varsa BULL_CASE_WATCH'a geçebilir.
    """
    today = datetime.now(TR_TZ).strftime("%Y-%m-%d")
    top10 = ranked[:10]
    top10_map = {x.get("ticker"): (i, x) for i, x in enumerate(top10, 1) if x.get("ticker")}
    analyzed_map = {x.get("ticker"): x for x in analyzed if x.get("ticker")}

    # 1) Mevcut case'leri güncelle / doğal çıkışları kapat / Base temasını işle.
    for tk in list(active.get("cases", {}).keys()):
        rec = active["cases"].get(tk)
        item = analyzed_map.get(tk)
        if not rec:
            continue
        status = rec.get("status")

        if status == "BULL_CASE_WATCH":
            # Yeni Base revizyonu nedeniyle tekrar TOP10'a girdiyse eski bull case kapanır;
            # aşağıda sıfırdan yeni case açılır.
            if tk in top10_map:
                _close_case(tk, active, history, today, "REENTERED_TOP10_NEW_CASE", item=item)
                continue
            if item is None:
                continue
            _copy_case_metrics(rec, item)
            supported, bull_gap = _bull_watch_supported(item)
            rec["bull_upside_pct"] = round(bull_gap, 1) if bull_gap is not None else None
            rec["last_seen_date"] = today
            rec["days_tracked"] = int(rec.get("days_tracked", 0)) + 1
            if not supported:
                _close_case(tk, active, history, today, "BULL_CASE_NO_LONGER_SUPPORTED", item=item)
                continue
            _append_daily_snapshot(daily, rec, item, None, today)
            continue

        # ACTIVE / APPROACHING ana TOP10 case'i
        if tk in top10_map:
            rank, item = top10_map[tk]
            print(f"  [CASE-DEVAM {tk}] {rec.get('case_id')} | gün={int(rec.get('days_tracked', 0)) + 1} | rank=#{rank}")
            _copy_case_metrics(rec, item)
            gap = _safe_float(item.get("base_upside_pct"))
            rec["last_seen_date"] = today
            rec["last_rank"] = rank
            rec["days_tracked"] = int(rec.get("days_tracked", 0)) + 1
            price = _safe_float(item.get("price"))
            alpha = _safe_float(item.get("alpha_score"))
            if price is not None:
                rec["max_price"] = max([x for x in [_safe_float(rec.get("max_price")), price] if x is not None])
                rec["min_price"] = min([x for x in [_safe_float(rec.get("min_price")), price] if x is not None])
            if alpha is not None:
                prev_max_alpha = _safe_float(rec.get("max_alpha"))
                if prev_max_alpha is None:
                    prev_max_alpha = alpha
                rec["max_alpha"] = max(prev_max_alpha, alpha)
            if gap is not None:
                old_gap = _safe_float(rec.get("min_base_gap_pct"))
                rec["min_base_gap_pct"] = min(gap, old_gap) if old_gap is not None else gap
                rec["status"] = "APPROACHING_BASE" if 0 < gap < APPROACHING_ENTRY_PCT else "ACTIVE_OPPORTUNITY"
            _append_daily_snapshot(daily, rec, item, rank, today)
            continue

        # TOP10'da değil. Base gerçekten ulaşılmışsa bull-case'e geçme hakkı var;
        # aksi halde bu case doğal TOP10 çıkışıyla kapanır.
        if item is not None:
            _copy_case_metrics(rec, item)
            gap = _safe_float(item.get("base_upside_pct"))
            if gap is not None and gap <= 0:
                if rec.get("base_reached_date") is None:
                    rec["base_reached_date"] = today
                    rec["base_reached_price"] = _safe_float(item.get("price"))
                supported, bull_gap = _bull_watch_supported(item)
                rec["bull_upside_pct"] = round(bull_gap, 1) if bull_gap is not None else None
                if supported:
                    rec["status"] = "BULL_CASE_WATCH"
                    rec["last_seen_date"] = today
                    rec["days_tracked"] = int(rec.get("days_tracked", 0)) + 1
                    _append_daily_snapshot(daily, rec, item, None, today)
                    print(f"  [CASE-DURUM {tk}] {rec.get('case_id')} -> BULL_CASE_WATCH")
                    continue
                _close_case(tk, active, history, today, "BASE_REACHED_THESIS_COMPLETE", item=item)
                continue
        _close_case(tk, active, history, today, "TOP10_EXIT", item=item)

    # 2) Bugünkü TOP10'da olup aktif case'i bulunmayan her ticker için YENİ case aç.
    for rank, item in enumerate(top10, 1):
        tk = item.get("ticker")
        if not tk:
            continue
        if tk not in active.get("cases", {}):
            rec = _new_case(tk, item, rank, active, history, today)
            rec["days_tracked"] = 1
            _append_daily_snapshot(daily, rec, item, rank, today)

    # 3) Görsel için istatistik logları
    approaching_count = sum(1 for r in active.get("cases", {}).values() if r.get("status") == "APPROACHING_BASE")
    bull_count = sum(1 for r in active.get("cases", {}).values() if r.get("status") == "BULL_CASE_WATCH")
    print(f"[CASE-TAKİP] aktif case={len(active.get('cases', {}))} | approaching={approaching_count} | bull_watch={bull_count} | kapalı case={len(history.get('closed_cases', []))}")
    return active, history, daily


def _followup_priority(rec):
    alpha = _safe_float(rec.get("alpha_score")) or 0.0
    fundamental = _safe_float(rec.get("fundamental_score")) or 0.0
    if rec.get("status") == "APPROACHING_BASE":
        gap = max(0.0, _safe_float(rec.get("base_upside_pct")) or 0.0)
        return (APPROACHING_ENTRY_PCT - min(gap, APPROACHING_ENTRY_PCT)) * 3.0 + alpha + fundamental * 0.15
    if rec.get("status") == "BULL_CASE_WATCH":
        bull_gap = max(0.0, _safe_float(rec.get("bull_upside_pct")) or 0.0)
        return alpha + fundamental * 0.15 + min(bull_gap, 30.0) * 0.20
    return alpha


def followup_visible_payload(active_store):
    """İkinci görsel bir kapasite limiti değil, görünüm limitidir.
    APPROACHING_BASE yalnızca hâlen TOP10 case'i olanlardan oluşur ve en fazla 3'ü gösterilir.
    Bull Watch tarafında da aktif case'ler arasından en anlamlı 3 gösterilir.
    """
    records = list(active_store.get("cases", {}).values())
    approaching = [r for r in records if r.get("status") == "APPROACHING_BASE"]
    bull = [r for r in records if r.get("status") == "BULL_CASE_WATCH"]
    approaching.sort(key=_followup_priority, reverse=True)
    bull.sort(key=_followup_priority, reverse=True)
    return approaching[:FOLLOWUP_MAX_PER_BLOCK], bull[:FOLLOWUP_MAX_PER_BLOCK]

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


def _safe_float(v):
    try:
        if v is None or v == "N/A":
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


FINAL_ALPHA_MIN_ACTIVE = 45.0

# Fundamental Opportunity Gate
# ----------------------------
# Sistemin ana amacı yalnızca iyi şirket bulmak değil; güncel temel analize
# göre henüz yeterince fiyatlanmamış şirketleri bulmaktır. Bu yüzden teknik
# skor sıralamaya girmeden ÖNCE üç koşul aranır:
#   1) Fundamental kalite asgari seviyenin üzerinde olmalı.
#   2) Güncel Base FV, mevcut fiyatın anlamlı ölçüde üzerinde olmalı.
#   3) Final Alpha asgari aktif eşiği geçmeli.
# Böylece teknik skor, güncel Base değerini zaten fiyatlamış bir hisseyi
# yeniden TOP3/TOP10'a taşıyamaz.
MIN_FUNDAMENTAL_SCORE_ACTIVE = 55.0
MIN_BASE_UPSIDE_PCT_ACTIVE = 8.0


def _opportunity_gate(item, fundamental, final_alpha, continuing_case=False):
    price = _safe_float(item.get("price"))
    base_fv = _safe_float(item.get("base_fv"))

    if price is None or price <= 0:
        return False, None, "güncel fiyat yok"
    if base_fv is None or base_fv <= 0:
        return False, None, "Base FV yok/geçersiz"

    base_upside_pct = ((base_fv / price) - 1.0) * 100.0

    if fundamental < MIN_FUNDAMENTAL_SCORE_ACTIVE:
        return False, base_upside_pct, (
            f"fundamental skor {fundamental:.1f} < {MIN_FUNDAMENTAL_SCORE_ACTIVE:.1f}"
        )

    # Yeni bir case ancak en az %8 Base alanı varken ana fırsat havuzuna girebilir.
    # FAKAT hâlihazırda TOP10 case'i olan bir hisse %8'in altına düştü diye
    # listeden atılmaz; Base'e gerçekten ulaşana kadar ranking'de kalabilir.
    if base_upside_pct <= 0:
        return False, base_upside_pct, (
            f"güncel fiyat Base FV'yi fiyatlamış/aşmış (Base fark %{base_upside_pct:.1f})"
        )
    if base_upside_pct < MIN_BASE_UPSIDE_PCT_ACTIVE and not continuing_case:
        return False, base_upside_pct, (
            f"yeni case için Base fark yalnızca %{base_upside_pct:.1f}; "
            f"giriş eşiği %{MIN_BASE_UPSIDE_PCT_ACTIVE:.1f}"
        )

    if final_alpha < FINAL_ALPHA_MIN_ACTIVE:
        return False, base_upside_pct, (
            f"Final Alpha {final_alpha:.1f} < {FINAL_ALPHA_MIN_ACTIVE:.1f}"
        )

    return True, base_upside_pct, "uygun"


def assign_final_alpha_scores(analyzed, continuing_case_tickers=None):
    """Final Alpha + Fundamental Opportunity Gate.

    Akış:
        1) CRO'nun güncel fundamental analizi ve Base FV'si alınır.
        2) Fundamental fırsat filtresi uygulanır.
        3) Yalnızca filtreden geçen hisseler için Final Alpha sıralaması yapılır.

    Final Alpha = Fundamental %70 + Teknik %30

    Teknik katman bir *timing/ranking overlay*'idir. Tek başına bir şirketi
    fırsat haline getiremez ve fiyat Base FV'yi zaten tüketmişse o şirketi
    aktif listeye geri sokamaz.
    """
    for item in analyzed:
        fundamental = _safe_float(item.get("fundamental_score"))
        if fundamental is None:
            fundamental = _safe_float(item.get("alpha_score")) or 0.0
        technical = _safe_float(item.get("technical_score")) or 50.0

        final_alpha = max(
            0.0,
            min(
                100.0,
                fundamental * TECH_FUNDAMENTAL_WEIGHT
                + technical * TECH_TECHNICAL_WEIGHT,
            ),
        )

        continuing = item.get("ticker") in (continuing_case_tickers or set())
        eligible, base_upside_pct, reason = _opportunity_gate(
            item, fundamental, final_alpha, continuing_case=continuing
        )

        item["fundamental_score"] = round(fundamental, 1)
        item["final_alpha_score"] = round(final_alpha, 1)
        item["alpha_score"] = round(final_alpha, 1)
        item["base_upside_pct"] = (
            round(base_upside_pct, 1) if base_upside_pct is not None else None
        )
        item["quality_eligible"] = bool(eligible)
        item["top3_eligible"] = bool(eligible)
        item["exclusion_reason"] = None if eligible else reason

        status = "GEÇTİ" if eligible else "ELENDİ"
        upside_txt = (
            "N/A" if base_upside_pct is None else f"%{base_upside_pct:.1f}"
        )
        print(
            f"  [OPPORTUNITY-GATE {item.get('ticker')}] {status} | "
            f"fundamental={fundamental:.1f} | teknik={technical:.1f} | "
            f"final={final_alpha:.1f} | Base fark={upside_txt} | neden={reason}"
        )

    active = [x for x in analyzed if x.get("quality_eligible")]
    excluded = [x for x in analyzed if not x.get("quality_eligible")]

    # Gate'ten geçenler arasında sıralama yine TEK public skor olan Final Alpha.
    active.sort(key=lambda x: x.get("alpha_score", 0), reverse=True)
    excluded.sort(key=lambda x: x.get("alpha_score", 0), reverse=True)

    for i, item in enumerate(active, 1):
        item["rank"] = i
    for item in excluded:
        item["rank"] = None

    return active, excluded

def build_report(ranked, last_seen_map, total_scanned, deep_count, previous_day_top10_map=None, quality_excluded=None):
    """Telegram görseli üretilemezse kullanılacak sade metin fallback raporu.
    V5'te kullanıcıya tek skor gösterilir: Final Alpha Score.
    """
    today = datetime.now(TR_TZ).strftime("%d.%m.%Y")
    previous_day_top10_map = previous_day_top10_map or {}
    top10 = ranked[:10]
    reserve = ranked[10:15]
    lines = [
        "📊 BEIQ — GÜNLÜK ANALİZ ÖZETİ",
        f"Tarih: {today}",
        f"Taranan: {total_scanned} | Derin analiz: {deep_count}",
        ""
    ]
    for i, item in enumerate(ranked[:3], 1):
        prev = last_seen_map.get(item["ticker"]) if last_seen_map else None
        prev_rank = f"#{prev.get('rank')}" if prev and prev.get("rank") else "YENİ"
        lines.append(f"{['🥇','🥈','🥉'][i-1]} {i}. {item['ticker']} [{item.get('cap_bucket','—')}]")
        lines.append(f"Alpha Score: {item.get('alpha_score','N/A')}/100 | Veri Güveni: {confidence_label(item.get('data_confidence'))}")
        lines.append(f"Analitik Profil: {profile_label(item.get('alpha_score'))}")
        lines.append(f"Fiyat: {fmt(item.get('price'))} TL | Downside/Base/Upside: {fmt(item.get('bear_fv'))} / {fmt(item.get('base_fv'))} / {fmt(item.get('bull_fv'))}")
        lines.append(f"Önceki Sıra: {prev_rank}")
        lines.append("")

    lines.append("📋 TOP 10 — FINAL ALPHA SCORE")
    for i, item in enumerate(top10, 1):
        lines.append(f"{i}. {item['ticker']} | Alpha {item.get('alpha_score','N/A')} | Fiyat {fmt(item.get('price'))} | Base {fmt(item.get('base_fv'))}")
    if reserve:
        lines.append("")
        lines.append("🧾 RESERVE (11–15)")
        for i, item in enumerate(reserve, 11):
            lines.append(f"{i}. {item['ticker']} | Alpha {item.get('alpha_score','N/A')} | Base {fmt(item.get('base_fv'))}")
    lines.append("")
    lines.append("Not: Listeye girişte güncel Base değerleme ve fundamental uygunluk filtresi uygulanır; Alpha Score, bu filtreden geçen adayları fundamental (%70) ve teknik (%30) bileşenlerle sıralar.")
    lines.append("Bu içerik yalnızca karşılaştırmalı analiz özetidir; alım-satım çağrısı veya kişiye özel yatırım tavsiyesi içermez.")
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


def send_telegram_photo(path, caption=None):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    with open(path, "rb") as f:
        data = {"chat_id": TELEGRAM_CHAT_ID}
        if caption:
            data["caption"] = caption[:1000]
        r = requests.post(url, data=data, files={"photo": f}, timeout=120)
    if not r.ok:
        raise RuntimeError(f"Telegram görsel gönderim hatası: {r.text}")



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

    # Değerleme hafızasını Gemini analizinden ÖNCE yükle. Böylece aynı veriyle
    # Base FV'nin her gün fiyatı takip ederek yukarı taşınması engellenebilir.
    today_str = datetime.now(TR_TZ).strftime("%Y-%m-%d")
    full_history = load_full_history()
    last_seen_map = build_last_seen_map(full_history, exclude_date=today_str)
    previous_day_top10_map = build_previous_day_top10_map(full_history, exclude_date=today_str)
    active_cases, case_history, case_daily_history = load_case_store()
    reconcile_case_continuity(previous_day_top10_map, active_cases, case_history, case_daily_history)
    candidates_df = augment_candidates_with_active_cases(candidates_df, active_cases)

    print(f"Round 2: {len(candidates_df)} aday için Gemini analizi başlıyor...")
    if "cap_bucket" in candidates_df.columns:
        cap_counts = candidates_df["cap_bucket"].value_counts().to_dict()
        print("[ÖLÇEK-DAĞILIMI ROUND2] " + " | ".join(
            f"{bucket}={cap_counts.get(bucket, 0)}" for bucket in ["Büyük", "Orta", "Küçük"]
        ))
    analyzed = round2_deep_analysis(candidates_df, last_seen_map=last_seen_map)
    if not analyzed:
        send_telegram_message("⚠️ Gemini analizinden sonuç alınamadı, lütfen logları kontrol et.")
        return

    print("Teknik Analiz Katmanı: 22 finalist için hesaplanıyor (fundamental %70 + teknik %30)...")
    for item in analyzed:
        tech_score, tech_raw, tech_meta = compute_technical_score(
            item["ticker"], item.get("price"),
            return_20d=item.get("_return_20d"), return_60d=item.get("_return_60d"),
        )
        item["technical_score"] = round(tech_score, 1)
        item["technical_components"] = tech_raw
        item["technical_coverage"] = tech_meta["coverage"]
        item["technical_available_signals"] = tech_meta["available_signals"]
        # CRO'nun fundamental alpha'sını arka planda sakla. Public alpha daha sonra Final Alpha olacak.
        item["fundamental_score"] = round(float(item.get("alpha_score", 0)), 1)
        time.sleep(0.5)   # borsapy/İş Yatırım'a nazik davranalım
    save_foreign_ratio_history()

    continuing_case_tickers = active_main_case_tickers(active_cases)
    ranked, quality_excluded = assign_final_alpha_scores(
        analyzed, continuing_case_tickers=continuing_case_tickers
    )
    active_cases, case_history, case_daily_history = update_case_lifecycle(
        ranked, analyzed, active_cases, case_history, case_daily_history
    )
    save_case_store(active_cases, case_history, case_daily_history)
    print(f"Final Alpha Engine: {len(ranked)}/{len(analyzed)} aday aktif fırsat havuzunda; TOP10 case takibi ayrı state'te tutuluyor.")
    for item in ranked[:15]:
        print(f"  [FINAL-ALPHA {item['ticker']}] fundamental={item.get('fundamental_score')} | teknik={item.get('technical_score')} | final={item.get('alpha_score')} | Base fark=%{item.get('base_upside_pct')}")
    top10_cap_counts = {}
    for item in ranked[:10]:
        bucket = item.get("cap_bucket", "Bilinmiyor")
        top10_cap_counts[bucket] = top10_cap_counts.get(bucket, 0) + 1
    print("[ÖLÇEK-DAĞILIMI TOP10] " + " | ".join(
        f"{bucket}={top10_cap_counts.get(bucket, 0)}" for bucket in ["Büyük", "Orta", "Küçük"]
    ))

    report = build_report(
        ranked,
        last_seen_map,
        total_scanned=len(tickers),
        deep_count=len(candidates_df),
        previous_day_top10_map=previous_day_top10_map,
        quality_excluded=quality_excluded,
    )
    print(report)
    try:
        daily_png = render_daily_report(
            ranked, last_seen_map, previous_day_top10_map,
            total_scanned=len(tickers), deep_count=len(candidates_df)
        )
        send_telegram_photo(
            daily_png,
            caption=f"BEIQ | Günlük Analiz Özeti | {datetime.now(TR_TZ).strftime('%d.%m.%Y')}\nKarşılaştırmalı analizdir; yatırım tavsiyesi değildir."
        )
        print(f"Günlük görsel Telegram'a gönderildi: {daily_png}")
    except Exception as e:
        print(f"[GÖRSEL UYARI] Günlük görsel üretilemedi/gönderilemedi: {e}")
        print("Metin fallback raporu gönderiliyor.")
        send_telegram_message(report)

    # İkinci günlük görsel: sticky tez yaşam döngüsü takibi. Bildirim değil,
    # sadece maksimum 3+3 hisselik takip paneli.
    try:
        approaching_rows, bull_rows = followup_visible_payload(active_cases)
        followup_png = render_followup_report(approaching_rows, bull_rows)
        send_telegram_photo(
            followup_png,
            caption=f"BEIQ | Tez Takibi | {datetime.now(TR_TZ).strftime('%d.%m.%Y')}\nBase'e yaklaşanlar ve destekli Upside Watch."
        )
        print(f"Tez takip görseli Telegram'a gönderildi: {followup_png}")
    except Exception as e:
        print(f"[TAKİP GÖRSEL UYARI] Tez takip görseli üretilemedi/gönderilemedi: {e}")

    # Faz 4 (Thesis Tracking) için zenginleştirilmiş kayıt: artık sadece
    # sıra/skor değil, Base/Bear/Bull FV ve Verdict de saklanıyor ki
    # ileride "bu hissenin tezi nasıl değişti" karşılaştırması yapılabilsin.
    save_state([
        {
            "ticker": r["ticker"], "rank": r["rank"],
            "alpha_score": r.get("alpha_score"),  # V5 canonical Final Alpha
            "fundamental_score": r.get("fundamental_score"),
            "technical_score": r.get("technical_score"),
            "technical_coverage": r.get("technical_coverage"),
            "technical_available_signals": r.get("technical_available_signals"),
            "base_upside_pct": r.get("base_upside_pct"),
            "quality_eligible": r.get("quality_eligible"),
            "top3_eligible": r.get("top3_eligible"),
            "price": r.get("price"),
            "bear_fv": r.get("bear_fv"), "base_fv": r.get("base_fv"), "bull_fv": r.get("bull_fv"),
            "base_revision_status": r.get("base_revision_status"),
            "base_revision_evidence": r.get("base_revision_evidence"),
            "upside_case_status": r.get("upside_case_status"),
            "upside_evidence": r.get("upside_evidence"),
            "revenue_growth": r.get("revenue_growth"),
            "earnings_growth": r.get("earnings_growth"),
            "revenue_acceleration": r.get("revenue_acceleration"),
            "earnings_acceleration": r.get("earnings_acceleration"),
            "gross_margin_trend": r.get("gross_margin_trend"),
            "operating_margin_trend": r.get("operating_margin_trend"),
            "roe": r.get("roe"),
            "net_debt_to_equity": r.get("net_debt_to_equity"),
            "cfo_to_net_income": r.get("cfo_to_net_income"),
            "app_version": APP_VERSION,
            "scoring_version": SCORING_VERSION,
            "valuation_version": VALUATION_VERSION,
            "case_engine_version": CASE_ENGINE_VERSION,
            "verdict": r.get("verdict"),
        }
        for r in ranked[:15]
    ])
    print("Tamamlandı.")


if __name__ == "__main__":
    main()
