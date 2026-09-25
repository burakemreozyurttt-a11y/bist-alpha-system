"""
BIST FUNDAMENTAL ALPHA — Faz 2: HAFTALIK ÖZET
================================================
daily_scan.yml workflow'unun, scanner.py'den HEMEN SONRA çalışan ikinci
adımıdır. Sadece Cuma günleri (ya da elle tetiklendiğinde) devreye girer.

Ne yapar:
  1) history.json'daki bu haftanın (Pazartesi-Cuma) günlük TOP15 kayıtlarını
     toplar
  2) Hisse bazında "hangi gün kaçıncı sırada, hangi alpha score ile listede
     kaldı" tablosunu çıkarır
  3) Bu tabloyu Gemini'ye vererek Türkçe bir "Bu Hafta Ne Oldu" özeti
     yazdırır (yeni dış veri çekmez, SADECE o hafta zaten topladığımız
     verilere dayanır — Gemini'nin bilmediği haberi uydurmasını istemiyoruz)
  4) Telegram'a yollar

Not: history.json, scanner.py'nin save_state() fonksiyonu tarafından her
günün sonunda otomatik güncelleniyor, bu script sadece OKUYOR.
"""
import json
import os
import time
from datetime import datetime, timedelta, timezone

import requests
from google import genai
from google.genai import types
from visual_report import render_weekly_report

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

# scanner.py'deki ile aynı yedekleme zinciri
MODEL_CANDIDATES = [
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-3.6-flash",
]

HISTORY_FILE = os.path.join(os.path.dirname(__file__), "history.json")
TR_TZ = timezone(timedelta(hours=3))

client = genai.Client(api_key=GEMINI_API_KEY)


def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def this_week_entries(history):
    """Bu haftanın (Pazartesi'den bugüne kadar) kayıtlarını filtreler."""
    now = datetime.now(TR_TZ)
    monday = (now - timedelta(days=now.weekday())).date()
    entries = []
    for h in history:
        try:
            d = datetime.strptime(h["date"], "%Y-%m-%d").date()
        except Exception:
            continue
        if monday <= d <= now.date():
            entries.append(h)
    return sorted(entries, key=lambda x: x["date"])


def build_aggregate(entries):
    """Hisse bazında: hangi günler TOP15'te göründü, hangi sıra/alpha score ile."""
    agg = {}
    for e in entries:
        for item in e.get("ranking", []):
            tk = item.get("ticker")
            if not tk:
                continue
            agg.setdefault(tk, []).append({
                "date": e["date"],
                "rank": item.get("rank"),
                "alpha_score": item.get("alpha_score"),
                "scoring_version": item.get("scoring_version") or e.get("scoring_version") or "LEGACY_UNVERSIONED",
            })
    return agg


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



def summarize_with_gemini(agg, methodology_changed=False):
    lines = []
    for tk, records in sorted(agg.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        detail = ", ".join(f"{r['date']}: sıra {r['rank']}, alpha {r['alpha_score']}" for r in records)
        lines.append(f"{tk} ({len(records)} gün): {detail}")
    raw_data = "\n".join(lines)

    methodology_instruction = (
        "Hafta içinde scoring metodolojisi değişti. Bunu yalnızca TEK kısa cümleyle kullanıcıya açıkla; "
        "iç model/sürüm kodlarını ASLA yazma. Skor değişimlerini şirket temel/teknik görünümündeki değişimin doğrudan kanıtı gibi yorumlama."
        if methodology_changed else
        "Hafta boyunca karşılaştırılabilir scoring metodolojisi kullanıldı."
    )

    prompt = f"""
Sen BEIQ haftalık araştırma notunu yazan deneyimli bir piyasa analistisin.
Aşağıdaki veri yalnızca günlük liste sıralamaları ve Alpha Score'lardan oluşuyor.
Bu nedenle haber, bilanço nedeni, şirket olayı veya fiyat hareketinin sebebi hakkında veri dışı açıklama UYDURMA.

120-180 kelimelik, doğal Türkçe bir "Genel Değerlendirme" yaz.
Dil aşırı resmî olmasın; gevşek de olmasın. Deneyimli bir analistin kendi kısa haftalık notu gibi akıcı olsun.

YAZIM KURALLARI:
- İç sistem/model isimlerini, sürüm kodlarını veya teknik sabitleri ASLA kullanıcıya gösterme.
- "skor hareketi izlendi", "analitik görünüm çeşitlilik gösterdi" gibi boş/genel kalıpları mümkün olduğunca kullanma.
- Somut isim ve sayı kullan: kim kaç gün listede kaldı, kim üst sıralarda kaldı, kim belirgin sıra/skor değişimi yaşadı.
- Aynı hisseyi ve aynı veriyi paragraf içinde tekrar tekrar söyleme.
- Sadece skor/sıra verisine bakarak "şirket güçlendi", "temeller bozuldu" gibi neden-sonuç çıkarma.
- Al/sat, hedef, kaçırılmamalı, yükselir/düşer gibi yatırım yönlendirmesi kullanma.
- Son cümle kısa bir veri sınırlaması/yatırım tavsiyesi değildir notu olabilir.

METODOLOJİ DURUMU:
{methodology_instruction}

HAFTALIK VERİ:
{raw_data}
"""
    for model_name in MODEL_CANDIDATES:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.35),
            )
            return response.text.strip()
        except Exception as e:
            print(f"  {model_name} başarısız: {e}")
            continue
    return None

def main():
    now = datetime.now(TR_TZ)
    # Daily workflow elle çalıştırılsa bile haftalık özet yalnızca Cuma gider.
    if now.weekday() != 4:  # 4 = Cuma
        print("Bugün Cuma değil, haftalık özet atlanıyor.")
        return

    history = load_history()
    entries = this_week_entries(history)
    if not entries:
        print("Bu hafta için kayıt bulunamadı, haftalık özet atlanıyor.")
        return

    agg = build_aggregate(entries)
    if not agg:
        print("Bu hafta hiç hisse kaydı yok, haftalık özet atlanıyor.")
        return

    versions = sorted({
        (item.get("scoring_version") or e.get("scoring_version") or "LEGACY_UNVERSIONED")
        for e in entries for item in e.get("ranking", [])
    })
    methodology_changed = len(versions) > 1
    if methodology_changed:
        # İç sürüm isimleri yalnız logda kalır; kullanıcıya açık rapora yazılmaz.
        print("[HAFTALIK-METODOLOJİ] Hafta içinde birden fazla scoring sürümü kullanıldı: " + ", ".join(versions))
    print(f"Bu hafta {len(entries)} günlük kayıt, {len(agg)} farklı hisse bulundu. Gemini özeti isteniyor...")
    summary_text = summarize_with_gemini(agg, methodology_changed=methodology_changed)
    if methodology_changed and summary_text:
        # Model metodoloji uyarısını unutursa, kullanıcıya teknik kod vermeden tek sade cümle ekle.
        marker_words = ("metodoloji", "yöntem", "karşılaştırılabilir")
        if not any(w in summary_text.lower() for w in marker_words):
            summary_text = (
                "Bu hafta skor hesaplama yönteminde güncelleme yapıldığı için günlük Alpha değişimleri tek başına şirket görünümündeki değişim olarak okunmamalıdır. "
                + summary_text
            )

    header = f"🗓️ HAFTALIK ÖZET ({entries[0]['date']} — {entries[-1]['date']})\n\n"
    if summary_text:
        message = header + summary_text
    else:
        # Gemini başarısız olursa en azından ham tabloyu gönderelim, boş geçmeyelim
        print("Gemini özeti alınamadı, ham veriyle devam ediliyor.")
        message = header + "Gemini özeti oluşturulamadı, bu haftanın ham verileri:\n\n"
        for tk, records in sorted(agg.items(), key=lambda kv: -len(kv[1]))[:15]:
            message += f"{tk}: {len(records)} gün listede\n"

    message += "\n\nBu içerik karşılaştırmalı analiz özetidir; alım-satım çağrısı veya kişiye özel yatırım tavsiyesi içermez."
    try:
        weekly_png = render_weekly_report(entries, agg, summary_text=summary_text)
        send_telegram_photo(
            weekly_png,
            caption=f"BEIQ | Haftalık Analiz Özeti | {entries[0]['date']} — {entries[-1]['date']}\nKarşılaştırmalı analizdir; yatırım tavsiyesi değildir."
        )
        print(f"Haftalık görsel Telegram'a gönderildi: {weekly_png}")
    except Exception as e:
        print(f"[GÖRSEL UYARI] Haftalık görsel üretilemedi/gönderilemedi: {e}")
        send_telegram_message(message)
        print("Haftalık metin fallback raporu gönderildi.")


if __name__ == "__main__":
    main()
