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

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

# scanner.py'deki ile aynı yedekleme zinciri
MODEL_CANDIDATES = [
    "gemini-2.5-flash-lite",
    "gemini-3.6-flash-lite",
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


def summarize_with_gemini(agg):
    lines = []
    # En çok gün listede kalanlar en üstte (en "istikrarlı" fırsatlar)
    for tk, records in sorted(agg.items(), key=lambda kv: -len(kv[1])):
        detail = ", ".join(f"{r['date']}: sıra {r['rank']}, alpha {r['alpha_score']}" for r in records)
        lines.append(f"{tk} ({len(records)} gün listede): {detail}")
    raw_data = "\n".join(lines)

    prompt = f"""
Sen bir BIST fon yöneticisisin. Aşağıda bu haftanın (Pazartesi-Cuma) günlük
TOP15 fundamental tarama sonuçları hisse bazında özetlenmiş şekilde
veriliyor (hangi gün kaçıncı sırada, hangi alpha score ile listede
kaldığı). Bu veriye dayanarak yatırımcıya Türkçe, 150-250 kelimelik bir
"Bu Hafta Ne Oldu" özeti yaz. En istikrarlı şekilde listede kalan
hisseleri, alpha score'u en çok yükselen/düşen hisseleri ve haftanın genel
görünümünü vurgula. SADECE verilen veriye dayan; bilmediğin bir haberi,
KAP açıklamasını ya da gelişmeyi ASLA uydurma — sadece sıra/skor
değişimlerinden çıkarılabilecek gözlemleri paylaş.

VERİ:
{raw_data}
"""
    for model_name in MODEL_CANDIDATES:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.4),
            )
            return response.text.strip()
        except Exception as e:
            print(f"  {model_name} başarısız: {e}")
            continue
    return None


def main():
    now = datetime.now(TR_TZ)
    is_manual_run = os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch"
    if now.weekday() != 4 and not is_manual_run:  # 4 = Cuma
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

    print(f"Bu hafta {len(entries)} günlük kayıt, {len(agg)} farklı hisse bulundu. Gemini özeti isteniyor...")
    summary_text = summarize_with_gemini(agg)

    header = f"🗓️ HAFTALIK ÖZET ({entries[0]['date']} — {entries[-1]['date']})\n\n"
    if summary_text:
        message = header + summary_text
    else:
        # Gemini başarısız olursa en azından ham tabloyu gönderelim, boş geçmeyelim
        print("Gemini özeti alınamadı, ham veriyle devam ediliyor.")
        message = header + "Gemini özeti oluşturulamadı, bu haftanın ham verileri:\n\n"
        for tk, records in sorted(agg.items(), key=lambda kv: -len(kv[1]))[:15]:
            message += f"{tk}: {len(records)} gün listede\n"

    message += "\n\nBu içerik yatırım tavsiyesi değildir."
    send_telegram_message(message)
    print("Haftalık özet gönderildi.")


if __name__ == "__main__":
    main()
