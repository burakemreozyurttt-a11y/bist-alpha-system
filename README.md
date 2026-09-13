# BIST Fundamental Alpha System — Faz 1 (MVP)

## Dosyalar
- `scanner.py` — ana tarama scripti
- `requirements.txt` — Python bağımlılıkları
- `bist_tickers_fallback.csv` — KAP listesi çekilemezse kullanılan yedek hisse kodu listesi
- `state.json` — bir önceki günün TOP 15 sıralamasını tutar (otomatik güncellenir)
- `.github/workflows/daily_scan.yml` — hafta içi 18:30 TSİ'de otomatik çalıştırma ayarı

## Elle test etmek istersen (opsiyonel)
```bash
pip install -r requirements.txt
export GEMINI_API_KEY="..."
export TELEGRAM_BOT_TOKEN="..."
export TELEGRAM_CHAT_ID="..."
python scanner.py
```

## Notlar
- Bu Faz 1 (MVP) sürümüdür. Sector Router, Multi-Agent (Bull/Bear/CRO),
  Thesis Tracking ve RAG modülleri sonraki fazlarda eklenecek.
- Yatırım tavsiyesi değildir.
