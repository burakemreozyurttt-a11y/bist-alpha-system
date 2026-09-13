"""
TEK SEFERLİK KEŞİF SCRIPTİ
==========================
Amaç: İş Yatırım'ın fetch_financials fonksiyonunun THYAO için gerçekte hangi
kalem isimlerini (FINANCIAL_ITEM_CODE / FINANCIAL_ITEM_NAME_TR) ve hangi
sütun yapısını döndürdüğünü görmek. Bu bilgiyle scanner.py'deki fundamental
oran hesaplamasını (F/K, ROE, borç/özsermaye vb.) doğru kalem isimleriyle
yazacağız.

Bu dosya sadece TEŞHİS amaçlıdır, tamamlandıktan sonra silinecek.
"""
import pandas as pd
from isyatirimhisse import fetch_financials

pd.set_option("display.max_rows", 300)
pd.set_option("display.max_colwidth", 60)

print("=== financial_group='2' (UFRS) ile THYAO çekiliyor ===")
try:
    df = fetch_financials(symbols="THYAO", start_year=2024, end_year=2025, financial_group='2')
    print("Sütunlar:", df.columns.tolist())
    print("Toplam satır sayısı:", len(df))
    print()
    print("İlk 40 kalem (kod - Türkçe isim):")
    for _, row in df.head(40).iterrows():
        print(f"  {row.get('FINANCIAL_ITEM_CODE')}\t{row.get('FINANCIAL_ITEM_NAME_TR')}")
except Exception as e:
    print(f"UFRS (group 2) başarısız: {e}")

print()
print("=== financial_group='1' (XI_29) ile THYAO çekiliyor (yedek deneme) ===")
try:
    df2 = fetch_financials(symbols="THYAO", start_year=2024, end_year=2025, financial_group='1')
    print("Sütunlar:", df2.columns.tolist())
    print("Toplam satır sayısı:", len(df2))
except Exception as e:
    print(f"XI_29 (group 1) başarısız: {e}")
