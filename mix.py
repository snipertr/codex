import os
import requests, json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, inspect, MetaData, Table, Column, text
from sqlalchemy.dialects.mysql import DATE, FLOAT, VARCHAR, INTEGER
from pathlib import Path
from io import StringIO
from bs4 import BeautifulSoup

# Veritabanı ve API bilgileri
DATABASE_URL = "mysql+pymysql://root:root@localhost/fiyatlar"
API_URL = "https://www.isyatirim.com.tr/_layouts/15/Isyatirim.Website/Common/Data.aspx/HisseTekil?"
CACHE_PATH = Path("cache")
CACHE_PATH.mkdir(exist_ok=True)

# Yardımcı fonksiyonlar

def get_cached_api_data(hisse: str, start: str, end: str) -> pd.DataFrame:
    pd.set_option("future.no_silent_downcasting", True)
    cache_file = CACHE_PATH / f"{hisse}_{start}_{end}.json"
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
    else:
        params = {"hisse": hisse, "startdate": start, "enddate": end}
        raw = requests.get(API_URL, params=params).json().get("value", [])
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)

    df = pd.DataFrame.from_dict(raw)
    if df.empty or "HGDG_TARIH" not in df:
        return pd.DataFrame()

    df["HGDG_TARIH"] = pd.to_datetime(df["HGDG_TARIH"], format="%d-%m-%Y")
    df.set_index("HGDG_TARIH", inplace=True)
    df.index.name = "tarih"
    df = df.sort_index()

    df = df.rename(columns={
        "HGDG_KAPANIS": "KAPANIŞ",
        "HGDG_MAX": "HIGH",
        "HGDG_MIN": "LOW",
        "HGDG_HACIM": "HACIM",
        "DOLAR_BAZLI_FIYAT": "CUSD"
    })

    df = df.replace([pd.NA, pd.NaT, float("inf"), -float("inf")], 0).fillna(0)
    df = df.infer_objects(copy=False)

    try:
        data1 = pd.DataFrame({
            "KAPANIŞ": pd.to_numeric(df["KAPANIŞ"], errors="raise").round(2),
            "HIGH": pd.to_numeric(df["HIGH"], errors="raise").round(2),
            "LOW": pd.to_numeric(df["LOW"], errors="raise").round(2),
            "HACIM": pd.to_numeric(df["HACIM"], errors="raise").astype(int),
            "CUSD": pd.to_numeric(df["CUSD"], errors="raise").round(2),
        })
        print(data1)
    except Exception as e:
        print(f"Dönüşüm hatası: {e}")
        return pd.DataFrame()

    return data1

def get_table_data(hisse: str, engine) -> pd.DataFrame:
    query = text(f"SELECT tarih, `KAPANIŞ` FROM `{hisse.lower()}` ORDER BY tarih ASC")
    return pd.read_sql(query, engine, parse_dates=["tarih"]).set_index("tarih")

def update_stock_table(hisse: str, engine):
    print(f"\nHisse: {hisse}")
    df_db = get_table_data(hisse, engine)
    if df_db.empty:
        print("Tablo boş, tamamen indiriliyor.")
        start_date = "01-01-2008"
        end_date = pd.Timestamp.today().strftime("%d-%m-%Y")
        df_api = get_cached_api_data(hisse, start_date, end_date)
        if not df_api.empty:
            df_api.reset_index(inplace=True)
            df_api.to_sql(hisse.lower(), engine, if_exists="replace", index=False, method="multi")
            print(f"{len(df_api)} kayıt eklendi.")
        else:
            print("API verisi boş.")
        return

    first_date = df_db.index[0]
    last_date = df_db.index[-1]
    try:
        local_first_price = df_db.loc[first_date, "KAPANIŞ"].values[0]
    except:
        local_first_price = df_db.loc[first_date, "KAPANIŞ"]

    df_api = get_cached_api_data(hisse, first_date.strftime("%d-%m-%Y"), first_date.strftime("%d-%m-%Y"))

    try:
        api_first_price = df_api.loc[first_date, "KAPANIŞ"]
    except KeyError:
        print(f"API verisi ilk tarih ({first_date}) için mevcut değil.")
        return

    if isinstance(api_first_price, pd.Series):
        api_first_price = api_first_price.iloc[0]

    if abs(api_first_price - local_first_price) < 1e-4:
        print("Veriler uyumlu. Eksik günler tamamlanacak.")
        start = last_date + pd.Timedelta(days=1)
    else:
        print("Veriler uyumsuz. Tüm veriler yenilenecek.")
        start = first_date

    today = pd.Timestamp.today()
    end = today if today.weekday() < 5 else pd.date_range(end=today, periods=1, freq="B")[0]

    date_range = pd.date_range(start=start, end=end, freq="B")
    if not date_range.empty:
        df_new = get_cached_api_data(hisse, date_range[0].strftime("%d-%m-%Y"), date_range[-1].strftime("%d-%m-%Y"))
        df_new = df_new.loc[df_new.index.isin(date_range)]
        if not df_new.empty:
            df_new.reset_index(inplace=True)
            df_new.to_sql(hisse.lower(), engine, if_exists="append", index=False, method="multi")
            print(f"{len(df_new)} yeni veri eklendi.")
        else:
            print("API veri dönmedi.")
    else:
        print("Eklenecek yeni gün yok.")

def clean_cache():
    for file in CACHE_PATH.glob("*.json"):
        os.remove(file)

def get_si_data(hisse_kodu):
    print(f"\nBIST100 endeks değerleri için API'den veri alınıyor...")
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT MAX(tarih) FROM usdtry_xu100_c"))
        last_date = result.scalar()

    startdate = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=1, freq='B')[0].strftime('%d-%m-%Y')
    simdi = pd.Timestamp.now()
    if simdi.hour < 19:
        enddate = pd.date_range(end=simdi - pd.Timedelta(days=1), periods=1, freq='B')[0].strftime('%d-%m-%Y')
    else:
        enddate = pd.date_range(end=simdi, periods=1, freq='B')[0].strftime('%d-%m-%Y')

    parametreler = {"hisse": hisse_kodu, "startdate": startdate, "enddate": enddate}
    url2 = API_URL

    try:
        response = requests.get(url2, params=parametreler, timeout=10)
        response.raise_for_status()
        r2 = response.json().get("value", [])
        if not r2:
            print(f"BIST100 endeks için verilerin güncel.")
            return

        data = pd.DataFrame.from_dict(r2)
        data["HGDG_TARIH"] = pd.to_datetime(data["HGDG_TARIH"], format='%d-%m-%Y')
        data.rename(columns={"HGDG_TARIH": "tarih", "END_DEGER": "XU100", "DD_DEGER": "USDTRY"}, inplace=True)
        data["XU100"] = data["XU100"].replace(-np.inf, np.nan).bfill()
        data["USDTRY"] = data["USDTRY"].replace(-np.inf, np.nan).bfill()
        data = data.sort_values("tarih")
        veri = data[["tarih", "XU100", "USDTRY"]].astype({"XU100": float, "USDTRY": float})

        print(f"\nBIST100 endeks için alınan API verisi:")
        print(veri)

        if not veri.empty:
            veri.to_sql(name="usdtry_xu100_c", con=engine, if_exists="append", index=False, method="multi")
            print(f"{len(veri)} kayıt veritabanına eklendi.")

    except Exception as e:
        print(f"{hisse_kodu} için API verisi alınamadı: {e}")


def create_missing_stock_tables(engine_url, eksik_hisseler):
    engine = create_engine(engine_url)
    metadata = MetaData()
    for hisse in eksik_hisseler:
        tablo_adi = hisse.lower()
        tablo = Table(
            tablo_adi,
            metadata,
            Column('ID', INTEGER, primary_key=True, autoincrement=True),
            Column('tarih', DATE),
            Column('KAPANIS', FLOAT(precision=15, scale=2)),
            Column('HIGH', FLOAT(precision=15, scale=2)),
            Column('LOW', FLOAT(precision=15, scale=2)),
            Column('HACIM', VARCHAR(50)),
            Column('CUSD', FLOAT(precision=10, scale=2))
        )
        tablo.create(bind=engine, checkfirst=True)
        print(f"{tablo_adi} tablosu oluşturuldu.")

def odenmis_sermaye_csv():
    try:
        url = "https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/Temel-Degerler-Ve-Oranlar.aspx#page-2"
        r = requests.get(url)
        soup = BeautifulSoup(r.text, "html.parser")
        table = soup.find("table", {"id": "summaryBasicData"})
        if not table:
            print("Hedef tablo bulunamadı.")
            return

        df = pd.read_html(StringIO(str(table)))[0]
        df.rename(columns={"Kod": "Hisse", "Sermaye (mn TL)": "Sermaye"}, inplace=True)
        df = df[["Hisse", "Sermaye"]]
        df.set_index("Hisse", inplace=True)
        df["Sermaye"] = df["Sermaye"].str.replace("[.,]", "", regex=True).astype(int) * 10000
        csv_path = 'C:/Users/enish/OneDrive/Belgeler/Takaslar/ode_ser.csv'
        df.to_csv(csv_path, sep=";", index=True, header=True)
        print(f"Ödenmiş sermaye verisi kaydedildi: {csv_path}")

    except Exception as e:
        print(f"Hata oluştu: {e}")

def sync_preview():
    engine = create_engine(DATABASE_URL)
    inspector = inspect(engine)
    tablolari_al = inspector.get_table_names()
    haric = {"enflasyon_oranlari", "usdtry_xu100_c", "stock_names"}
    hisseler = [t.upper() for t in tablolari_al if t not in haric]
    for hisse in hisseler:
        try:
            update_stock_table(hisse, engine)
        except Exception as e:
            print(f"{hisse} için hata: {e}")
    clean_cache()

# === Yabancı Oranı Güncelleme Başlangıcı ===
def update_yabanci_orani():
    engine_yabanci = create_engine("mysql+pymysql://root:root@localhost/yabanci_orani")
    inspector_yabanci = inspect(engine_yabanci)

    now = datetime.now()
    if now.weekday() >= 5:
        hedef_tarih = now - timedelta(days=(now.weekday() - 4 if now.weekday() == 5 else 2))
    elif now.hour < 19:
        hedef_tarih = now - timedelta(days=1)
        while hedef_tarih.weekday() >= 5:
            hedef_tarih -= timedelta(days=1)
    else:
        hedef_tarih = now

    hedef_tarih = hedef_tarih.replace(hour=0, minute=0, second=0, microsecond=0)
    print(f"Hedef Tarih: {hedef_tarih.strftime('%d-%m-%Y')}")

    all_tables_yabanci = inspector_yabanci.get_table_names()
    exclude_tables = ["almad", "grtrk", "karye", "mipaz", "mtryo", "qnbfb", "qnbfl", "tetmt"]
    tables_yabanci = [t for t in all_tables_yabanci if t.lower() not in exclude_tables]

    headers = {'Content-type': 'application/json'}
    yabanci_url = "https://www.isyatirim.com.tr/_layouts/15/IsYatirim.Website/StockInfo/CompanyInfoAjax.aspx/GetYabanciOranlarXHR"
    hatali_tablolar = []

    for table in tables_yabanci:
        try:
            query = f"SELECT MAX(Tarih) as max_date FROM {table}"
            df = pd.read_sql(text(query), engine_yabanci)
            max_date_raw = df.loc[0, "max_date"]

            if pd.isna(max_date_raw):
                raise ValueError("max_date is None")

            max_date = pd.to_datetime(max_date_raw)

            if max_date.normalize() != hedef_tarih:
                print(f"{table} tablosunda son kayıt tarihi: {max_date.strftime('%d-%m-%Y')}")
                eksik_gunler = pd.date_range(start=max_date + timedelta(days=1), end=hedef_tarih, freq='B')
                print(f"{table} tablosu için eksik günler:")
                print(eksik_gunler.strftime('%d-%m-%Y').tolist())

                veri_eklenecek = []

                for tarih in eksik_gunler:
                    tarih_str = tarih.strftime("%d-%m-%Y")
                    parametreler = {
                        "baslangicTarih": tarih_str,
                        "bitisTarihi": tarih_str,
                        "sektor": "",
                        "endeks": "09",
                        "hisse": ""
                    }

                    try:
                        response = requests.post(url=yabanci_url, json=parametreler, headers=headers)
                        response.raise_for_status()
                        raw_json = response.json()
                        reff = pd.json_normalize(json.loads(json.dumps(raw_json)))
                        inner_data = reff["d"].iloc[0]
                        data = pd.DataFrame.from_dict(inner_data)
                        data.drop(columns=["__type", "PRICE_TL", "HISSE_TANIM", "HISSE_TANIM_YD", "ETKI", "DEGISIM", "YAB_ORAN_END"], inplace=True, errors='ignore')
                        data.rename(columns={"YAB_ORAN_START": "Oran"}, inplace=True)
                        data = data[["HISSE_KODU", "Oran"]]
                        data.insert(0, "Tarih", tarih.strftime("%Y-%m-%d"))

                        ilgili = data[data["HISSE_KODU"] == table.upper()]
                        if not ilgili.empty:
                            oran = float(ilgili["Oran"].values[0])
                            veri_eklenecek.append({"Tarih": tarih.strftime("%Y-%m-%d"), "Oran": oran})
                    except Exception as e:
                        print(f"{table} - {tarih_str} için kaynakta veri yok.\n")

                if veri_eklenecek:
                    df_ekle = pd.DataFrame(veri_eklenecek)
                    df_ekle["Tarih"] = pd.to_datetime(df_ekle["Tarih"])
                    df_ekle.to_sql(name=table, con=engine_yabanci, if_exists="append", index=False, method="multi")
                    print(f"{table} tablosuna {len(veri_eklenecek)} kayıt eklendi.\n")

        except Exception as e:
            print(f"{table} tablosunda hata oluştu: {e}")
            hatali_tablolar.append(table)

    if hatali_tablolar:
        print("\nİşlenemeyen tablolar:")
        print(hatali_tablolar)

def hisseler_ana():
    url = "https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/Temel-Degerler-Ve-Oranlar.aspx?endeks=09#page-1"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "tr-TR,tr;q=0.9"
    }

    with requests.Session() as session:
        response = session.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")

    hisse_kodlari = [a_tag.text.strip().upper() for tr in soup.find_all("tr") if (a_tag := tr.find("a"))]
    engine_fiyatlar = create_engine(DATABASE_URL)
    inspector = inspect(engine_fiyatlar)
    haric_tutulanlar = {"enflasyon_oranlari", "usdtry_xu100_c", "stock_names"}
    tum_tablolar = inspector.get_table_names()
    filtrelenmis_tablolar = [t.upper() for t in tum_tablolar if t not in haric_tutulanlar]
    eksik_hisseler = [kod for kod in hisse_kodlari if kod not in filtrelenmis_tablolar]

    if eksik_hisseler:
        print("\nYeni hisseler:")
        print(eksik_hisseler)
        create_missing_stock_tables(DATABASE_URL, eksik_hisseler)
    else:
        print("Yeni hisse yoktur.")

def main():
    odenmis_sermaye_csv()
    get_si_data("TUPRS") #XU100 ve USD fiyatlarını almak için.
    hisseler_ana()  #Olmayan hisselerin tablo oluşumu.
    sync_preview()  #Günlük veri kaydı.
    update_yabanci_orani()


if __name__ == "__main__":
    main()

