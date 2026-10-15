"""Validated, optional external BTC market series for the daily PDF.
Never infer missing observations, and never treat a failed feed as zero.
"""
import io
import json
import re
from pathlib import Path

import pandas as pd
import requests

ROOT=Path(__file__).parent
OUT=ROOT/"external_data.json"
NOW=pd.Timestamp.now(tz="UTC")
TODAY=NOW.normalize()
HEADERS={"User-Agent":"Mozilla/5.0 (compatible; BBottomWatch/1.0; research report)"}

def fetch_cm():
    """Coin Metrics free community daily endpoint. Realized cap and MVRV ratio are NOT Glassnode metrics."""
    endpoint="https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
    params={"assets":"btc","metrics":"CapMrktCurUSD,CapRealUSD,SplyCur,CapMVRVCur",
            "frequency":"1d","start_time":"2011-01-01","page_size":"10000"}
    values=[]
    with requests.Session() as ss:
        url=endpoint
        for _ in range(6):
            response=ss.get(url,params=params if url==endpoint else None,timeout=45,headers=HEADERS)
            response.raise_for_status()
            payload=response.json()
            values+=payload.get("data",[])
            next_page=payload.get("next_page_url")
            if not next_page:break
            url=next_page
        else:raise ValueError("Too many Coin Metrics pages")
    data=pd.DataFrame(values)
    required={"time","CapMrktCurUSD","CapRealUSD","SplyCur"}
    if not required.issubset(data.columns):raise ValueError("Coin Metrics metrics unavailable: "+str(required-set(data.columns)))
    data["date"]=pd.to_datetime(data["time"],utc=True).dt.normalize()
    for col in ("CapMrktCurUSD","CapRealUSD","SplyCur"):
        data[col]=pd.to_numeric(data[col],errors="coerce")
    data=data.set_index("date").sort_index()
    data=data[data.index<TODAY]
    data=data.dropna(subset=["CapMrktCurUSD","CapRealUSD","SplyCur"])
    data=data[~data.index.duplicated(keep="last")]
    if len(data)<365 or (TODAY-data.index[-1]).days>4:raise ValueError("Coin Metrics missing history or stale")
    if (data[["CapMrktCurUSD","CapRealUSD","SplyCur"]]<=0).any().any():raise ValueError("Invalid Coin Metrics positive values")
    data["realized_price"]=data["CapRealUSD"]/data["SplyCur"]
    data["mvrv_ratio"]=data["CapMrktCurUSD"]/data["CapRealUSD"]
    if (data["mvrv_ratio"]>50).any() or (data["mvrv_ratio"]<=0).any():raise ValueError("MVRV implausible")
    # Proper inception-to-date cumulative historical market-cap std, not a short rolling std.
    # Label methodology explicitly as Coin Metrics-derived, NOT Glassnode's published Z score.
    data["mvrv_z_proxy"]=(data["CapMrktCurUSD"]-data["CapRealUSD"])/data["CapMrktCurUSD"].expanding(min_periods=3650).std(ddof=0)
    return data[["realized_price","mvrv_ratio","mvrv_z_proxy"]].iloc[-1095:]

def parse_money(v):
    x=str(v).strip().replace(",","").replace("$","")
    if not x or x in ("-","—","nan","None"):return None
    if re.fullmatch(r"\\([0-9]+(?:\\.[0-9]+)?\\)",x):x="-"+x[1:-1]
    try:return float(x)
    except ValueError:return None

def fetch_farside():
    url="https://farside.co.uk/bitcoin-etf-flow-all-data/"
    r=requests.get(url,headers=HEADERS,timeout=40)
    r.raise_for_status()
    tables=pd.read_html(io.StringIO(r.text))
    viable=[]
    for table in tables:
        table.columns=[str(v).strip() for v in table.columns]
        if "Date" in table.columns and "Total" in table.columns:
            viable.append(table)
    if not viable:raise ValueError("Farside table with Date/Total headings absent")
    tab=max(viable,key=len).copy()
    tab["date"]=pd.to_datetime(tab["Date"],format="%d %b %Y",errors="coerce",utc=True)
    tab["flow_usdm"]=tab["Total"].map(parse_money)
    tab=tab.dropna(subset=["date","flow_usdm"]).sort_values("date")
    tab=tab[tab.date<=TODAY]
    tab=tab.drop_duplicates("date",keep="last")
    if len(tab)<25 or (TODAY-tab.date.iloc[-1]).days>6:raise ValueError("ETF data stale or too few observations")
    if (tab.flow_usdm.abs()>5000).any():raise ValueError("Implausible daily ETF flows")
    # Farside total is an independent published total; don't silently sum incomplete individual ETF columns.
    return tab.set_index("date")[["flow_usdm"]].iloc[-120:]

def fetch_fear():
    url="https://api.alternative.me/fng/?limit=120&format=json"
    r=requests.get(url,headers=HEADERS,timeout=20);r.raise_for_status()
    d=r.json()["data"]
    rows=[{"date":pd.to_datetime(int(x["timestamp"]),unit="s",utc=True).normalize(),
           "score":int(x["value"])} for x in d]
    tab=pd.DataFrame(rows).drop_duplicates("date").set_index("date").sort_index()
    tab=tab[tab.index<=TODAY]
    if len(tab)<14 or (TODAY-tab.index[-1]).days>2 or not tab.score.between(0,100).all():
        raise ValueError("Fear & Greed missing/stale/out of bounds")
    return tab.iloc[-90:]

def main():
    result={"generated_utc":NOW.isoformat(),"feeds":{}}
    for name,fn,url in (
        ("onchain",fetch_cm,"https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"),
        ("etf",fetch_farside,"https://farside.co.uk/bitcoin-etf-flow-all-data/"),
        ("fear",fetch_fear,"https://api.alternative.me/fng/?limit=120&format=json"),
    ):
        try:
            series=fn()
            result["feeds"][name]={"status":"verified","source":url,
                  "latest_date":str(series.index[-1].date()),"records":len(series),
                  "data":[{"date":str(k.date()),**{c:(round(float(v),7) if pd.notna(v) else None) for c,v in row.items()}}
                          for k,row in series.iterrows()]}
        except Exception as exc:
            result["feeds"][name]={"status":"unavailable","source":url,
                                    "error":str(exc)[:250],"data":[]}
        print(name,result["feeds"][name]["status"],result["feeds"][name].get("error",""))
    OUT.write_text(json.dumps(result,indent=2,allow_nan=False)+"\\n",encoding="utf-8")

if __name__=="__main__":main()
