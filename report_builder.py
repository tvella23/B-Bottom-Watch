"""Generate a dated BTC Bottom Watch PDF from quality-checked public inputs.
Never invent on-chain or demand data; plot only dated observed series.
"""
from __future__ import annotations
import datetime as dt
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, PageBreak, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether

ROOT=Path(__file__).parent
REPORT=ROOT/'Bitcoin_Bottom_Watch_Latest.pdf'
SNAP=json.loads((ROOT/'latest.json').read_text(encoding='utf-8'))
DF=pd.read_csv(ROOT/'history.csv', parse_dates=['date']).sort_values('date')
DF['date']=pd.to_datetime(DF['date'],utc=True)
DF=DF.drop_duplicates('date').set_index('date')
DF['close']=pd.to_numeric(DF['close'])
C=DF['close']
OBS=SNAP['last_completed_candles']
assert str(C.index[-1].date())==OBS['day'],'History does not match indicator snapshot'
assert len(C)>=900,'Insufficient historic daily observations'
assert len(pd.date_range(C.index[0],C.index[-1],freq='D').difference(C.index))==0,'History has missing days'
AS_OF=pd.Timestamp(SNAP['generated_utc']).strftime('%d %B %Y %H:%M UTC')
TODAY=pd.Timestamp(OBS['day'],tz='UTC')
assert (pd.Timestamp.now(tz='UTC')-TODAY).days<=2,'Data stale; refuse PDF publication'
W=C.resample('W-SUN').last(); W=W[W.index<=pd.Timestamp(OBS['week_ending_sunday'],tz='UTC')]
M=C.resample('ME').last(); M=M[M.index<=pd.Timestamp(OBS['month_ending'],tz='UTC')]

BLUE='#15334A'; TEAL='#087A75'; GOLD='#B27C32'; GREY='#54697A'; RED='#B0524F'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.16,'savefig.facecolor':'white'})
CHARTS=ROOT/'report_charts';CHARTS.mkdir(exist_ok=True)

def chart(name, fn):
    fig, ax=plt.subplots(figsize=(10,4.2),dpi=140);fn(ax);fig.tight_layout(pad=1.6)
    path=CHARTS/name;fig.savefig(path,dpi=150,bbox_inches='tight');plt.close(fig)
    return path

# Use exactly the GitHub-calculated current observations. Chart calculations use same formulas.
def monthly_rsi(prices,n=14):
    dif=prices.diff();g=dif.clip(lower=0);l=-dif.clip(upper=0)
    out=pd.Series(np.nan,index=prices.index)
    ga=g.iloc[1:n+1].mean();la=l.iloc[1:n+1].mean()
    def score(a,b):return 100.0 if b==0 and a>0 else 50.0 if b==0 else 100-100/(1+a/b)
    out.iloc[n]=score(ga,la)
    for i in range(n+1,len(prices)):
        ga=(ga*(n-1)+g.iloc[i])/n;la=(la*(n-1)+l.iloc[i])/n
        out.iloc[i]=score(ga,la)
    return out
R=monthly_rsi(M);RS=R.rolling(14).mean()
assert abs(float(R.iloc[-1])-SNAP['monthly_rsi']['rsi_14'])<.015,'RSI mismatch'
assert abs(float(RS.iloc[-1])-SNAP['monthly_rsi']['sma_14_of_rsi'])<.015,'RSI moving-average mismatch'
PER=(20,25,30,35,40,45,50,55)
EV={p:W.ewm(span=p,adjust=False).mean() for p in PER}
assert all(abs(EV[p].iloc[-1]-SNAP['weekly_ema_ribbon']['ema_values'][str(p)])<.1 for p in PER),'EMA mismatch'
SHORT=C.ewm(span=150,adjust=False).mean(); LONG=C.rolling(471).mean()*.745
assert abs(float(SHORT.iloc[-1])-SNAP['pi_cycle_bottom']['ema_150'])<.1,'Pi EMA mismatch'
assert abs(float(LONG.iloc[-1])-SNAP['pi_cycle_bottom']['sma_471_times_0_745'])<.1,'Pi SMA mismatch'

P_PRICE=chart('price.png',lambda ax:(ax.plot(C.loc['2024':].index,C.loc['2024':].values,color=BLUE,lw=1.4),ax.set_title('BTC/USD | Daily close'),ax.set_ylabel('USD')))
def rsi_plot(ax):
    r=R.iloc[-72:];s=RS.iloc[-72:];ax.plot(r.index,r,label='Monthly RSI(14)',color=TEAL,lw=1.9);ax.plot(s.index,s,label='14-month SMA of RSI',color=GOLD,lw=1.7)
    ax.axhline(50,color=GREY,lw=.8,linestyle='--');ax.set_ylim(10,90);ax.set_ylabel('RSI points');ax.set_title('Monthly momentum | Completed calendar months');ax.legend(fontsize=8)
P_RSI=chart('monthly_rsi.png',rsi_plot)
def ema_plot(ax):
    ax.plot(W.iloc[-80:].index,W.iloc[-80:],label='Weekly close',color=BLUE,lw=2)
    for p in PER:ax.plot(EV[p].iloc[-80:].index,EV[p].iloc[-80:],label=f'EMA {p}',lw=1.1)
    ax.set_title('Weekly trend | Eight EMA ribbon');ax.set_ylabel('USD');ax.legend(ncol=5,fontsize=7,loc='upper left')
P_EMA=chart('weekly_ema.png',ema_plot)
def pi_plot(ax):
    st='2016-01-01';ax.plot(C.loc[st:].index,C.loc[st:],color='#B6C0C7',lw=.6,label='BTC close')
    ax.plot(SHORT.loc[st:].index,SHORT.loc[st:],color=TEAL,lw=1.5,label='150-day EMA')
    ax.plot(LONG.loc[st:].index,LONG.loc[st:],color=GOLD,lw=1.5,label='471-day SMA x 0.745')
    crossed=(LONG>SHORT)&(LONG.shift(1)<=SHORT.shift(1))
    for ts in crossed[crossed].loc[st:].index:ax.axvline(ts,color=RED,alpha=.5,ls='--',lw=1)
    ax.set_title('Pi Cycle Bottom | Historical crossover markers');ax.set_ylabel('USD');ax.legend(fontsize=8)
P_PI=chart('pi_cycle.png',pi_plot)

# Sentiment: only plot results read directly from the primary Alternative.me API; no estimated values.
FEAR=None;FEAR_SOURCE='https://api.alternative.me/fng/?limit=35&format=json';FEAR_STATUS='Not verified / unavailable'
try:
    res=requests.get(FEAR_SOURCE,timeout=12);res.raise_for_status(); data=res.json()['data']
    series=pd.Series({pd.to_datetime(int(x['timestamp']),unit='s',utc=True).normalize():int(x['value']) for x in data}).sort_index()
    series=series[(series.index<=TODAY+pd.Timedelta(days=1)) & (series.index>=TODAY-pd.Timedelta(days=35))]
    assert len(series)>=10 and 0<=series.iloc[-1]<=100
    assert (TODAY-series.index[-1]).days<=2
    FEAR=series;FEAR_STATUS='VERIFIED (primary API, UTC daily records)'
except Exception as exc:
    FEAR_STATUS='UNAVAILABLE - primary-source daily data could not be verified'
P_FEAR=None
if FEAR is not None:
    def fear_plot(ax):
        ax.axhspan(0,24,color=RED,alpha=.1);ax.axhspan(25,44,color=GOLD,alpha=.08);ax.axhspan(45,55,color=GREY,alpha=.09);ax.axhspan(56,74,color=TEAL,alpha=.08);ax.axhspan(75,100,color=TEAL,alpha=.13)
        ax.plot(FEAR.index,FEAR.values,color=TEAL,marker='o',ms=3,lw=1.6);ax.set_ylim(0,100);ax.set_title('Fear & Greed | Primary-source daily history');ax.set_ylabel('Score 0-100')
    P_FEAR=chart('fear_greed.png',fear_plot)

# ETF flow timeseries remains withheld until current dated Farside figures are fetched/parsed and validated.
# On-chain MVRV/realized-cost metrics are withheld without synchronized, reliable history.

ss=getSampleStyleSheet()
ss.add(ParagraphStyle(name='CoverKicker',fontName='Helvetica-Bold',fontSize=12,leading=18,textColor=colors.HexColor(TEAL),spaceAfter=12))
ss.add(ParagraphStyle(name='CoverTitle',fontName='Helvetica-Bold',fontSize=33,leading=39,textColor=colors.HexColor(BLUE),spaceAfter=20))
ss.add(ParagraphStyle(name='CoverSub',fontName='Helvetica',fontSize=13,leading=21,textColor=colors.HexColor(GREY)))
ss.add(ParagraphStyle(name='Section',fontName='Helvetica-Bold',fontSize=18,leading=24,spaceAfter=14,textColor=colors.HexColor(BLUE)))
ss.add(ParagraphStyle(name='Header',fontName='Helvetica-Bold',fontSize=11,leading=16,spaceBefore=8,spaceAfter=6,textColor=colors.HexColor(BLUE)))
ss.add(ParagraphStyle(name='BodyTextReport',fontName='Helvetica',fontSize=9.2,leading=14.5,spaceAfter=9,textColor=colors.HexColor('#273747')))
ss.add(ParagraphStyle(name='SmallReport',fontName='Helvetica',fontSize=7.6,leading=11.5,spaceAfter=7,textColor=colors.HexColor(GREY)))
ss.add(ParagraphStyle(name='Verdict',fontName='Helvetica-Bold',fontSize=10.5,leading=16,spaceAfter=12,textColor=colors.HexColor(TEAL)))

def para(text,sty='BodyTextReport'):return Paragraph(text,ss[sty])
def img(path):
    from PIL import Image as PILImage
    im=PILImage.open(path);a=184*mm
    return Image(str(path),width=a,height=a*im.height/im.width)
def section(title,body):
    story.append(para(title,'Section'));story.extend(body);story.append(PageBreak())
def textblock(h,s):return [para(h,'Header'),para(s)]
def tab(rows):
    t=Table([[para(str(v),'SmallReport') for v in row] for row in rows],colWidths=[43*mm,135*mm],hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E7F0F0')),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#B8C9D0'))]));return t

m=SNAP['monthly_rsi'];e=SNAP['weekly_ema_ribbon'];pi=SNAP['pi_cycle_bottom']
price=float(C.iloc[-1]);rsi_bull=m['bullish_crossover_last_month']
ema_bull=e['fully_bullish_ordered'];weekly_above=e['close_above_all']
pi_signal=pi['crossed_today']

story=[]
# Cover page intentionally contains only decision-oriented research framing, not a change log.
story += [Spacer(1,35*mm),para('INDEPENDENT MARKET MONITORING','CoverKicker'),para('BITCOIN<br/>BOTTOM WATCH','CoverTitle'),para('Evidence-family chart report','CoverSub'),Spacer(1,30*mm),para(f'<b>Market observations through {OBS["day"]} (UTC)</b>','CoverSub'),para(f'Generated {AS_OF}','CoverSub'),Spacer(1,25*mm)]
story += [para('ON-CHAIN VALUATION  /  CYCLE STRUCTURE  /  DEMAND & LIQUIDITY  /  PRICE MOMENTUM  /  SENTIMENT','SmallReport'),Spacer(1,23*mm),para('Research note: Indicator confirmation is not a prediction or a trading authorization. Missing independent data reduces confidence in bottom and early-bull inferences.','SmallReport'),PageBreak()]

rows=[['Evidence family','Bottom / early-bull interpretation'],['On-chain valuation','UNVERIFIED: no defensible synchronized MVRV, realized price or holder metrics'],['Cycle structure','Pi Cycle Bottom: '+('new crossover' if pi_signal else 'no new crossover')+'; absent signal does not rule out bottom'],['Demand / liquidity','UNVERIFIED: no validated latest ETF-flow time series / stablecoin demand'],['Price / momentum',('Confirmed monthly RSI bullish crossover; ' if rsi_bull else 'No new monthly RSI bullish crossover; ')+('bullish EMA ribbon' if ema_bull else 'EMA ribbon not fully ordered')],['Sentiment',f'{int(FEAR.iloc[-1])} / {"Fear" if FEAR.iloc[-1]<45 else "Neutral" if FEAR.iloc[-1]<=55 else "Greed"} (overlay only)' if FEAR is not None else 'UNVERIFIED: latest Fear & Greed unavailable']]
section('Executive evidence summary',[
    para('This assessment differentiates evidence supporting a completed cycle bottom from evidence supporting an emerging sustained bull market. Technical momentum is ONE evidence family, not multiple independent confirmations.'),
    tab(rows),Spacer(1,7*mm),
    para(f'Observed BTC/USD daily close: <b>US${price:,.2f}</b> on <b>{OBS["day"]}</b>. Last completed week: {OBS["week_ending_sunday"]}; last completed month: {OBS["month_ending"]}.'),
    para('<b>Overall assessment:</b> Price/momentum evidence is constructive, but on-chain valuation and sustained demand cannot be assessed with sufficient confidence. Neither a definitive completed bottom nor an early bull market is independently confirmed by this report.'),
    para('<b>Tranche discipline:</b> No additional borrowed-funds tranche is justified by technical indicators alone. Bottom/bull probabilities are not quantified without an adequate independent evidence base.'),
    para('Data: '+AS_OF+'. Source: Bitstamp daily OHLC published and validated by GitHub collector.','SmallReport')])

section('Family 1 | On-chain valuation',[
    para('STATUS: NOT VERIFIED - no numerical MVRV or realized-price chart has been published.' ,'Verdict'),
    para('MVRV Z-Score, realized price, realized profit/loss and long-term-holder activity require dated, internally consistent on-chain observations. No reliable synchronized price/valuation series is available in this automated report.'),
    *textblock('How to interpret the missing indicators','MVRV Z-Score compares market value with realized cost basis after volatility normalization. Lower readings have historically accompanied more depressed valuation; recovery from low levels can support bottom formation. Realized price estimates holders’ on-chain cost basis. Accumulation, capitulation and Sell-Side Risk help distinguish a true cycle reset from temporary rallies.'),
    *textblock('Bottom or early-bull assessment','<b>INCONCLUSIVE.</b> Without independently verified valuation and holder activity, momentum strength cannot establish that Bitcoin is attractively priced or that long-term holders are accumulating.'),
    img(P_PRICE),para('Chart: verified market-price context ONLY, not a proxy for MVRV or realized price. Source: Bitstamp daily OHLC.','SmallReport')])

section('Family 2 | Cycle / structural',[
    img(P_PI),
    *textblock('How to read this chart','The green curve is the 150-day EMA, and gold is 0.745 x the 471-day SMA. A Pi Cycle Bottom event is recorded only when the adjusted long average crosses ABOVE the shorter EMA. Dashed markers show historical occurrences; they are not forward predictions.'),
    para(f'Latest verified values ({OBS["day"]}): 150-day EMA <b>US${pi["ema_150"]:,.0f}</b>; adjusted 471-day SMA <b>US${pi["sma_471_times_0_745"]:,.0f}</b>. New bottom crossover: <b>{"YES" if pi_signal else "NO"}</b>.'),
    *textblock('Bottom or early-bull assessment',('A verified new Pi crossover supports a historical-cycle bottom hypothesis, but demands independent confirmation.' if pi_signal else 'No new Pi Bottom crossover. This does not deny a completed bottom: the indicator is episodic, can lag, and may not trigger during every constructive market regime. It offers no fresh bottom confirmation today.')),
    para('Source: Bitstamp standard BTCUSD daily closes, 150-day EMA and 471-day SMA x 0.745; historical dates and latest values validated to published JSON.','SmallReport')])

section('Family 3 | Demand / liquidity',[
    para('STATUS: INSUFFICIENT CURRENT DATA - no ETF-flow or stablecoin chart has been fabricated.','Verdict'),
    *textblock('How to read demand charts','Persistent positive net US spot ETF flows can indicate fresh capital entering Bitcoin; persistent outflows can contradict a new bull thesis. Confirm alongside spot trading volume, stablecoin growth, exchange flows and evidence that rallies are not primarily derivatives short covering.'),
    *textblock('Bottom or early-bull assessment','<b>INCONCLUSIVE.</b> Without a fresh, adequately long and verified net-flow series, positive momentum cannot be attributed to new spot demand. This family must not be counted as confirming an early bull market.'),
    para('Source to verify in a future expanded data collector: Farside Investors, https://farside.co.uk/btc/ ; data NOT inferred from other price measures.','SmallReport')])

section('Family 4 | Price and momentum',[
    img(P_RSI),
    *textblock('How to read monthly RSI','The teal line is 14-period Wilder RSI on monthly BTC closing prices; gold is the 14-month simple average of RSI. A confirmed monthly bullish crossover requires the completed RSI bar to close above the RSI average after previously being at or below it. A crossover supports momentum recovery, not necessarily deep valuation.'),
    para(f'Completed month {OBS["month_ending"]}: RSI <b>{m["rsi_14"]:.2f}</b> vs RSI-SMA <b>{m["sma_14_of_rsi"]:.2f}</b>; bullish crossover: <b>{"YES" if rsi_bull else "NO"}</b>.'),
    *textblock('Bottom or early-bull assessment','Momentum recovery is '+('supported by a completed monthly RSI crossover.' if rsi_bull else 'not newly confirmed by an RSI crossover.')+' This improves the technical case for an early bull, but cannot independently confirm the cycle bottom.'),
    para('Source: Bitstamp standard daily closes, completed UTC months only; latest output cross-checked to GitHub collector.','SmallReport')])

section('Family 4 | Weekly trend confirmation',[
    img(P_EMA),
    *textblock('How to read the EMA ribbon','A strictly bullish ordering requires EMA20 > EMA25 > EMA30 > EMA35 > EMA40 > EMA45 > EMA50 > EMA55. BTC closing above every EMA is constructive but insufficient for a fully ordered bullish ribbon. Watch slope, persistence and widening separation.'),
    para(f'Last completed weekly close ({OBS["week_ending_sunday"]}): <b>US${e["completed_week_close"]:,.2f}</b>. Above all eight averages: <b>{"YES" if weekly_above else "NO"}</b>. Fully bullish ordering: <b>{"YES" if ema_bull else "NO"}</b>.'),
    *textblock('Bottom or early-bull assessment',('Fully ordered weekly EMAs support strengthening medium-term uptrend.' if ema_bull else 'Price above the weekly EMA ribbon supports improving trend, but incomplete EMA ordering means the stronger structural confirmation has not yet arrived.')+' Treat this together with monthly RSI as ONE momentum family.'),
    para('Source: Bitstamp standard BTCUSD daily prices aggregated into fully completed UTC Sunday-ending weeks.','SmallReport')])

sent=[para('Fear & Greed is a contrarian context indicator, not a standalone bottom confirmation. Extremely fearful conditions can accompany capitulation; persistent high greed can raise the risk of chasing a mature rally.')]
if P_FEAR:
    val=int(FEAR.iloc[-1]);label='Extreme fear' if val<=24 else 'Fear' if val<=44 else 'Neutral' if val<=55 else 'Greed' if val<=74 else 'Extreme greed'
    sent=[img(P_FEAR),para(f'Latest verified value: <b>{val}/100 ({label})</b>, {FEAR.index[-1].date()}. Source: Alternative.me primary API.'),*textblock('How to read it','Shaded bands correspond to extreme fear (0–24), fear (25–44), neutral (45–55), greed (56–74) and extreme greed (75–100). Look for multiweek shifts; no single sentiment reading identifies a durable bottom.'),*textblock('Bottom or early-bull assessment',('Contrarian fear can improve prospective entry context but is not independent proof of a bottom.' if val<45 else 'Neutral-to-greedy sentiment does not confirm capitulation and may reduce the attractiveness of chasing the rally.' if val>55 else 'Neutral sentiment supplies little standalone confirmation either way.'))]
else:sent += [para('No current primary-source Fear & Greed reading passed validation. Status: UNVERIFIED; no value chart is published.'),para(FEAR_STATUS,'SmallReport')]
section('Sentiment overlay | Fear & Greed',sent+[para('Source: '+FEAR_SOURCE+' | '+FEAR_STATUS,'SmallReport')])

section('Sources and indicator integrity',[
    para('All technical time series use verified daily Bitstamp BTC/USD standard closes, not Heikin Ashi. The daily series is checked for continuity and the month/week series contain only completed bars.'),
    para('Published reference: https://raw.githubusercontent.com/tvella23/B-Bottom-Watch/main/latest.json'),
    para('Pi Cycle: EMA(150, daily close) versus 0.745 x SMA(471, daily close); bullish crossover of the long adjusted SMA above the short EMA is the bottom event. RSI: Wilder(14) on completed monthly closes, compared with a simple moving average over 14 RSI observations. EMA ribbon: 20/25/30/35/40/45/50/55 on completed Sunday-ending weekly closes.'),
    para('The report does not silently replace missing on-chain, ETF, stablecoin or derivatives statistics with stale web previews, a different data provider or illustrative numbers. Only dated, valid primary-source observations are plotted.'),
    para('Future changes to calculated indicator methodology, source and aggregation rules require a revalidation against trusted chart references before publication.'),
    para('Generated '+AS_OF+'. Research and education only; not personal investment advice.','SmallReport')])

class Doc(BaseDocTemplate):pass
doc=Doc(str(REPORT),pagesize=A4,leftMargin=15*mm,rightMargin=15*mm,topMargin=21*mm,bottomMargin=17*mm,title='Bitcoin Bottom Watch | Evidence-family report',author='Bitcoin Bottom Watch')
frame=Frame(doc.leftMargin,doc.bottomMargin,doc.width,doc.height,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
def onpage(canvas,doc):
    canvas.setStrokeColor(colors.HexColor('#D9E2E7'));canvas.line(15*mm,15*mm,195*mm,15*mm)
    canvas.setFillColor(colors.HexColor(GREY));canvas.setFont('Helvetica',7)
    canvas.drawString(15*mm,11*mm,'BITCOIN BOTTOM WATCH  |  RESEARCH BRIEF')
    canvas.drawRightString(195*mm,11*mm,f'{OBS["day"]}  |  {doc.page}')
doc.addPageTemplates(PageTemplate(id='standard',frames=frame,onPage=onpage))
doc.build(story)
print('Created',REPORT,'size',REPORT.stat().st_size,'Fear & Greed primary',FEAR_STATUS)