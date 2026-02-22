import warnings, time, itertools
from datetime import datetime
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import streamlit as st
import yfinance as yf
import ta

PERIODS=['1mo','3mo','6mo','1y','2y','5y','10y']
INTERVALS=['1d','1wk','1mo']
PL={'1mo':'1M','3mo':'3M','6mo':'6M','1y':'1Y','2y':'2Y','5y':'5Y','10y':'10Y'}
IL={'1d':'Daily','1wk':'Weekly','1mo':'Monthly'}
UO={'5min':300,'10min':600,'15min':900,'30min':1800,'1hr':3600}
POPULAR=[
    ('1326.T','SPDR Gold'),('7203.T','Toyota'),('6758.T','Sony'),
    ('9984.T','SBG'),('6861.T','Keyence'),('8306.T','MUFG'),
    ('^N225','Nikkei225'),('AAPL','Apple'),('NVDA','NVIDIA'),('^GSPC','SP500'),
]
C={
    'bg':'#0d1117','panel':'#161b22','grid':'#21262d',
    'text':'#e6edf3','sub':'#8b949e',
    'buy':'#3fb950','sell':'#f85149','neutral':'#58a6ff',
    'sma25':'#ffa657','sma75':'#58a6ff','sma200':'#bc8cff',
    'bb':'#388bfd','macd':'#58a6ff','msig':'#ffa657',
    'hup':'#3fb950','hdn':'#f85149','rsi':'#d2a8ff',
    'cup':'#3fb950','cdn':'#f85149',
}

def flatten_df(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df

def compute_indicators(df):
    cl=df['Close']; hi=df['High']; lo=df['Low']; vo=df['Volume']
    df['SMA25'] =cl.rolling(25).mean()
    df['SMA75'] =cl.rolling(75).mean()
    df['SMA200']=cl.rolling(200).mean()
    bb=ta.volatility.BollingerBands(cl,20,2)
    df['BB_u']=bb.bollinger_hband()
    df['BB_m']=bb.bollinger_mavg()
    df['BB_l']=bb.bollinger_lband()
    df['BB_w']=(df['BB_u']-df['BB_l'])/df['BB_m']
    mc=ta.trend.MACD(cl,26,12,9)
    df['MACD']=mc.macd(); df['MSIG']=mc.macd_signal(); df['MHST']=mc.macd_diff()
    df['RSI']=ta.momentum.RSIIndicator(cl,14).rsi()
    df['ATR']=ta.volatility.AverageTrueRange(hi,lo,cl,14).average_true_range()
    adx=ta.trend.ADXIndicator(hi,lo,cl,14)
    df['ADX']=adx.adx()
    sto=ta.momentum.StochasticOscillator(hi,lo,cl,14,3)
    df['SK']=sto.stoch(); df['SD']=sto.stoch_signal()
    df['VMA']=vo.rolling(20).mean()
    return df

def compute_signals(df, p):
    s=df
    mxu=(s['MACD']>s['MSIG'])&(s['MACD'].shift(1)<=s['MSIG'].shift(1))
    mxd=(s['MACD']<s['MSIG'])&(s['MACD'].shift(1)>=s['MSIG'].shift(1))
    sxu=(s['SK']>s['SD'])&(s['SK'].shift(1)<=s['SD'].shift(1))
    sxd=(s['SK']<s['SD'])&(s['SK'].shift(1)>=s['SD'].shift(1))
    bsc=(
        (s['SMA25']>s['SMA75']).astype(int) * p['w_trend'] +
        mxu.astype(int) * p['w_macd'] +
        ((s['RSI']>p['rsi_buy_th'])&(s['RSI'].shift(1)<=p['rsi_buy_th'])).astype(int) +
        (s['Close']<=s['BB_l']*1.01).astype(int) +
        (s['ADX']>p['adx_th']).astype(int) +
        (sxu&(s['SK']<p['stoch_buy_th'])).astype(int)
    )
    ssc=(
        (s['SMA25']<s['SMA75']).astype(int) * p['w_trend'] +
        mxd.astype(int) * p['w_macd'] +
        ((s['RSI']<p['rsi_sell_th'])&(s['RSI'].shift(1)>=p['rsi_sell_th'])).astype(int) +
        (s['Close']>=s['BB_u']*0.99).astype(int) +
        (s['ADX']>p['adx_th']).astype(int) +
        (sxd&(s['SK']>p['stoch_sell_th'])).astype(int)
    )
    df=df.copy()
    df['bsc']=bsc; df['ssc']=ssc; df['sig']=0
    df.loc[bsc>=p['buy_th'],'sig']=1
    df.loc[ssc>=p['sell_th'],'sig']=-1
    df['trend']='Range'
    df.loc[(s['SMA25']>s['SMA75'])&(s['SMA75']>s['SMA200'])&(s['ADX']>20),'trend']='Up'
    df.loc[(s['SMA25']<s['SMA75'])&(s['SMA75']<s['SMA200'])&(s['ADX']>20),'trend']='Down'
    return df

DEFAULT_PARAMS={
    'w_trend':1,'w_macd':2,
    'rsi_buy_th':30,'rsi_sell_th':70,
    'adx_th':25,
    'stoch_buy_th':25,'stoch_sell_th':75,
    'buy_th':3,'sell_th':3,
}

@st.cache_data(ttl=60, show_spinner=False)
def fetch_raw(code, period, interval):
    try:
        df=yf.download(code,period=period,interval=interval,
                       auto_adjust=True,progress=False)
        if df is None or df.empty: return None
        df=flatten_df(df)
        df=df.dropna(subset=['Close','Open','High','Low','Volume'])
        if len(df)<60: return None
        return df
    except Exception as e:
        return None

def run_backtest(df, cost=0.001):
    cl=df['Close'].values; sig=df['sig'].values; dates=df.index
    trades=[]; eq=1.0; equity=[1.0]; pos=0; ep=0.0; ed=None
    for i in range(1,len(df)):
        p=cl[i]; ps=sig[i-1]
        if pos==0 and ps==1:
            pos=1; ep=p*(1+cost); ed=dates[i]
        elif pos==1 and ps==-1:
            xp=p*(1-cost); ret=(xp-ep)/ep
            eq*=(1+ret)
            trades.append({'entry_date':ed,'exit_date':dates[i],
                           'entry':ep,'exit':xp,'ret':ret*100,
                           'result':'Win' if ret>0 else 'Loss'})
            pos=0
        equity.append(eq)
    if len(trades)<3: return None
    eq_s=pd.Series(equity,index=dates)
    bh=(cl[-1]-cl[0])/cl[0]*100
    wins=[t for t in trades if t['ret']>0]
    loss=[t for t in trades if t['ret']<=0]
    n=len(trades)
    wr=len(wins)/n*100
    aw=np.mean([t['ret'] for t in wins]) if wins else 0
    al=np.mean([t['ret'] for t in loss]) if loss else 0
    pf=abs(sum(t['ret'] for t in wins)/sum(t['ret'] for t in loss)) if loss else 999.0
    roll_max=eq_s.cummax()
    dd=(eq_s-roll_max)/roll_max*100
    mdd=dd.min()
    yrs=(dates[-1]-dates[0]).days/365.25
    cagr=((eq)**(1/yrs)-1)*100 if yrs>0 else 0
    dr=eq_s.pct_change().dropna()
    sharpe=dr.mean()/dr.std()*np.sqrt(252) if dr.std()>0 else 0
    sr=(eq-1)*100
    return {'trades':trades,'equity':eq_s,'drawdown':dd,
            'stats':{'n':n,'wr':wr,'aw':aw,'al':al,'pf':pf,
                     'sr':sr,'bh':bh,'mdd':mdd,'cagr':cagr,'sharpe':sharpe}}

@st.cache_data(ttl=300, show_spinner=False)
def grid_search(code, period, interval, cost=0.001):
    raw=fetch_raw(code,period,interval)
    if raw is None: return None
    base=compute_indicators(raw.copy())

    param_grid={
        'w_trend':[1,2],
        'w_macd':[1,2,3],
        'rsi_buy_th':[25,30,35],
        'rsi_sell_th':[65,70,75],
        'adx_th':[20,25,30],
        'stoch_buy_th':[20,25,30],
        'stoch_sell_th':[70,75,80],
        'buy_th':[2,3,4],
        'sell_th':[2,3,4],
    }

    keys=list(param_grid.keys())
    vals=list(param_grid.values())
    combos=list(itertools.product(*vals))

    best_score=-999
    best_params=None
    best_bt=None
    results=[]

    for combo in combos:
        p=dict(zip(keys,combo))
        try:
            df=compute_signals(base.copy(),p)
            bt=run_backtest(df,cost)
            if bt is None: continue
            s=bt['stats']
            score=s['sr']-max(0,-s['mdd'])*0.3+s['sharpe']*5
            results.append({**p,'score':score,'sr':s['sr'],'bh':s['bh'],
                            'cagr':s['cagr'],'sharpe':s['sharpe'],'mdd':s['mdd'],
                            'wr':s['wr'],'n':s['n']})
            if score>best_score:
                best_score=score
                best_params=p
                best_bt=bt
        except:
            continue

    if best_params is None: return None
    res_df=pd.DataFrame(results).sort_values('score',ascending=False)
    return {'best_params':best_params,'best_bt':best_bt,
            'results':res_df,'base':base}

def make_bt_chart(bt, title):
    eq=bt['equity']; dd=bt['drawdown']; trades=bt['trades']
    fig,axes=plt.subplots(2,1,figsize=(14,8),facecolor=C['bg'],
                           gridspec_kw={'height_ratios':[3,1],'hspace':0.06})
    for ax in axes:
        ax.set_facecolor(C['panel'])
        ax.tick_params(colors=C['sub'],labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor(C['grid'])
        ax.grid(color=C['grid'],lw=0.5,ls='--',alpha=0.6)
    fig.suptitle(title,fontsize=11,color=C['text'],fontweight='bold',x=0.02,ha='left',y=0.99)
    ax1,ax2=axes
    bh_vals=np.linspace(1.0,1+bt['stats']['bh']/100,len(eq))
    ax1.plot(eq.index,eq.values,color=C['buy'],lw=2.0,label='Strategy',zorder=3)
    ax1.plot(eq.index,bh_vals,color=C['sub'],lw=1.3,ls='--',label='Buy&Hold',zorder=2)
    ax1.axhline(1.0,color=C['grid'],lw=0.8)
    for t in trades:
        col=C['buy'] if t['ret']>0 else C['sell']
        ax1.axvline(t['entry_date'],color=col,lw=0.4,alpha=0.35)
    ax1.set_ylabel('Equity',color=C['sub'],fontsize=9)
    ax1.legend(loc='upper left',fontsize=8,framealpha=0.3,
               facecolor=C['bg'],edgecolor=C['grid'],labelcolor=C['text'])
    ax1.set_xticklabels([])
    ax2.fill_between(dd.index,dd.values,0,where=dd.values<0,color=C['sell'],alpha=0.45)
    ax2.plot(dd.index,dd.values,color=C['sell'],lw=0.9)
    ax2.axhline(0,color=C['grid'],lw=0.8)
    ax2.set_ylabel('Drawdown%',color=C['sub'],fontsize=9)
    plt.close('all')
    return fig

def slabel(v):
    return 'BUY' if v==1 else ('SELL' if v==-1 else 'NEUTRAL')

def draw_candles(ax,df):
    op=df['Open'].values; hi=df['High'].values
    lo=df['Low'].values; cl=df['Close'].values
    for i in range(len(df)):
        col=C['cup'] if cl[i]>=op[i] else C['cdn']
        ax.plot([i,i],[lo[i],hi[i]],color=col,lw=0.7,zorder=2)
        b0=min(op[i],cl[i]); b1=max(op[i],cl[i])
        ax.bar(i,max(b1-b0,1e-6),bottom=b0,width=0.6,color=col,linewidth=0,zorder=3)

def make_chart(df,title,mobile=False):
    w,h=(9,14) if mobile else (16,13)
    fig=plt.figure(figsize=(w,h),facecolor=C['bg'])
    fig.suptitle(title,fontsize=9 if mobile else 12,color=C['text'],
                 fontweight='bold',x=0.02,ha='left',y=0.99)
    gs=gridspec.GridSpec(5,1,figure=fig,height_ratios=[4,1,1.3,1.3,1.3],
                         hspace=0.05,top=0.96,bottom=0.04,
                         left=0.10 if mobile else 0.07,right=0.97)
    axes=[fig.add_subplot(gs[i]) for i in range(5)]
    for ax in axes:
        ax.set_facecolor(C['panel'])
        ax.tick_params(colors=C['sub'],labelsize=6 if mobile else 7.5)
        for sp in ax.spines.values(): sp.set_edgecolor(C['grid'])
        ax.grid(axis='y',color=C['grid'],lw=0.5,ls='--',alpha=0.7)
        ax.grid(axis='x',color=C['grid'],lw=0.3,alpha=0.4)
    x=np.arange(len(df))
    ds=df.index.strftime('%m/%d').tolist()
    step=max(1,len(df)//(6 if mobile else 10))
    tks=list(range(0,len(df),step))
    for ax in axes:
        ax.set_xlim(-1,len(df)); ax.set_xticks(tks)
    for ax in axes[:-1]: ax.set_xticklabels([])
    axes[-1].set_xticklabels([ds[i] for i in tks],rotation=35,ha='right',
                              fontsize=5 if mobile else 6.5)
    cv=df['Close'].values
    ax1=axes[0]
    draw_candles(ax1,df)
    ax1.plot(x,df['SMA25'].values,color=C['sma25'],lw=1.0,label='SMA25')
    ax1.plot(x,df['SMA75'].values,color=C['sma75'],lw=1.2,label='SMA75')
    ax1.plot(x,df['SMA200'].values,color=C['sma200'],lw=1.3,ls='--',label='SMA200')
    ax1.fill_between(x,df['BB_u'].values,df['BB_l'].values,alpha=0.1,color=C['bb'])
    ax1.plot(x,df['BB_u'].values,color=C['bb'],lw=0.7,ls=':',alpha=0.8)
    ax1.plot(x,df['BB_l'].values,color=C['bb'],lw=0.7,ls=':',alpha=0.8)
    bx=[df.index.get_loc(i) for i in df.index[df['sig']==1]]
    sx=[df.index.get_loc(i) for i in df.index[df['sig']==-1]]
    by=[float(df['Low'].iloc[i])*0.985 for i in bx]
    sy=[float(df['High'].iloc[i])*1.015 for i in sx]
    ms=60 if mobile else 100
    ax1.scatter(bx,by,marker='^',s=ms,color=C['buy'],zorder=6,label='BUY')
    ax1.scatter(sx,sy,marker='v',s=ms,color=C['sell'],zorder=6,label='SELL')
    tc={'Up':'#3fb95015','Down':'#f8514915','Range':'none'}
    prev,start=None,0
    for i,(_,row) in enumerate(df.iterrows()):
        t=row['trend']
        if t!=prev:
            if prev and tc[prev]!='none': ax1.axvspan(start,i,color=tc[prev],lw=0,zorder=0)
            start,prev=i,t
    if prev and tc[prev]!='none': ax1.axvspan(start,len(df),color=tc[prev],lw=0,zorder=0)
    tnow=df['trend'].iloc[-1]
    tclr={'Up':C['buy'],'Down':C['sell'],'Range':C['neutral']}
    ax1.text(0.995,0.97,tnow,transform=ax1.transAxes,ha='right',va='top',
             fontsize=8 if mobile else 10,fontweight='bold',color=tclr[tnow])
    ax1.set_ylabel('Price',color=C['sub'],fontsize=6)
    ax1.legend(loc='upper left',fontsize=5.5,framealpha=0.3,
               facecolor=C['bg'],edgecolor=C['grid'],labelcolor=C['text'])
    ax2=axes[1]
    vc=[C['cup'] if cv[i]>=df['Open'].values[i] else C['cdn'] for i in range(len(df))]
    ax2.bar(x,df['Volume'].values,color=vc,alpha=0.7,width=0.8)
    ax2.plot(x,df['VMA'].values,color=C['sma25'],lw=0.9)
    ax2.set_ylabel('Vol',color=C['sub'],fontsize=6)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda v,_: f'{v/1e6:.0f}M' if v>=1e6 else f'{int(v/1e3)}K'))
    ax3=axes[2]
    hist=df['MHST'].values
    ax3.bar(x,hist,color=[C['hup'] if h>=0 else C['hdn'] for h in hist],alpha=0.8,width=0.8)
    ax3.plot(x,df['MACD'].values,color=C['macd'],lw=1.0,label='MACD')
    ax3.plot(x,df['MSIG'].values,color=C['msig'],lw=0.9,ls='--',label='Sig')
    ax3.axhline(0,color=C['grid'],lw=0.7)
    ax3.set_ylabel('MACD',color=C['sub'],fontsize=6)
    ax3.legend(loc='upper left',fontsize=5.5,framealpha=0.2,
               facecolor=C['bg'],edgecolor='none',labelcolor=C['text'])
    ax4=axes[3]
    ax4.plot(x,df['RSI'].values,color=C['rsi'],lw=1.1)
    ax4.axhline(70,color=C['sell'],lw=0.8,ls='--')
    ax4.axhline(50,color=C['grid'],lw=0.5)
    ax4.axhline(30,color=C['buy'],lw=0.8,ls='--')
    ax4.fill_between(x,df['RSI'].values,70,where=df['RSI'].values>=70,alpha=0.2,color=C['sell'])
    ax4.fill_between(x,df['RSI'].values,30,where=df['RSI'].values<=30,alpha=0.2,color=C['buy'])
    ax4.set_ylim(0,100); ax4.set_ylabel('RSI',color=C['sub'],fontsize=6)
    ax5=axes[4]
    ax5.plot(x,df['SK'].values,color=C['macd'],lw=1.1,label='%K')
    ax5.plot(x,df['SD'].values,color=C['msig'],lw=0.9,ls='--',label='%D')
    ax5.axhline(80,color=C['sell'],lw=0.8,ls='--')
    ax5.axhline(20,color=C['buy'],lw=0.8,ls='--')
    ax5.fill_between(x,df['SK'].values,80,where=df['SK'].values>=80,alpha=0.15,color=C['sell'])
    ax5.fill_between(x,df['SK'].values,20,where=df['SK'].values<=20,alpha=0.15,color=C['buy'])
    ax5.set_ylim(0,100); ax5.set_ylabel('Stoch',color=C['sub'],fontsize=6)
    ax5.legend(loc='upper left',fontsize=5.5,framealpha=0.2,
               facecolor=C['bg'],edgecolor='none',labelcolor=C['text'])
    plt.close('all')
    return fig

st.set_page_config(page_title='Trend Signal',page_icon='chart_with_upwards_trend',
                   layout='wide',initial_sidebar_state='collapsed')
st.markdown('''<style>
.stApp,[data-testid="stAppViewContainer"]{background:#0d1117!important;color:#e6edf3;}
section[data-testid="stSidebar"]{background:#161b22!important;}
[data-testid="stHeader"]{background:#0d1117!important;}
.bb{background:#1a3d24;color:#3fb950;border:1.5px solid #3fb950;border-radius:10px;padding:7px 16px;font-size:1.05rem;font-weight:bold;display:inline-block;}
.bs{background:#3d1a1a;color:#f85149;border:1.5px solid #f85149;border-radius:10px;padding:7px 16px;font-size:1.05rem;font-weight:bold;display:inline-block;}
.bn{background:#1a1f3d;color:#58a6ff;border:1.5px solid #58a6ff;border-radius:10px;padding:7px 16px;font-size:1.05rem;font-weight:bold;display:inline-block;}
[data-testid="stMetricValue"]{color:#e6edf3!important;font-size:1.15rem!important;}
[data-testid="metric-container"]{background:#161b22;border-radius:10px;border:1px solid #21262d;padding:10px 14px;}
.stButton>button{background:#21262d;color:#e6edf3;border:1px solid #30363d;border-radius:8px;font-weight:600;}
.stTextInput>div>div>input{background:#161b22!important;color:#e6edf3!important;border:1px solid #30363d!important;border-radius:8px!important;}
.stTabs [data-baseweb="tab"]{color:#8b949e;}
.stTabs [aria-selected="true"]{color:#e6edf3!important;}
.ut{color:#8b949e;font-size:0.72rem;}
@media(max-width:768px){[data-testid="stMetricValue"]{font-size:0.95rem!important;}h1{font-size:1.3rem!important;}.block-container{padding:0.5rem 0.8rem!important;}}
h1,h2,h3{color:#e6edf3!important;}hr{border-color:#21262d;}
</style>''',unsafe_allow_html=True)

for k,v in [('result',None),('raw_df',None),('code',''),('period','2y'),('interval','1d'),
            ('auto_on',False),('auto_interval','15min'),('ulog',[]),('nxt',None),
            ('opt_result',None),('use_opt',False),('active_params',None)]:
    if k not in st.session_state: st.session_state[k]=v

with st.sidebar:
    st.markdown('### Settings')
    period  =st.selectbox('Period',  PERIODS,  index=4,format_func=lambda x:PL[x])
    interval=st.selectbox('Interval',INTERVALS,index=0,format_func=lambda x:IL[x])
    st.divider()
    st.markdown('### Auto-Update')
    aint=st.selectbox('Every',list(UO.keys()),index=2,disabled=st.session_state['auto_on'])
    c1,c2=st.columns(2)
    with c1:
        if st.button('Start',use_container_width=True,
                     disabled=not st.session_state['code'] or st.session_state['auto_on']):
            st.session_state.update({'auto_on':True,'auto_interval':aint,
                                     'nxt':time.time()+UO[aint]}); st.rerun()
    with c2:
        if st.button('Stop',use_container_width=True,disabled=not st.session_state['auto_on']):
            st.session_state['auto_on']=False; st.rerun()
    if st.session_state['auto_on']:
        st.success(f"Every {st.session_state['auto_interval']}")
    else:
        st.caption('Stopped')
    if st.session_state['ulog']:
        st.divider(); st.markdown('### Log')
        for e in st.session_state['ulog'][:6]:
            col='#3fb950' if e['s']=='BUY' else ('#f85149' if e['s']=='SELL' else '#58a6ff')
            st.markdown(f"<small><span style='color:#8b949e'>{e['t']}</span> "
                        f"<b>{e['c']}</b> <span style='color:{col}'>{e['s']}</span> "
                        f"{e['p']}</small>",unsafe_allow_html=True)

st.markdown('# Trend Signal Analyzer')
i1,i2,i3=st.columns([4,1,1])
with i1:
    typed=st.text_input('t',value=st.session_state.get('code',''),
                        placeholder='1326.T / 7203.T / AAPL / ^N225',
                        label_visibility='collapsed')
with i2:
    fbtn=st.button('Analyze',use_container_width=True,type='primary')
with i3:
    rbtn=st.button('Refresh',use_container_width=True,disabled=not st.session_state['code'])

def do_fetch(code, period, interval, params=None):
    raw=fetch_raw(code,period,interval)
    if raw is None: return None
    p=params if params else DEFAULT_PARAMS
    df=compute_indicators(raw.copy())
    df=compute_signals(df,p)
    return {'df':df,'at':datetime.now().strftime('%Y/%m/%d %H:%M:%S')}

if fbtn and typed.strip():
    code=typed.strip().upper()
    fetch_raw.clear(); grid_search.clear()
    st.session_state.update({'opt_result':None,'use_opt':False,'active_params':None})
    params=st.session_state.get('active_params') or DEFAULT_PARAMS
    with st.spinner(f'Fetching {code}...'):
        res=do_fetch(code,period,interval,params)
    if res:
        st.session_state.update({'result':res,'code':code,'period':period,'interval':interval})
        st.rerun()
    else:
        st.error('Failed to fetch data. Check ticker code.')

if rbtn and st.session_state['code']:
    fetch_raw.clear()
    params=st.session_state.get('active_params') or DEFAULT_PARAMS
    with st.spinner('Refreshing...'):
        res=do_fetch(st.session_state['code'],st.session_state['period'],
                     st.session_state['interval'],params)
    if res:
        st.session_state['result']=res; dr=res['df']
        lg=st.session_state['ulog']
        lg.insert(0,{'t':res['at'][11:],'c':st.session_state['code'],
                     's':slabel(int(dr['sig'].iloc[-1])),
                     'p':f"{float(dr['Close'].iloc[-1]):,.1f}"})
        st.session_state['ulog']=lg[:30]; st.rerun()

if (st.session_state['auto_on'] and st.session_state['code']
        and st.session_state['nxt'] and time.time()>=st.session_state['nxt']):
    fetch_raw.clear()
    params=st.session_state.get('active_params') or DEFAULT_PARAMS
    with st.spinner('Auto-updating...'):
        res=do_fetch(st.session_state['code'],st.session_state['period'],
                     st.session_state['interval'],params)
    if res:
        st.session_state['result']=res; dr=res['df']
        lg=st.session_state['ulog']
        lg.insert(0,{'t':res['at'][11:],'c':st.session_state['code'],
                     's':slabel(int(dr['sig'].iloc[-1])),
                     'p':f"{float(dr['Close'].iloc[-1]):,.1f}"})
        st.session_state['ulog']=lg[:30]
    st.session_state['nxt']=time.time()+UO[st.session_state['auto_interval']]

result=st.session_state.get('result')
if result is None:
    st.info('Enter a ticker above or pick one below.')
    st.divider()
    cols=st.columns(5)
    for i,(c,n) in enumerate(POPULAR):
        if cols[i%5].button(n,key=f'h{c}',use_container_width=True):
            fetch_raw.clear()
            with st.spinner(f'Fetching {n}...'):
                res=do_fetch(c,'2y','1d',DEFAULT_PARAMS)
            if res:
                st.session_state.update({'result':res,'code':c,'period':'2y','interval':'1d',
                                         'active_params':DEFAULT_PARAMS})
                st.rerun()
    st.stop()

df=result['df']; code=st.session_state['code']
per=st.session_state['period']; inv=st.session_state['interval']
lat=df.iloc[-1]
cv=float(df['Close'].iloc[-1]); pv=float(df['Close'].iloc[-2])
chgp=(cv-pv)/pv*100; chga=cv-pv
sig=int(lat['sig']); trend=lat['trend']; fat=result['at']

using_opt=st.session_state.get('use_opt',False)
opt_label=' (Optimized)' if using_opt else ' (Default)'
st.markdown(f"### {code}  {PL[per]}/{IL[inv]}{opt_label}  <span class='ut'>Updated: {fat}</span>",
            unsafe_allow_html=True)
sm={1:('bb','BUY Signal'),-1:('bs','SELL Signal'),0:('bn','NEUTRAL')}
cc,st2=sm[sig]; tc2={'Up':'#3fb950','Down':'#f85149','Range':'#58a6ff'}[trend]
r1,r2,r3=st.columns([2,2,3])
with r1: st.markdown(f"<div class='{cc}'>{st2}</div>",unsafe_allow_html=True)
with r2: st.markdown(f"<div style='font-size:0.95rem;margin-top:9px;color:{tc2};font-weight:bold'>{trend}</div>",unsafe_allow_html=True)
with r3:
    bs=int(lat['bsc']); ss=int(lat['ssc'])
    st.progress(bs/7,text=f'Buy {bs}/7'); st.progress(ss/7,text=f'Sell {ss}/7')
st.divider()
m1,m2,m3,m4=st.columns(4)
m1.metric('Price',f'{cv:,.1f}',f'{chgp:+.2f}%')
m2.metric('RSI',  f"{float(lat['RSI']):.1f}",
          'Overbought' if float(lat['RSI'])>70 else ('Oversold' if float(lat['RSI'])<30 else '-'))
m3.metric('ADX',  f"{float(lat['ADX']):.1f}",'Trend' if float(lat['ADX'])>25 else 'Range')
m4.metric('MACD', f"{float(lat['MACD']):.2f}")
m5,m6,m7,m8=st.columns(4)
m5.metric('Change',f'{chga:+,.1f}')
m6.metric('Stoch%K',f"{float(lat['SK']):.1f}")
m7.metric('SMA25',f"{float(lat['SMA25']):.1f}")
m8.metric('BBWidth',f"{float(lat['BB_w']):.4f}")
st.divider()

tab1,tab2,tab3,tab4,tab5=st.tabs(['Chart','Backtest','Optimize','Signals','Data'])

with tab1:
    mob=st.toggle('Mobile view',value=False)
    with st.spinner('Rendering...'):
        fig=make_chart(df,f"{code} {PL[per]}/{IL[inv]} ({len(df)} bars)",mobile=mob)
        st.pyplot(fig,use_container_width=True)

with tab2:
    st.markdown('### Backtest')
    st.caption('Buy signal next bar -> hold -> Sell signal next bar -> repeat')
    cost_pct=st.slider('Transaction cost per side (%)',0.0,1.0,0.1,0.05,key='bt_cost')
    bt=run_backtest(df,cost=cost_pct/100)
    if bt is None:
        st.warning('Not enough trades. Try longer period (2Y+ recommended).')
    else:
        s=bt['stats']
        diff=s['sr']-s['bh']
        c1,c2,c3,c4=st.columns(4)
        c1.metric('Strategy Return',f"{s['sr']:+.1f}%",
                  f"{diff:+.1f}% vs BuyHold",
                  delta_color='normal' if diff>=0 else 'inverse')
        c2.metric('CAGR',f"{s['cagr']:+.1f}%")
        c3.metric('Max Drawdown',f"{s['mdd']:.1f}%")
        c4.metric('Sharpe',f"{s['sharpe']:.2f}")
        c5,c6,c7,c8=st.columns(4)
        c5.metric('Trades',f"{s['n']}")
        c6.metric('Win Rate',f"{s['wr']:.1f}%")
        c7.metric('Avg Win',f"{s['aw']:+.2f}%")
        c8.metric('Avg Loss',f"{s['al']:+.2f}%")
        pf_str=f"{s['pf']:.2f}" if s['pf']<999 else 'inf'
        st.metric('Profit Factor',pf_str)
        st.divider()
        with st.spinner('Drawing equity curve...'):
            fig2=make_bt_chart(bt,f"{code} Backtest  {PL[per]}  cost={cost_pct}%/side{opt_label}")
            st.pyplot(fig2,use_container_width=True)
        st.markdown('#### Trade List')
        tdf=pd.DataFrame(bt['trades'])
        tdf['entry_date']=tdf['entry_date'].dt.strftime('%Y/%m/%d')
        tdf['exit_date'] =tdf['exit_date'].dt.strftime('%Y/%m/%d')
        tdf['entry']=tdf['entry'].round(1); tdf['exit']=tdf['exit'].round(1)
        tdf['ret']=tdf['ret'].round(2)
        tdf.columns=['Entry','Exit','Buy Price','Sell Price','Return%','Result']
        st.dataframe(tdf.iloc[::-1].reset_index(drop=True),use_container_width=True,height=350)

with tab3:
    st.markdown('### Parameter Optimization')
    st.caption('Grid search across signal parameters to maximize risk-adjusted return vs Buy&Hold.')
    st.info('Searches ~1,000+ parameter combinations. Takes 20-60 seconds.')
    cost_opt=st.slider('Transaction cost (%)',0.0,1.0,0.1,0.05,key='opt_cost')

    if st.button('Run Optimization', type='primary', use_container_width=True):
        grid_search.clear()
        with st.spinner('Running grid search... please wait (up to 60s)'):
            opt=grid_search(st.session_state['code'],
                            st.session_state['period'],
                            st.session_state['interval'],
                            cost=cost_opt/100)
        if opt is None:
            st.error('Optimization failed. Try a longer period.')
        else:
            st.session_state['opt_result']=opt
            st.success('Optimization complete!')

    opt=st.session_state.get('opt_result')
    if opt:
        bp=opt['best_params']; bs_bt=opt['best_bt']; s=bs_bt['stats']
        diff=s['sr']-s['bh']

        st.markdown('#### Best Parameters Found')
        c1,c2,c3,c4=st.columns(4)
        c1.metric('Strategy Return',f"{s['sr']:+.1f}%",
                  f"{diff:+.1f}% vs BuyHold",
                  delta_color='normal' if diff>=0 else 'inverse')
        c2.metric('CAGR',f"{s['cagr']:+.1f}%")
        c3.metric('Max Drawdown',f"{s['mdd']:.1f}%")
        c4.metric('Sharpe',f"{s['sharpe']:.2f}")

        st.markdown('**Optimized parameter values:**')
        param_cols=st.columns(3)
        param_items=list(bp.items())
        for i,(k,v) in enumerate(param_items):
            def_v=DEFAULT_PARAMS[k]
            delta=f"(default: {def_v})" if v!=def_v else '(unchanged)'
            param_cols[i%3].metric(k,str(v),delta)

        if st.button('Apply Optimized Parameters', type='primary', use_container_width=True):
            st.session_state['active_params']=bp
            st.session_state['use_opt']=True
            fetch_raw.clear()
            raw=fetch_raw(st.session_state['code'],
                          st.session_state['period'],
                          st.session_state['interval'])
            if raw is not None:
                dfo=compute_indicators(raw.copy())
                dfo=compute_signals(dfo,bp)
                st.session_state['result']={'df':dfo,'at':datetime.now().strftime('%Y/%m/%d %H:%M:%S')}
            st.rerun()

        if st.button('Reset to Default Parameters',use_container_width=True):
            st.session_state['active_params']=DEFAULT_PARAMS
            st.session_state['use_opt']=False
            fetch_raw.clear()
            raw=fetch_raw(st.session_state['code'],
                          st.session_state['period'],
                          st.session_state['interval'])
            if raw is not None:
                dfo=compute_indicators(raw.copy())
                dfo=compute_signals(dfo,DEFAULT_PARAMS)
                st.session_state['result']={'df':dfo,'at':datetime.now().strftime('%Y/%m/%d %H:%M:%S')}
            st.rerun()

        st.divider()
        with st.spinner('Drawing optimized equity curve...'):
            fig3=make_bt_chart(bs_bt,f"{code} OPTIMIZED  {PL[per]}  cost={cost_opt}%/side")
            st.pyplot(fig3,use_container_width=True)

        st.markdown('#### Top 20 Parameter Combinations')
        show_cols=['score','sr','bh','cagr','sharpe','mdd','wr','n',
                   'buy_th','sell_th','w_macd','adx_th','rsi_buy_th','rsi_sell_th']
        rdf=opt['results'][show_cols].head(20).round(2)
        st.dataframe(rdf,use_container_width=True,height=350)

with tab4:
    sd=df[df['sig']!=0].copy()
    if not sd.empty:
        rows=[]
        for ih,rh in sd.iterrows():
            rows.append({'Date':ih.strftime('%Y/%m/%d'),'Signal':slabel(int(rh['sig'])),
                         'Price':f"{float(df.loc[ih,'Close']):,.1f}",
                         'BuySc':int(rh['bsc']),'SellSc':int(rh['ssc']),
                         'RSI':round(float(rh['RSI']),1),'ADX':round(float(rh['ADX']),1),
                         'Trend':rh['trend']})
        st.dataframe(pd.DataFrame(rows).iloc[::-1].reset_index(drop=True),
                     use_container_width=True,height=400)
    else:
        st.info('No signals. Try longer period.')

with tab5:
    sc=['Open','High','Low','Close','Volume','SMA25','SMA75','RSI','ADX','MACD','sig','trend']
    dfs=df[[c for c in sc if c in df.columns]].tail(60).copy()
    st.dataframe(dfs.style.format(precision=2),use_container_width=True,height=400)

if st.session_state['auto_on'] and st.session_state['nxt']:
    rem=int(st.session_state['nxt']-time.time())
    if rem>0:
        st.caption(f'Next update in {rem}s')
        time.sleep(min(rem,30)); st.rerun()
    else:
        st.rerun()
