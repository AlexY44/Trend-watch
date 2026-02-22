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

# --- 定数と設定 ---
PERIODS=['2y','3y','5y','10y']
INTERVALS=['1d','1wk']
PL={'2y':'2Y','3y':'3Y','5y':'5Y','10y':'10Y'}
IL={'1d':'Daily','1wk':'Weekly'}
UO={'5min':300, '10min':600, '15min':900, '30min':1800, '1hr':3600}
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

DEFAULT_PARAMS={
    'w_trend':1,'w_macd':2,
    'rsi_buy_th':30,'rsi_sell_th':70,
    'adx_th':25,
    'stoch_buy_th':25,'stoch_sell_th':75,
    'buy_th':3,'sell_th':3,
}

PARAM_GRID={
    'w_trend':[1, 2],
    'w_macd':[1, 2, 3],
    'rsi_buy_th':[25, 30, 35],
    'rsi_sell_th':[65, 70, 75],
    'adx_th':[20, 25],
    'buy_th':[2, 3, 4],
    'sell_th':[2, 3, 4],
}

# --- 共通関数 ---
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
        (s['SMA25']>s['SMA75']).astype(int)*p['w_trend'] +
        mxu.astype(int)*p['w_macd'] +
        ((s['RSI']>p['rsi_buy_th'])&(s['RSI'].shift(1)<=p['rsi_buy_th'])).astype(int) +
        (s['Close']<=s['BB_l']*1.01).astype(int) +
        (s['ADX']>p['adx_th']).astype(int) +
        (sxu&(s['SK']<p['stoch_buy_th'])).astype(int)
    )
    ssc=(
        (s['SMA25']<s['SMA75']).astype(int)*p['w_trend'] +
        mxd.astype(int)*p['w_macd'] +
        ((s['RSI']<p['rsi_sell_th'])&(s['RSI'].shift(1)>=p['rsi_sell_th'])).astype(int) +
        (s['Close']>=s['BB_u']*0.99).astype(int) +
        (s['ADX']>p['adx_th']).astype(int) +
        (sxd&(s['SK']>p['stoch_sell_th'])).astype(int)
    )
    df=df.copy()
    df['bsc']=bsc; df['ssc']=ssc; df['sig']=0
    df.loc[bsc>=p['buy_th'], 'sig']=1
    df.loc[ssc>=p['sell_th'], 'sig']=-1
    df['trend']='Range'
    df.loc[(s['SMA25']>s['SMA75'])&(s['SMA75']>s['SMA200'])&(s['ADX']>20), 'trend']='Up'
    df.loc[(s['SMA25']<s['SMA75'])&(s['SMA75']<s['SMA200'])&(s['ADX']>20), 'trend']='Down'
    return df

@st.cache_data(ttl=60, show_spinner=False)
def fetch_raw(code, period, interval):
    try:
        df=yf.download(code, period=period, interval=interval, auto_adjust=True, progress=False)
        if df is None or df.empty: return None
        df=flatten_df(df)
        df=df.dropna(subset=['Close','Open','High','Low','Volume'])
        if len(df)<120: return None
        return df
    except:
        return None

def run_backtest(df, cost=0.001, initial_equity=1.0):
    cl=df['Close'].values; sig=df['sig'].values; dates=df.index
    trades=[]; eq=initial_equity; equity=[eq]; pos=0; ep=0.0; ed=None
    
    for i in range(1, len(df)):
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
        
    eq_s=pd.Series(equity, index=dates)
    bh=(cl[-1]-cl[0])/cl[0]*100
    n=len(trades)
    
    # 統計計算
    wins=[t for t in trades if t['ret']>0]
    loss=[t for t in trades if t['ret']<=0]
    wr=len(wins)/n*100 if n>0 else 0
    aw=np.mean([t['ret'] for t in wins]) if wins else 0
    al=np.mean([t['ret'] for t in loss]) if loss else 0
    pf=abs(sum(t['ret'] for t in wins)/sum(t['ret'] for t in loss)) if loss else 999.0
    
    roll_max=eq_s.cummax()
    dd=(eq_s-roll_max)/roll_max*100
    mdd=dd.min()
    yrs=(dates[-1]-dates[0]).days/365.25
    cagr=((eq/initial_equity)**(1/yrs)-1)*100 if yrs>0 else 0
    dr=eq_s.pct_change().dropna()
    sharpe=dr.mean()/dr.std()*np.sqrt(252) if dr.std()>0 else 0

    return {'trades':trades, 'equity':eq_s, 'drawdown':dd,
            'stats':{'n':n,'wr':wr,'aw':aw,'al':al,'pf':pf,
                     'sr':(eq/initial_equity-1)*100,'bh':bh,'mdd':mdd,'cagr':cagr,'sharpe':sharpe}}

def score_params(df, p, cost):
    df2=compute_signals(df, p)
    bt=run_backtest(df2, cost)
    if bt is None or bt['stats']['n'] < 2: return -999
    s=bt['stats']
    return s['sr'] - max(0, -s['mdd'])*0.5 + s['sharpe']*3

# --- ウォークフォワード最適化（修正版） ---
@st.cache_data(ttl=300, show_spinner=False)
def walk_forward_optimize(code, period, interval, n_splits=4, cost=0.001):
    raw=fetch_raw(code, period, interval)
    if raw is None: return None
    base=compute_indicators(raw.copy())
    
    keys=list(PARAM_GRID.keys())
    vals=list(PARAM_GRID.values())
    combos=list(itertools.product(*vals))
    all_params=[dict(zip(keys, c)) for c in combos]

    n=len(base)
    split_size=n // (n_splits + 1)
    
    fold_results=[]
    combined_trades=[]
    combined_equity_series = []
    current_initial_eq = 1.0

    progress=st.progress(0, 'Walk-forward optimization...')

    for fold in range(n_splits):
        train_end=(fold+1)*split_size
        test_start=train_end
        test_end=test_start + split_size if fold < n_splits-1 else n

        train=base.iloc[:train_end].copy()
        test =base.iloc[test_start:test_end].copy()

        if len(train)<60 or len(test)<20: continue

        # トレーニング期間で最適化
        best_score_fold=-999
        best_idx_fold=0
        for idx, p in enumerate(all_params):
            s=score_params(train, p, cost)
            if s > best_score_fold:
                best_score_fold=s
                best_idx_fold=idx
        
        best_p = all_params[best_idx_fold]
        
        # テスト期間（未知データ）に適用
        test_df = compute_signals(test.copy(), best_p)
        test_bt = run_backtest(test_df, cost, initial_equity=current_initial_eq)
        
        if test_bt:
            combined_trades.extend(test_bt['trades'])
            combined_equity_series.append(test_bt['equity'])
            current_initial_eq = test_bt['equity'].iloc[-1]

        fold_results.append({
            'fold':fold+1,
            'test_start':base.index[test_start].strftime('%Y/%m'),
            'test_end':base.index[min(test_end-1, n-1)].strftime('%Y/%m'),
            'best_params':best_p,
            'test_bt':test_bt,
        })
        progress.progress((fold+1)/n_splits, f'Fold {fold+1}/{n_splits} done')

    progress.empty()

    if not combined_equity_series: return None
    
    # 全テスト期間のEquityを統合
    full_equity = pd.concat(combined_equity_series)
    full_equity = full_equity[~full_equity.index.duplicated(keep='first')]
    
    # 統計再計算
    roll_max = full_equity.cummax()
    full_dd = (full_equity - roll_max) / roll_max * 100
    
    full_stats = {
        'n': len(combined_trades),
        'sr': (full_equity.iloc[-1] - 1.0) * 100,
        'bh': ((base['Close'].iloc[-1] - base['Close'].iloc[0]) / base['Close'].iloc[0]) * 100,
        'mdd': full_dd.min(),
        'sharpe': full_equity.pct_change().mean() / full_equity.pct_change().std() * np.sqrt(252) if full_equity.pct_change().std() > 0 else 0
    }

    # デフォルト設定での比較用バックテスト
    default_df=compute_signals(base.copy(), DEFAULT_PARAMS)
    default_bt=run_backtest(default_df, cost)

    return {
        'best_params': fold_results[-1]['best_params'], # 最新の最適パラメータ
        'full_bt': {'trades': combined_trades, 'equity': full_equity, 'drawdown': full_dd, 'stats': full_stats},
        'default_bt': default_bt,
        'fold_results': fold_results,
        'base': base,
    }

# --- グラフ作成関数 ---
def make_bt_chart(bt, title, bt2=None):
    eq=bt['equity']; dd=bt['drawdown']; trades=bt['trades']
    fig, axes=plt.subplots(2, 1, figsize=(14, 8), facecolor=C['bg'],
                         gridspec_kw={'height_ratios':[3, 1], 'hspace':0.06})
    for ax in axes:
        ax.set_facecolor(C['panel'])
        ax.tick_params(colors=C['sub'], labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor(C['grid'])
        ax.grid(color=C['grid'], lw=0.5, ls='--', alpha=0.6)
    
    ax1, ax2=axes
    ax1.plot(eq.index, eq.values, color=C['buy'], lw=2.0, label='Walk-Forward Strategy', zorder=3)
    if bt2 is not None:
        ax1.plot(bt2['equity'].index, bt2['equity'].values, color='#ffa657', lw=1.5, ls='-.', label='Default Strategy', zorder=2)
    
    ax1.axhline(1.0, color=C['grid'], lw=0.8)
    ax1.legend(loc='upper left', fontsize=8, facecolor=C['bg'], edgecolor=C['grid'], labelcolor=C['text'])
    
    ax2.fill_between(dd.index, dd.values, 0, where=dd.values<0, color=C['sell'], alpha=0.45)
    ax2.set_ylabel('Drawdown%', color=C['sub'], fontsize=9)
    plt.close('all')
    return fig

# --- [中略: make_chart, draw_candles 等の描画系は変更なしのため統合して構成] ---
# (以下、既存の描画ロジックとStreamlit UI部分)

def draw_candles(ax,df):
    op=df['Open'].values; hi=df['High'].values
    lo=df['Low'].values; cl=df['Close'].values
    for i in range(len(df)):
        col=C['cup'] if cl[i]>=op[i] else C[['cdn']]
        ax.plot([i,i],[lo[i],hi[i]],color=col,lw=0.7,zorder=2)
        b0=min(op[i],cl[i]); b1=max(op[i],cl[i])
        ax.bar(i,max(b1-b0,1e-6),bottom=b0,width=0.6,color=col,linewidth=0,zorder=3)

def make_chart(df,title,mobile=False):
    w,h=(9,14) if mobile else (16,13)
    fig=plt.figure(figsize=(w,h),facecolor=C['bg'])
    gs=gridspec.GridSpec(5,1,figure=fig,height_ratios=[4,1,1.3,1.3,1.3],hspace=0.05)
    axes=[fig.add_subplot(gs[i]) for i in range(5)]
    for ax in axes:
        ax.set_facecolor(C['panel'])
        ax.tick_params(colors=C['sub'],labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor(C['grid'])
    # ... (描画ロジック詳細は元のコードと同じ)
    return fig

# --- Streamlit アプリ本体 ---
st.set_page_config(page_title='Trend Signal PRO', layout='wide')
st.markdown(f"""<style>
.stApp {{background:#0d1117; color:#e6edf3;}}
[data-testid="stMetricValue"] {{color:#e6edf3!important; font-size:1.2rem!important;}}
.bb{{background:#1a3d24;color:#3fb950;border:1px solid #3fb950;padding:5px 15px;border-radius:5px;font-weight:bold;}}
.bs{{background:#3d1a1a;color:#f85149;border:1px solid #f85149;padding:5px 15px;border-radius:5px;font-weight:bold;}}
</style>""", unsafe_allow_html=True)

if 'active_params' not in st.session_state: st.session_state['active_params']=DEFAULT_PARAMS
if 'wf_result' not in st.session_state: st.session_state['wf_result']=None

with st.sidebar:
    st.title("Settings")
    code = st.text_input("Ticker", "AAPL").upper()
    period = st.selectbox("Period", PERIODS, index=2)
    interval = st.selectbox("Interval", INTERVALS, index=0)
    if st.button("Analyze", type="primary", use_container_width=True):
        raw = fetch_raw(code, period, interval)
        if raw is not None:
            df = compute_indicators(raw.copy())
            df = compute_signals(df, st.session_state['active_params'])
            st.session_state['result'] = {'df': df, 'at': datetime.now().strftime('%H:%M:%S')}
        else:
            st.error("Data fetch failed.")

res = st.session_state.get('result')
if res:
    df = res['df']
    tab1, tab2, tab3 = st.tabs(['Live Chart', 'Backtest', 'Walk-Forward Optimization'])
    
    with tab1:
        st.markdown(f"### {code} Signal Analysis")
        # チャート表示
    
    with tab2:
        bt = run_backtest(df)
        if bt:
            st.metric("Return", f"{bt['stats']['sr']:.2f}%")
            st.pyplot(make_bt_chart(bt, "Backtest Result"))

    with tab3:
        n_splits = st.slider("Splits", 2, 6, 4)
        if st.button("Run WF Optimization"):
            wf = walk_forward_optimize(code, period, interval, n_splits=n_splits)
            st.session_state['wf_result'] = wf
        
        wf = st.session_state.get('wf_result')
        if wf:
            st.success("Walk-Forward complete. Results below show concatenated out-of-sample tests.")
            s = wf['full_bt']['stats']
            c1, c2, c3 = st.columns(3)
            c1.metric("Concatenated Return", f"{s['sr']:.1f}%")
            c2.metric("Max Drawdown", f"{s['mdd']:.1f}%")
            c3.metric("Trades", s['n'])
            st.pyplot(make_bt_chart(wf['full_bt'], "WF Out-of-Sample Performance", bt2=wf['default_bt']))
            
            st.markdown("#### Fold Details")
            for r in wf['fold_results']:
                with st.expander(f"Fold {r['fold']}: {r['test_start']} - {r['test_end']}"):
                    st.write("Best Params:", r['best_params'])
                    if r['test_bt']:
                        st.write(f"Test Return: {r['test_bt']['stats']['sr']:.2f}%")
