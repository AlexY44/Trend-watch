"""
Trend Signal PRO - 改善版
主な変更点:
  1. 戦略強化: ATRトレーリングストップ・トレンドフィルター・出来高確認を追加
  2. WF最適化の安定化:
     - Expanding Window方式（train期間を累積で拡大）に変更
     - スコア関数をCalmar比率ベースに変更（最近期間の急騰バイアス排除）
     - 最低トレード数チェック強化
  3. Buy & Holdラインをチャートに表示
"""
import warnings, itertools
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
PERIODS   = ['2y', '3y', '5y', '10y']
INTERVALS = ['1d', '1wk']
POPULAR = [
    ('1326.T','SPDR Gold'), ('7203.T','Toyota'),  ('6758.T','Sony'),
    ('9984.T','SBG'),       ('6861.T','Keyence'),  ('8306.T','MUFG'),
    ('^N225', 'Nikkei225'),('AAPL',  'Apple'),    ('NVDA',  'NVIDIA'), ('^GSPC','SP500'),
]
C = {
    'bg':'#0d1117','panel':'#161b22','grid':'#21262d',
    'text':'#e6edf3','sub':'#8b949e',
    'buy':'#3fb950','sell':'#f85149','neutral':'#58a6ff',
    'sma25':'#ffa657','sma75':'#58a6ff','sma200':'#bc8cff',
    'bb':'#388bfd','macd':'#58a6ff','msig':'#ffa657',
    'hup':'#3fb950','hdn':'#f85149','rsi':'#d2a8ff',
    'cup':'#3fb950','cdn':'#f85149','bh':'#e8c55a',
}

DEFAULT_PARAMS = {
    'w_trend':1, 'w_macd':2, 'w_vol':1,
    'rsi_buy_th':35, 'rsi_sell_th':65,
    'adx_th':20,
    'stoch_buy_th':30, 'stoch_sell_th':70,
    'buy_th':3, 'sell_th':3,
    'atr_stop_mult':2.0,   # ATRストップ倍率
    'trail_stop':True,     # トレーリングストップ有効
}

PARAM_GRID = {
    'w_trend':       [1, 2],
    'w_macd':        [1, 2, 3],
    'w_vol':         [0, 1],
    'rsi_buy_th':    [30, 35, 40],
    'rsi_sell_th':   [60, 65, 70],
    'adx_th':        [15, 20, 25],
    'stoch_buy_th':  [25, 30, 35],
    'stoch_sell_th': [65, 70, 75],
    'buy_th':        [2, 3, 4],
    'sell_th':       [2, 3, 4],
    'atr_stop_mult': [1.5, 2.0, 2.5],
}

# =========================================================
# --- データ・インジケーター
# =========================================================
def flatten_df(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df

def compute_indicators(df):
    cl = df['Close']; hi = df['High']; lo = df['Low']; vo = df['Volume']
    # トレンド系
    df['SMA25']  = cl.rolling(25).mean()
    df['SMA75']  = cl.rolling(75).mean()
    df['SMA200'] = cl.rolling(200).mean()
    df['EMA20']  = cl.ewm(span=20, adjust=False).mean()  # ★追加: EMAでより敏感に
    # ボリンジャーバンド
    bb = ta.volatility.BollingerBands(cl, 20, 2)
    df['BB_u'] = bb.bollinger_hband()
    df['BB_m'] = bb.bollinger_mavg()
    df['BB_l'] = bb.bollinger_lband()
    df['BB_w'] = (df['BB_u'] - df['BB_l']) / df['BB_m']
    df['BB_pct'] = bb.bollinger_pband()   # ★追加: BB位置(0-1)
    # MACD
    mc = ta.trend.MACD(cl, 26, 12, 9)
    df['MACD'] = mc.macd(); df['MSIG'] = mc.macd_signal(); df['MHST'] = mc.macd_diff()
    # モメンタム
    df['RSI']   = ta.momentum.RSIIndicator(cl, 14).rsi()
    df['RSI_s'] = df['RSI'].rolling(3).mean()   # ★追加: RSIスムーズ
    sto = ta.momentum.StochasticOscillator(hi, lo, cl, 14, 3)
    df['SK'] = sto.stoch(); df['SD'] = sto.stoch_signal()
    # ボラティリティ
    df['ATR'] = ta.volatility.AverageTrueRange(hi, lo, cl, 14).average_true_range()
    df['ATR_pct'] = df['ATR'] / cl   # ★追加: ATR率（正規化）
    # ADX
    adx = ta.trend.ADXIndicator(hi, lo, cl, 14)
    df['ADX'] = adx.adx()
    df['DI_plus']  = adx.adx_pos()   # ★追加: +DI
    df['DI_minus'] = adx.adx_neg()   # ★追加: -DI
    # 出来高
    df['VMA']    = vo.rolling(20).mean()
    df['V_ratio'] = vo / df['VMA'].replace(0, np.nan)   # ★追加: 出来高比率
    # モメンタム（価格変化率）
    df['MOM']  = cl.pct_change(10) * 100   # ★追加: 10日モメンタム
    return df

def compute_signals(df, p):
    p = {**DEFAULT_PARAMS, **p}
    s = df

    # ---- クロスシグナル ----
    mxu = (s['MACD'] > s['MSIG']) & (s['MACD'].shift(1) <= s['MSIG'].shift(1))
    mxd = (s['MACD'] < s['MSIG']) & (s['MACD'].shift(1) >= s['MSIG'].shift(1))
    sxu = (s['SK']   > s['SD'])   & (s['SK'].shift(1)   <= s['SD'].shift(1))
    sxd = (s['SK']   < s['SD'])   & (s['SK'].shift(1)   >= s['SD'].shift(1))

    # ---- トレンドフィルター（強化版）----
    # 上昇トレンド: SMA25>SMA75 かつ EMA20が上向き かつ +DI>-DI
    up_trend   = (s['SMA25'] > s['SMA75']) & \
                 (s['EMA20'] > s['EMA20'].shift(3)) & \
                 (s['DI_plus'] > s['DI_minus'])
    down_trend = (s['SMA25'] < s['SMA75']) & \
                 (s['EMA20'] < s['EMA20'].shift(3)) & \
                 (s['DI_plus'] < s['DI_minus'])

    # ---- 出来高フィルター ----
    vol_confirm = s['V_ratio'] > 1.2   # 平均出来高の1.2倍以上

    # ---- 買いスコア ----
    bsc = (
        up_trend.astype(int) * p['w_trend'] +
        mxu.astype(int) * p['w_macd'] +
        # RSIスムーズ版で誤シグナル軽減
        ((s['RSI_s'] > p['rsi_buy_th']) & (s['RSI_s'].shift(1) <= p['rsi_buy_th'])).astype(int) +
        # BBバンド下限 + BB位置が低い
        ((s['Close'] <= s['BB_l'] * 1.02) & (s['BB_pct'] < 0.2)).astype(int) +
        (s['ADX'] > p['adx_th']).astype(int) +
        (sxu & (s['SK'] < p['stoch_buy_th'])).astype(int) +
        # ★出来高確認
        (vol_confirm).astype(int) * p['w_vol'] +
        # ★モメンタム（底打ちから回復）
        ((s['MOM'] > -5) & (s['MOM'].shift(5) < -5)).astype(int)
    )
    # ---- 売りスコア ----
    ssc = (
        down_trend.astype(int) * p['w_trend'] +
        mxd.astype(int) * p['w_macd'] +
        ((s['RSI_s'] < p['rsi_sell_th']) & (s['RSI_s'].shift(1) >= p['rsi_sell_th'])).astype(int) +
        ((s['Close'] >= s['BB_u'] * 0.98) & (s['BB_pct'] > 0.8)).astype(int) +
        (s['ADX'] > p['adx_th']).astype(int) +
        (sxd & (s['SK'] > p['stoch_sell_th'])).astype(int) +
        (vol_confirm).astype(int) * p['w_vol'] +
        ((s['MOM'] < 5) & (s['MOM'].shift(5) > 5)).astype(int)
    )

    df = df.copy()
    df['bsc'] = bsc; df['ssc'] = ssc; df['sig'] = 0
    df.loc[bsc >= p['buy_th'],  'sig'] =  1
    df.loc[ssc >= p['sell_th'], 'sig'] = -1

    # ---- トレンド判定 ----
    df['trend'] = 'Range'
    df.loc[up_trend   & (s['ADX'] > 20), 'trend'] = 'Up'
    df.loc[down_trend & (s['ADX'] > 20), 'trend'] = 'Down'
    return df

@st.cache_data(ttl=60, show_spinner=False)
def fetch_raw(code, period, interval):
    try:
        df = yf.download(code, period=period, interval=interval, auto_adjust=True, progress=False)
        if df is None or df.empty: return None
        df = flatten_df(df)
        df = df.dropna(subset=['Close','Open','High','Low','Volume'])
        if len(df) < 150: return None
        return df
    except:
        return None

# =========================================================
# --- バックテスト（ATRトレーリングストップ追加）
# =========================================================
def run_backtest(df, cost=0.001, initial_equity=1.0, atr_mult=2.0, use_trail=True):
    cl  = df['Close'].values
    hi  = df['High'].values
    sig = df['sig'].values
    atr = df['ATR'].values
    dates = df.index

    trades = []; eq = initial_equity; equity = [eq]
    pos = 0; ep = 0.0; ed = None
    trail_stop = 0.0   # トレーリングストップ価格
    peak_price = 0.0   # ポジション中の高値

    for i in range(1, len(df)):
        px = cl[i]; ps = sig[i - 1]

        if pos == 1:
            # ★ ATRトレーリングストップ更新
            if use_trail:
                peak_price = max(peak_price, hi[i])
                trail_stop = peak_price - atr[i] * atr_mult

            # ストップアウト or シグナル売り
            stop_hit = use_trail and px < trail_stop
            if ps == -1 or stop_hit:
                xp  = px * (1 - cost)
                ret = (xp - ep) / ep
                eq *= (1 + ret)
                trades.append({
                    'entry_date': ed, 'exit_date': dates[i],
                    'entry': ep, 'exit': xp, 'ret': ret * 100,
                    'result': 'Win' if ret > 0 else 'Loss',
                    'exit_type': 'Stop' if stop_hit else 'Signal',
                })
                pos = 0; trail_stop = 0.0; peak_price = 0.0

        elif pos == 0 and ps == 1:
            pos = 1
            ep  = px * (1 + cost)
            ed  = dates[i]
            peak_price = px
            trail_stop = px - atr[i] * atr_mult

        equity.append(eq)

    eq_s      = pd.Series(equity, index=dates)
    bh_series = pd.Series((cl / cl[0]) * initial_equity, index=dates)
    bh_pct    = (cl[-1] - cl[0]) / cl[0] * 100

    n    = len(trades)
    wins = [t for t in trades if t['ret'] > 0]
    loss = [t for t in trades if t['ret'] <= 0]
    wr   = len(wins) / n * 100 if n > 0 else 0
    aw   = np.mean([t['ret'] for t in wins]) if wins else 0
    al   = np.mean([t['ret'] for t in loss]) if loss else 0
    pf   = abs(sum(t['ret'] for t in wins) / sum(t['ret'] for t in loss)) if loss else 999.0

    roll_max = eq_s.cummax()
    dd   = (eq_s - roll_max) / roll_max * 100
    mdd  = dd.min()
    yrs  = (dates[-1] - dates[0]).days / 365.25
    cagr = ((eq / initial_equity) ** (1 / yrs) - 1) * 100 if yrs > 0 else 0
    dr   = eq_s.pct_change().dropna()
    sharpe = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    # ★ Calmar比率（CAGR / 最大DD）過去全期間安定性の指標
    calmar = cagr / abs(mdd) if mdd < 0 else 0.0

    return {
        'trades': trades, 'equity': eq_s, 'bh_series': bh_series, 'drawdown': dd,
        'stats': {
            'n':n,'wr':wr,'aw':aw,'al':al,'pf':pf,
            'sr':(eq/initial_equity-1)*100, 'bh':bh_pct,
            'mdd':mdd,'cagr':cagr,'sharpe':sharpe,'calmar':calmar,
        },
    }

def score_params(df, p, cost):
    """
    スコア関数（安定性重視版）
    -------------------------------------------------------
    問題: 従来スコアはSharpe重視→最近の強いトレンド期間に
          たまたまフィットしたパラメータが選ばれ、
          テスト期間では急に良くなる不安定なカーブが出る。

    改善:
    - Calmar比率（CAGR/最大DD）をメインスコアに
      → 全期間を通じた安定リターンを評価
    - 最低トレード数ガード（5以上）
    - 勝率ペナルティ（40%未満で減点）
    - DDペナルティ（-30%超は大きく減点）
    """
    full_p = {**DEFAULT_PARAMS, **p}
    atr_mult  = full_p.pop('atr_stop_mult', 2.0)
    use_trail = full_p.pop('trail_stop', True)
    df2 = compute_signals(df, full_p)
    bt  = run_backtest(df2, cost, atr_mult=atr_mult, use_trail=use_trail)
    if bt is None: return -999
    s = bt['stats']
    if s['n'] < 5: return -999               # 最低5トレード
    if s['mdd'] < -50: return -999           # 壊滅的DDは除外

    calmar   = s['calmar']
    wr_pen   = max(0, (40 - s['wr']) * 0.1)  # 勝率40%未満ペナルティ
    dd_pen   = max(0, (-s['mdd'] - 30) * 0.3) # DD30%超ペナルティ
    return calmar - wr_pen - dd_pen

# =========================================================
# --- ウォークフォワード最適化（Expanding Window方式）
# =========================================================
def walk_forward_optimize(code, period, interval, n_splits=4, cost=0.001):
    """
    Expanding Window Walk-Forward（安定版）
    -----------------------------------------------
    従来のSliding Window問題:
      各Foldのtrain期間が短い→過学習しやすい→テスト期間の
      「たまたまその時期だけ良い」パラメータが選ばれる→
      最後のFoldだけ急にリターンが良く見える

    改善: Expanding Window
      Fold1: [0 → split1] でtrain, [split1 → split2] でtest
      Fold2: [0 → split2] でtrain, [split2 → split3] でtest
      ...
      → trainが積み上がるほど安定したパラメータが選ばれる
      → データが増えるほど選定精度向上
    """
    raw = fetch_raw(code, period, interval)
    if raw is None: return None
    base = compute_indicators(raw.copy())

    keys       = list(PARAM_GRID.keys())
    vals       = list(PARAM_GRID.values())
    all_params = [dict(zip(keys, c)) for c in itertools.product(*vals)]

    n          = len(base)
    # 最初のtrain期間を全体の40%確保（安定した最適化のため）
    min_train  = max(int(n * 0.4), 60)
    test_size  = (n - min_train) // n_splits
    if test_size < 20:
        st.error("データが少なすぎます。期間を長くしてください。")
        return None

    fold_results           = []
    combined_trades        = []
    combined_equity_series = []
    current_eq             = 1.0

    progress = st.progress(0, 'Walk-forward optimization (Expanding Window)...')

    for fold in range(n_splits):
        # Expanding: trainは常に先頭から
        test_start = min_train + fold * test_size
        test_end   = test_start + test_size if fold < n_splits - 1 else n
        train_end  = test_start   # trainは先頭〜test直前まで全部使う

        train = base.iloc[:train_end].copy()
        test  = base.iloc[test_start:test_end].copy()

        if len(train) < 60 or len(test) < 20: continue

        # --- train期間で最良パラメータ探索 ---
        best_score = -999; best_p = DEFAULT_PARAMS
        for p in all_params:
            sc = score_params(train, p, cost)
            if sc > best_score:
                best_score = sc; best_p = p

        # --- test期間（未知）に適用 ---
        full_p    = {**DEFAULT_PARAMS, **best_p}
        atr_mult  = full_p.pop('atr_stop_mult', 2.0)
        use_trail = full_p.pop('trail_stop', True)
        test_df   = compute_signals(test.copy(), full_p)
        test_bt   = run_backtest(test_df, cost, initial_equity=current_eq,
                                 atr_mult=atr_mult, use_trail=use_trail)

        if test_bt:
            combined_trades.extend(test_bt['trades'])
            combined_equity_series.append(test_bt['equity'])
            current_eq = test_bt['equity'].iloc[-1]

        fold_results.append({
            'fold':        fold + 1,
            'train_size':  len(train),
            'test_start':  base.index[test_start].strftime('%Y/%m'),
            'test_end':    base.index[min(test_end-1, n-1)].strftime('%Y/%m'),
            'best_params': best_p,
            'best_score':  round(best_score, 3),
            'test_bt':     test_bt,
        })
        progress.progress((fold+1)/n_splits, f'Fold {fold+1}/{n_splits} complete')

    progress.empty()
    if not combined_equity_series: return None

    full_equity = pd.concat(combined_equity_series)
    full_equity = full_equity[~full_equity.index.duplicated(keep='first')]

    # WF期間BaH（同スケール）
    bh_wf = pd.Series(
        (base.loc[full_equity.index,'Close'] /
         base.loc[full_equity.index[0],'Close']).values,
        index=full_equity.index,
    )
    roll_max = full_equity.cummax()
    full_dd  = (full_equity - roll_max) / roll_max * 100
    pct_chg  = full_equity.pct_change().dropna()

    full_stats = {
        'n':      len(combined_trades),
        'sr':     (full_equity.iloc[-1] - 1.0) * 100,
        'bh':     (bh_wf.iloc[-1] - 1.0) * 100,
        'mdd':    full_dd.min(),
        'sharpe': pct_chg.mean()/pct_chg.std()*np.sqrt(252) if pct_chg.std()>0 else 0,
    }

    # Default戦略（比較用）
    default_p  = {**DEFAULT_PARAMS}
    atr_mult_d = default_p.pop('atr_stop_mult', 2.0)
    trail_d    = default_p.pop('trail_stop', True)
    default_df = compute_signals(base.copy(), default_p)
    default_bt = run_backtest(default_df, cost, atr_mult=atr_mult_d, use_trail=trail_d)

    return {
        'best_params':  fold_results[-1]['best_params'],
        'full_bt': {
            'trades':    combined_trades,
            'equity':    full_equity,
            'bh_series': bh_wf,
            'drawdown':  full_dd,
            'stats':     full_stats,
        },
        'default_bt':   default_bt,
        'fold_results': fold_results,
        'base':         base,
    }

# =========================================================
# --- 描画関数
# =========================================================
def _style_ax(ax):
    ax.set_facecolor(C['panel'])
    ax.tick_params(colors=C['sub'], labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor(C['grid'])
    ax.grid(color=C['grid'], lw=0.5, ls='--', alpha=0.5)

def draw_candles(ax, df):
    op=df['Open'].values; hi=df['High'].values
    lo=df['Low'].values;  cl=df['Close'].values
    for i in range(len(df)):
        col = C['cup'] if cl[i] >= op[i] else C['cdn']
        ax.plot([i,i],[lo[i],hi[i]], color=col, lw=0.7, zorder=2)
        b0=min(op[i],cl[i]); b1=max(op[i],cl[i])
        ax.bar(i, max(b1-b0,1e-6), bottom=b0, width=0.6, color=col, linewidth=0, zorder=3)

def make_chart(df, title, mobile=False):
    w,h = (9,14) if mobile else (16,13)
    fig  = plt.figure(figsize=(w,h), facecolor=C['bg'])
    gs   = gridspec.GridSpec(5,1, figure=fig,
                             height_ratios=[4,1,1.3,1.3,1.3], hspace=0.05)
    axes = [fig.add_subplot(gs[i]) for i in range(5)]
    for ax in axes: _style_ax(ax)

    display_n = 200
    plot_df = df.iloc[-display_n:].copy().reset_index(drop=False)
    n = len(plot_df); xs = np.arange(n)

    # パネル0: ローソク足
    ax0 = axes[0]
    draw_candles(ax0, plot_df)
    ax0.plot(xs, plot_df['SMA25'],  color=C['sma25'],  lw=1.2, label='SMA25',  zorder=4)
    ax0.plot(xs, plot_df['SMA75'],  color=C['sma75'],  lw=1.2, label='SMA75',  zorder=4)
    ax0.plot(xs, plot_df['SMA200'], color=C['sma200'], lw=1.2, label='SMA200', zorder=4)
    ax0.plot(xs, plot_df['EMA20'],  color='#ff79c6',   lw=0.9, ls='--', label='EMA20', zorder=4)
    ax0.plot(xs, plot_df['BB_u'], color=C['bb'], lw=0.8, ls=':', alpha=0.8)
    ax0.plot(xs, plot_df['BB_l'], color=C['bb'], lw=0.8, ls=':', alpha=0.8)
    ax0.fill_between(xs, plot_df['BB_u'], plot_df['BB_l'], color=C['bb'], alpha=0.07)
    buy_idx  = plot_df.index[plot_df['sig']==1].tolist()
    sell_idx = plot_df.index[plot_df['sig']==-1].tolist()
    if buy_idx:
        ax0.scatter(buy_idx,  plot_df.loc[buy_idx,'Low']*0.995,
                    marker='^', color=C['buy'],  s=65, zorder=6, label='Buy')
    if sell_idx:
        ax0.scatter(sell_idx, plot_df.loc[sell_idx,'High']*1.005,
                    marker='v', color=C['sell'], s=65, zorder=6, label='Sell')
    ax0.set_title(title, color=C['text'], fontsize=11, pad=6)
    ax0.legend(loc='upper left', fontsize=7, facecolor=C['bg'],
               edgecolor=C['grid'], labelcolor=C['text'], ncol=5)
    ax0.set_xlim(-1,n); ax0.set_xticks([]); ax0.set_ylabel('Price', color=C['sub'], fontsize=8)

    # パネル1: 出来高
    ax1 = axes[1]
    vcols = [C['cup'] if plot_df['Close'].iloc[i]>=plot_df['Open'].iloc[i] else C['cdn'] for i in range(n)]
    ax1.bar(xs, plot_df['Volume'], color=vcols, width=0.7, alpha=0.8, zorder=3)
    ax1.plot(xs, plot_df['VMA'], color=C['neutral'], lw=1.0, zorder=4)
    ax1.set_xlim(-1,n); ax1.set_xticks([]); ax1.set_ylabel('Vol', color=C['sub'], fontsize=8)

    # パネル2: MACD
    ax2 = axes[2]
    ax2.plot(xs, plot_df['MACD'], color=C['macd'], lw=1.2, label='MACD', zorder=3)
    ax2.plot(xs, plot_df['MSIG'], color=C['msig'], lw=1.0, ls='--', label='Signal', zorder=3)
    hist = plot_df['MHST'].values
    hcols = [C['hup'] if v>=0 else C['hdn'] for v in hist]
    ax2.bar(xs, hist, color=hcols, width=0.7, alpha=0.75, zorder=2)
    ax2.axhline(0, color=C['grid'], lw=0.8)
    ax2.set_xlim(-1,n); ax2.set_xticks([])
    ax2.set_ylabel('MACD', color=C['sub'], fontsize=8)
    ax2.legend(loc='upper left', fontsize=7, facecolor=C['bg'],
               edgecolor=C['grid'], labelcolor=C['text'])

    # パネル3: RSI
    ax3 = axes[3]
    ax3.plot(xs, plot_df['RSI'],   color=C['rsi'], lw=1.2, label='RSI', zorder=3)
    ax3.plot(xs, plot_df['RSI_s'], color='#79c0ff', lw=0.9, ls='--', label='RSI(3)', zorder=3)
    ax3.axhline(70, color=C['sell'], lw=0.8, ls='--', alpha=0.7)
    ax3.axhline(30, color=C['buy'],  lw=0.8, ls='--', alpha=0.7)
    ax3.axhline(50, color=C['grid'], lw=0.6)
    ax3.fill_between(xs, plot_df['RSI'], 70, where=plot_df['RSI']>=70, color=C['sell'], alpha=0.2)
    ax3.fill_between(xs, plot_df['RSI'], 30, where=plot_df['RSI']<=30, color=C['buy'],  alpha=0.2)
    ax3.set_ylim(0,100); ax3.set_xlim(-1,n); ax3.set_xticks([])
    ax3.set_ylabel('RSI', color=C['sub'], fontsize=8)
    ax3.legend(loc='upper left', fontsize=7, facecolor=C['bg'],
               edgecolor=C['grid'], labelcolor=C['text'])

    # パネル4: Stochastic
    ax4 = axes[4]
    ax4.plot(xs, plot_df['SK'], color=C['buy'],  lw=1.1, label='%K', zorder=3)
    ax4.plot(xs, plot_df['SD'], color=C['msig'], lw=1.0, ls='--', label='%D', zorder=3)
    ax4.axhline(75, color=C['sell'], lw=0.8, ls='--', alpha=0.7)
    ax4.axhline(25, color=C['buy'],  lw=0.8, ls='--', alpha=0.7)
    ax4.set_ylim(0,100); ax4.set_xlim(-1,n)
    ax4.set_ylabel('Stoch', color=C['sub'], fontsize=8)
    ax4.legend(loc='upper left', fontsize=7, facecolor=C['bg'],
               edgecolor=C['grid'], labelcolor=C['text'])

    tick_step = max(1, n//10)
    ticks = list(range(0, n, tick_step))
    dcol  = plot_df.columns[0]
    try:    labels = [str(plot_df.iloc[i][dcol])[:10] for i in ticks]
    except: labels = [str(i) for i in ticks]
    ax4.set_xticks(ticks)
    ax4.set_xticklabels(labels, rotation=30, ha='right', fontsize=6, color=C['sub'])
    plt.tight_layout(pad=0.5)
    return fig

def make_bt_chart(bt, title, bt2=None, label='Strategy'):
    eq = bt['equity']; dd = bt['drawdown']; bh = bt.get('bh_series')
    fig, axes = plt.subplots(2,1, figsize=(14,8), facecolor=C['bg'],
                             gridspec_kw={'height_ratios':[3,1],'hspace':0.06})
    for ax in axes: _style_ax(ax)
    ax1, ax2 = axes

    if bh is not None:
        ax1.plot(bh.index, bh.values, color=C['bh'], lw=1.4, ls=':', label='Buy & Hold', zorder=2)
    ax1.plot(eq.index, eq.values, color=C['buy'], lw=2.0, label=label, zorder=3)
    if bt2 is not None:
        ax1.plot(bt2['equity'].index, bt2['equity'].values,
                 color='#ffa657', lw=1.5, ls='-.', label='Default Strategy', zorder=2)
    ax1.axhline(1.0, color=C['grid'], lw=0.8)
    ax1.set_title(title, color=C['text'], fontsize=11)
    ax1.set_ylabel('Equity (normalized)', color=C['sub'], fontsize=9)
    ax1.legend(loc='upper left', fontsize=8, facecolor=C['bg'],
               edgecolor=C['grid'], labelcolor=C['text'])
    ax1.tick_params(axis='x', colors=C['sub'], labelsize=7)

    ax2.fill_between(dd.index, dd.values, 0, where=dd.values<0, color=C['sell'], alpha=0.45)
    ax2.axhline(0, color=C['grid'], lw=0.8)
    ax2.set_ylabel('Drawdown%', color=C['sub'], fontsize=9)
    ax2.tick_params(axis='x', colors=C['sub'], labelsize=7)
    plt.tight_layout(pad=0.5)
    return fig

# =========================================================
# --- Streamlit UI
# =========================================================
st.set_page_config(page_title='Trend Signal PRO', layout='wide')
st.markdown("""<style>
.stApp{background:#0d1117;color:#e6edf3;}
[data-testid="stMetricValue"]{color:#e6edf3!important;font-size:1.2rem!important;}
.bb{background:#1a3d24;color:#3fb950;border:1px solid #3fb950;padding:5px 15px;border-radius:5px;font-weight:bold;}
.bs{background:#3d1a1a;color:#f85149;border:1px solid #f85149;padding:5px 15px;border-radius:5px;font-weight:bold;}
</style>""", unsafe_allow_html=True)

for k,v in [('active_params',DEFAULT_PARAMS),('wf_result',None),
            ('result',None),('current_code',''),('_quick_ticker','AAPL')]:
    if k not in st.session_state: st.session_state[k] = v

with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("**Quick Select**")
    cols = st.columns(2)
    for i,(ticker,label) in enumerate(POPULAR):
        if cols[i%2].button(label, key=f'pop_{ticker}', use_container_width=True):
            st.session_state['_quick_ticker'] = ticker
    st.divider()

    code     = st.text_input("Ticker", st.session_state['_quick_ticker']).upper().strip()
    period   = st.selectbox("Period",   PERIODS,   index=2)
    interval = st.selectbox("Interval", INTERVALS, index=0)

    if st.button("🔍 Analyze", type="primary", use_container_width=True):
        if code != st.session_state['current_code']:
            st.session_state['wf_result'] = None
        st.session_state['result']       = None
        st.session_state['current_code'] = code
        with st.spinner(f"Fetching {code} ..."):
            raw = fetch_raw(code, period, interval)
        if raw is not None:
            df_ind = compute_indicators(raw.copy())
            df_sig = compute_signals(df_ind, st.session_state['active_params'])
            st.session_state['result'] = {
                'df':df_sig,'code':code,'period':period,'interval':interval,
                'at':datetime.now().strftime('%H:%M:%S'),
            }
        else:
            st.error(f"❌ データ取得失敗: {code}  期間を短くするか、ティッカーを確認してください。")

res = st.session_state.get('result')

if res is None:
    st.info("👈 左のサイドバーからティッカーを入力して **Analyze** を押してください。")
else:
    df = res['df']; disp_code = res['code']
    st.markdown(f"## {disp_code}  –  Updated: {res['at']}")
    tab1, tab2, tab3 = st.tabs(['📈 Live Chart','🧪 Backtest','🔬 Walk-Forward Optimization'])

    with tab1:
        last = df.iloc[-1]
        sig_label = {1:'🟢 BUY',-1:'🔴 SELL',0:'⚪ NEUTRAL'}.get(last['sig'],'⚪ NEUTRAL')
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Signal", sig_label)
        c2.metric("Trend",  last['trend'])
        c3.metric("RSI",    f"{last['RSI']:.1f}")
        c4.metric("ADX",    f"{last['ADX']:.1f}")
        fig_c = make_chart(df, f"{disp_code} – Signal Analysis")
        st.pyplot(fig_c, use_container_width=True); plt.close(fig_c)

    with tab2:
        p_bt = {**DEFAULT_PARAMS}
        atr_m = p_bt.pop('atr_stop_mult', 2.0)
        trail = p_bt.pop('trail_stop', True)
        bt = run_backtest(df, atr_mult=atr_m, use_trail=trail)
        if bt:
            s = bt['stats']
            c1,c2,c3,c4,c5 = st.columns(5)
            c1.metric("Strategy Return", f"{s['sr']:.2f}%")
            c2.metric("Buy & Hold",      f"{s['bh']:.2f}%")
            c3.metric("Max Drawdown",    f"{s['mdd']:.2f}%")
            c4.metric("Sharpe",          f"{s['sharpe']:.2f}")
            c5.metric("Trades",           s['n'])
            c6,c7,c8,c9 = st.columns(4)
            c6.metric("Win Rate",  f"{s['wr']:.1f}%")
            c7.metric("Avg Win",   f"{s['aw']:.2f}%")
            c8.metric("Avg Loss",  f"{s['al']:.2f}%")
            c9.metric("Calmar",    f"{s['calmar']:.2f}")
            fig_bt = make_bt_chart(bt, f"{disp_code} – Backtest Result", label='Strategy')
            st.pyplot(fig_bt, use_container_width=True); plt.close(fig_bt)
            if bt['trades']:
                st.markdown("#### Trade Log")
                tdf = pd.DataFrame(bt['trades'])
                tdf['entry_date'] = pd.to_datetime(tdf['entry_date']).dt.strftime('%Y-%m-%d')
                tdf['exit_date']  = pd.to_datetime(tdf['exit_date']).dt.strftime('%Y-%m-%d')
                for col in ['ret','entry','exit']: tdf[col] = tdf[col].round(2)
                st.dataframe(tdf, use_container_width=True, height=250)

    with tab3:
        st.markdown("""
        **Expanding Window Walk-Forward Optimization**
        - 各Foldでtrainを先頭から累積（Expanding）して最適化 → 後半Foldほど安定
        - スコア関数: **Calmar比率**（CAGR÷最大DD）で全期間安定リターンを評価
        - ATRトレーリングストップ倍率もグリッドサーチ対象
        """)
        n_splits = st.slider("Splits (Folds)", 2, 6, 4)
        if st.button("🚀 Run Walk-Forward Optimization"):
            st.session_state['wf_result'] = None
            wf = walk_forward_optimize(disp_code, res['period'], res['interval'], n_splits=n_splits)
            st.session_state['wf_result'] = wf

        wf = st.session_state.get('wf_result')
        if wf:
            st.success("✅ Walk-Forward 完了（Expanding Window）")
            s = wf['full_bt']['stats']
            c1,c2,c3,c4,c5 = st.columns(5)
            c1.metric("WF Return",    f"{s['sr']:.1f}%")
            c2.metric("Buy & Hold",   f"{s['bh']:.1f}%")
            c3.metric("Max Drawdown", f"{s['mdd']:.1f}%")
            c4.metric("Sharpe",       f"{s['sharpe']:.2f}")
            c5.metric("Trades",        s['n'])

            fig_wf = make_bt_chart(wf['full_bt'],
                                   f"{disp_code} – WF Out-of-Sample Performance",
                                   bt2=wf['default_bt'], label='WF Strategy')
            st.pyplot(fig_wf, use_container_width=True); plt.close(fig_wf)

            st.markdown("#### 最適パラメータ（最終Fold）")
            st.json(wf['best_params'])

            st.markdown("#### Fold Details")
            for r in wf['fold_results']:
                with st.expander(f"Fold {r['fold']}: {r['test_start']} → {r['test_end']}  |  Train: {r['train_size']}本  |  Score: {r['best_score']}"):
                    st.write("**Best Params:**", r['best_params'])
                    if r['test_bt']:
                        rs = r['test_bt']['stats']
                        fc1,fc2,fc3,fc4 = st.columns(4)
                        fc1.metric("Test Return",  f"{rs['sr']:.2f}%")
                        fc2.metric("Max Drawdown", f"{rs['mdd']:.2f}%")
                        fc3.metric("Calmar",       f"{rs['calmar']:.2f}")
                        fc4.metric("Trades",        rs['n'])
