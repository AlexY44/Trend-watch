"""
Trend Signal PRO  v5.0
主な改善:
  - WF最適化: 組み合わせ数を531,441→729に削減（重みは固定、閾値のみ最適化）
  - 戦略: トレンドフォロー専用に再設計（上昇相場でのみエントリー）
  - バグ修正: DEFAULT_SIGをpopで破壊するバグを根絶
  - ATRトレーリングストップで利益保護
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

# =========================================================
# 定数
# =========================================================
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

# ---- シグナルパラメータ（固定デフォルト）----
DEFAULT_SIG = {
    'adx_th':         20,
    'rsi_buy_th':     45,    # RSIがこの値を上回ったら買い条件成立
    'rsi_sell_th':    55,    # RSIがこの値を下回ったら売り条件成立
    'stoch_buy_th':   30,
    'stoch_sell_th':  70,
    'buy_th':          3,    # 買いスコアの閾値
    'sell_th':         3,    # 売りスコアの閾値
}

# ---- 実行パラメータ（辞書とは分離して定数化）----
ATR_MULT_DEFAULT  = 2.0
USE_TRAIL_DEFAULT = True

# ---- WF最適化グリッド: 重みは固定、閾値だけ最適化 → 729通り ----
PARAM_GRID = {
    'adx_th':        [15, 20, 25],
    'rsi_buy_th':    [35, 40, 45],
    'rsi_sell_th':   [55, 60, 65],
    'buy_th':        [3, 4, 5],
    'sell_th':       [3, 4, 5],
    'atr_mult':      [1.5, 2.0, 2.5],
}

# =========================================================
# インジケーター計算
# =========================================================
def flatten_df(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df

def compute_indicators(df):
    cl = df['Close']; hi = df['High']; lo = df['Low']; vo = df['Volume']

    # ---- トレンド ----
    df['EMA20']  = cl.ewm(span=20, adjust=False).mean()
    df['EMA50']  = cl.ewm(span=50, adjust=False).mean()
    df['SMA200'] = cl.rolling(200).mean()
    df['SMA25']  = cl.rolling(25).mean()
    df['SMA75']  = cl.rolling(75).mean()

    # EMA傾き: 5日前比の変化率(%)
    df['EMA20_slope'] = (df['EMA20'] / df['EMA20'].shift(5) - 1) * 100
    df['EMA50_slope'] = (df['EMA50'] / df['EMA50'].shift(10) - 1) * 100

    # ---- Bollinger Bands ----
    bb = ta.volatility.BollingerBands(cl, 20, 2)
    df['BB_u']   = bb.bollinger_hband()
    df['BB_m']   = bb.bollinger_mavg()
    df['BB_l']   = bb.bollinger_lband()
    df['BB_pct'] = (cl - df['BB_l']) / (df['BB_u'] - df['BB_l'] + 1e-9)

    # ---- MACD ----
    mc = ta.trend.MACD(cl, 26, 12, 9)
    df['MACD'] = mc.macd()
    df['MSIG'] = mc.macd_signal()
    df['MHST'] = mc.macd_diff()

    # ---- RSI（EMAスムーズ）----
    df['RSI']   = ta.momentum.RSIIndicator(cl, 14).rsi()
    df['RSI_s'] = df['RSI'].ewm(span=5, adjust=False).mean()

    # ---- Stochastic ----
    sto = ta.momentum.StochasticOscillator(hi, lo, cl, 14, 3)
    df['SK'] = sto.stoch()
    df['SD'] = sto.stoch_signal()

    # ---- ADX / DI ----
    adx_i = ta.trend.ADXIndicator(hi, lo, cl, 14)
    df['ADX']      = adx_i.adx()
    df['DI_plus']  = adx_i.adx_pos()
    df['DI_minus'] = adx_i.adx_neg()

    # ---- ATR ----
    df['ATR']     = ta.volatility.AverageTrueRange(hi, lo, cl, 14).average_true_range()
    df['ATR_pct'] = df['ATR'] / cl.replace(0, np.nan)

    # ---- 出来高 ----
    df['VMA']     = vo.rolling(20).mean()
    df['V_ratio'] = vo / df['VMA'].replace(0, np.nan)

    return df

def compute_signals(df, p=None):
    """
    戦略設計:
    ─────────────────────────────────────────────
    【買い】  トレンドフォロー + 押し目エントリー
      必須: EMA20 > EMA50 (上昇トレンド)
            EMA20が上向き (slope > 0)
            ADX > adx_th (トレンド強度あり)
      補助スコア:
        +2 MACDがゼロライン上でSig上抜け  (強いモメンタム)
        +1 MACDがゼロライン下でSig上抜け  (底打ち反転)
        +1 RSI(smooth) > rsi_buy_th       (モメンタム確認)
        +1 Stoch %K が buy_th 以下でSig上抜け (押し目)
        +1 BB_pct < 0.4 (BBの下半分: 押し目)
      → buy_th 以上でシグナル発生

    【売り】  トレンド崩壊 or モメンタム喪失
      必須条件なし（どちらかで売り）
      スコア:
        +2 EMA20 < EMA50 かつ 下向き (トレンド崩壊)
        +2 MACDがSig下抜け
        +1 RSI(smooth) < rsi_sell_th
        +1 Stoch %K が sell_th 以上でSig下抜け
        +1 BB_pct > 0.7 (高値圏)
      → sell_th 以上でシグナル発生
    ─────────────────────────────────────────────
    """
    if p is None: p = {}
    # DEFAULT_SIGをベースに上書き（元の辞書は変更しない）
    cfg = {**DEFAULT_SIG}
    for k in DEFAULT_SIG:
        if k in p:
            cfg[k] = p[k]

    s = df

    # ---- トレンド判定 ----
    ema_up   = (s['EMA20'] > s['EMA50']) & (s['EMA20_slope'] > 0)
    ema_down = (s['EMA20'] < s['EMA50']) & (s['EMA20_slope'] < 0)
    adx_ok   = s['ADX'] > cfg['adx_th']
    di_up    = s['DI_plus'] > s['DI_minus']
    di_down  = s['DI_plus'] < s['DI_minus']

    # ---- MACDクロス ----
    macd_xu = (s['MACD'] > s['MSIG']) & (s['MACD'].shift(1) <= s['MSIG'].shift(1))
    macd_xd = (s['MACD'] < s['MSIG']) & (s['MACD'].shift(1) >= s['MSIG'].shift(1))

    # ---- Stochasticクロス ----
    sk_xu = (s['SK'] > s['SD']) & (s['SK'].shift(1) <= s['SD'].shift(1))
    sk_xd = (s['SK'] < s['SD']) & (s['SK'].shift(1) >= s['SD'].shift(1))

    # ========================================================
    # 買いスコア
    # ========================================================
    bsc = pd.Series(0, index=s.index)

    # MACDクロス上 (ゼロライン上なら+2、下でも+1)
    bsc += (macd_xu & (s['MACD'] > 0)).astype(int) * 2
    bsc += (macd_xu & (s['MACD'] <= 0)).astype(int) * 1

    # RSIモメンタム確認
    bsc += (s['RSI_s'] > cfg['rsi_buy_th']).astype(int)

    # Stoch押し目クロス
    bsc += (sk_xu & (s['SK'] < cfg['stoch_buy_th'])).astype(int)

    # BB押し目（バンドの下半分）
    bsc += (s['BB_pct'] < 0.4).astype(int)

    # トレンドフィルター: 上昇トレンド + ADX + +DI優位でない場合スコアを半減
    trend_ok = ema_up & adx_ok & di_up
    bsc = bsc.where(trend_ok, bsc // 2)

    # ========================================================
    # 売りスコア
    # ========================================================
    ssc = pd.Series(0, index=s.index)

    # トレンド崩壊（これだけで大きな加点）
    ssc += (ema_down & adx_ok & di_down).astype(int) * 2

    # MACDクロス下
    ssc += (macd_xd & (s['MACD'] < 0)).astype(int) * 2
    ssc += (macd_xd & (s['MACD'] >= 0)).astype(int) * 1

    # RSI過熱
    ssc += (s['RSI_s'] < cfg['rsi_sell_th']).astype(int)

    # Stoch高値クロス
    ssc += (sk_xd & (s['SK'] > cfg['stoch_sell_th'])).astype(int)

    # BB過熱圏
    ssc += (s['BB_pct'] > 0.7).astype(int)

    df = df.copy()
    df['bsc'] = bsc
    df['ssc'] = ssc
    df['sig'] = 0
    df.loc[bsc >= cfg['buy_th'],  'sig'] =  1
    df.loc[ssc >= cfg['sell_th'], 'sig'] = -1

    # トレンド表示用
    df['trend'] = 'Range'
    df.loc[ema_up   & adx_ok, 'trend'] = 'Up'
    df.loc[ema_down & adx_ok, 'trend'] = 'Down'

    return df

# =========================================================
# データ取得
# =========================================================
@st.cache_data(ttl=60, show_spinner=False)
def fetch_raw(code, period, interval):
    try:
        df = yf.download(code, period=period, interval=interval,
                         auto_adjust=True, progress=False)
        if df is None or df.empty: return None
        df = flatten_df(df)
        df = df.dropna(subset=['Close','Open','High','Low','Volume'])
        return df if len(df) >= 150 else None
    except:
        return None

# =========================================================
# バックテスト
# =========================================================
def run_backtest(df, cost=0.001, initial_equity=1.0,
                 atr_mult=ATR_MULT_DEFAULT, use_trail=USE_TRAIL_DEFAULT):
    cl    = df['Close'].values
    hi    = df['High'].values
    sig   = df['sig'].values
    atr   = df['ATR'].values
    dates = df.index

    trades = []; eq = initial_equity; equity = [eq]
    pos = 0; ep = 0.0; ed = None
    peak_px = 0.0; tstop = 0.0

    for i in range(1, len(df)):
        px = cl[i]; ps = sig[i - 1]

        if pos == 1:
            if use_trail:
                peak_px = max(peak_px, hi[i])
                tstop   = peak_px - atr[i] * atr_mult
            stop_hit = use_trail and (px < tstop)

            if ps == -1 or stop_hit:
                xp  = px * (1 - cost)
                ret = (xp - ep) / ep
                eq *= (1 + ret)
                trades.append({
                    'entry_date': ed, 'exit_date': dates[i],
                    'entry': ep, 'exit': xp, 'ret': ret * 100,
                    'result':    'Win' if ret > 0 else 'Loss',
                    'exit_type': 'Stop' if stop_hit else 'Signal',
                })
                pos = 0; peak_px = 0.0; tstop = 0.0

        elif pos == 0 and ps == 1:
            pos = 1; ep = px * (1 + cost); ed = dates[i]
            peak_px = px; tstop = px - atr[i] * atr_mult

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
    yrs  = max((dates[-1] - dates[0]).days / 365.25, 0.01)
    cagr = ((eq / initial_equity) ** (1 / yrs) - 1) * 100
    dr   = eq_s.pct_change().dropna()
    sharpe = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    calmar = cagr / abs(mdd) if mdd < -0.01 else 0.0

    return {
        'trades': trades, 'equity': eq_s,
        'bh_series': bh_series, 'drawdown': dd,
        'stats': {
            'n':n,'wr':wr,'aw':aw,'al':al,'pf':pf,
            'sr':(eq/initial_equity-1)*100,'bh':bh_pct,
            'mdd':mdd,'cagr':cagr,'sharpe':sharpe,'calmar':calmar,
        },
    }

# =========================================================
# スコア関数（辞書破壊なし）
# =========================================================
def score_params(df, p, cost):
    """
    p は PARAM_GRID の 1コンボ（atr_mult含む）
    辞書を一切変更しない
    """
    atr_mult = float(p.get('atr_mult', ATR_MULT_DEFAULT))
    sig_p    = {k: p[k] for k in DEFAULT_SIG if k in p}

    df2 = compute_signals(df, sig_p)
    bt  = run_backtest(df2, cost, atr_mult=atr_mult, use_trail=True)
    if bt is None: return -9999

    s = bt['stats']
    if s['n'] < 5:     return -9999
    if s['mdd'] < -55: return -9999

    wr_pen = max(0.0, (40 - s['wr']) * 0.05)
    dd_pen = max(0.0, (-s['mdd'] - 30) * 0.2)
    return s['calmar'] * 0.6 + s['sharpe'] * 0.4 - wr_pen - dd_pen

# =========================================================
# Walk-Forward 最適化（Expanding Window）
# 729通り × n_splits fold: 約15〜60秒で完了
# =========================================================
def walk_forward_optimize(code, period, interval, n_splits=4, cost=0.001):
    raw = fetch_raw(code, period, interval)
    if raw is None: return None
    base = compute_indicators(raw.copy())

    keys       = list(PARAM_GRID.keys())
    vals       = list(PARAM_GRID.values())
    all_params = [dict(zip(keys, c)) for c in itertools.product(*vals)]
    # 確認: 729通りのはず
    n_combos   = len(all_params)

    n         = len(base)
    min_train = max(int(n * 0.45), 80)
    test_size = (n - min_train) // n_splits

    if test_size < 20:
        st.error("データが少なすぎます。期間を長くしてください（5y推奨）。")
        return None

    fold_results           = []
    combined_trades        = []
    combined_equity_series = []
    current_eq             = 1.0

    progress = st.progress(0, f'Walk-Forward Optimization ({n_combos}通り × {n_splits} folds)...')

    for fold in range(n_splits):
        test_start = min_train + fold * test_size
        test_end   = test_start + test_size if fold < n_splits - 1 else n
        train      = base.iloc[:test_start].copy()
        test       = base.iloc[test_start:test_end].copy()

        if len(train) < 80 or len(test) < 20: continue

        # train で最適化
        best_score = -9999
        best_p     = all_params[len(all_params)//2]
        for p in all_params:
            sc = score_params(train, p, cost)
            if sc > best_score:
                best_score = sc
                best_p     = p

        # test に適用（辞書コピーを使い元を変更しない）
        atr_m  = float(best_p.get('atr_mult', ATR_MULT_DEFAULT))
        sig_p  = {k: best_p[k] for k in DEFAULT_SIG if k in best_p}
        tdf    = compute_signals(test.copy(), sig_p)
        tbt    = run_backtest(tdf, cost, initial_equity=current_eq,
                              atr_mult=atr_m, use_trail=True)

        if tbt:
            combined_trades.extend(tbt['trades'])
            combined_equity_series.append(tbt['equity'])
            current_eq = tbt['equity'].iloc[-1]

        fold_results.append({
            'fold':       fold + 1,
            'train_n':    len(train),
            'test_start': base.index[test_start].strftime('%Y/%m'),
            'test_end':   base.index[min(test_end-1, n-1)].strftime('%Y/%m'),
            'best_p':     best_p,
            'score':      round(best_score, 3),
            'tbt':        tbt,
        })
        progress.progress((fold+1)/n_splits,
                          f'Fold {fold+1}/{n_splits} 完了 (score={best_score:.2f})')

    progress.empty()
    if not combined_equity_series: return None

    full_eq = pd.concat(combined_equity_series)
    full_eq = full_eq[~full_eq.index.duplicated(keep='first')]

    bh_wf = pd.Series(
        (base.loc[full_eq.index,'Close'] /
         base.loc[full_eq.index[0],'Close']).values,
        index=full_eq.index,
    )
    rm      = full_eq.cummax()
    full_dd = (full_eq - rm) / rm * 100
    pc      = full_eq.pct_change().dropna()

    full_stats = {
        'n':  len(combined_trades),
        'sr': (full_eq.iloc[-1] - 1.0) * 100,
        'bh': (bh_wf.iloc[-1] - 1.0) * 100,
        'mdd': full_dd.min(),
        'sharpe': pc.mean()/pc.std()*np.sqrt(252) if pc.std()>0 else 0,
    }

    # Default比較
    def_df = compute_signals(base.copy(), DEFAULT_SIG)
    def_bt = run_backtest(def_df, cost,
                          atr_mult=ATR_MULT_DEFAULT, use_trail=USE_TRAIL_DEFAULT)

    return {
        'best_p':       fold_results[-1]['best_p'],
        'full_bt': {
            'trades':    combined_trades,
            'equity':    full_eq,
            'bh_series': bh_wf,
            'drawdown':  full_dd,
            'stats':     full_stats,
        },
        'default_bt':   def_bt,
        'fold_results': fold_results,
        'base':         base,
        'n_combos':     n_combos,
    }

# =========================================================
# 描画
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
        ax.bar(i, max(b1-b0,1e-6), bottom=b0, width=0.6,
               color=col, linewidth=0, zorder=3)

def make_chart(df, title, mobile=False):
    w,h = (9,14) if mobile else (16,13)
    fig = plt.figure(figsize=(w,h), facecolor=C['bg'])
    gs  = gridspec.GridSpec(5,1, figure=fig,
                            height_ratios=[4,1,1.3,1.3,1.3], hspace=0.05)
    axes = [fig.add_subplot(gs[i]) for i in range(5)]
    for ax in axes: _style_ax(ax)

    display_n = 200
    plot_df = df.iloc[-display_n:].copy().reset_index(drop=False)
    n = len(plot_df); xs = np.arange(n)

    ax0 = axes[0]
    draw_candles(ax0, plot_df)
    ax0.plot(xs, plot_df['EMA20'],  color='#ff79c6',   lw=1.0, label='EMA20', zorder=4)
    ax0.plot(xs, plot_df['EMA50'],  color=C['sma25'],   lw=1.2, label='EMA50', zorder=4)
    ax0.plot(xs, plot_df['SMA200'], color=C['sma200'],  lw=1.2, label='SMA200',zorder=4)
    ax0.plot(xs, plot_df['BB_u'],   color=C['bb'], lw=0.8, ls=':', alpha=0.8)
    ax0.plot(xs, plot_df['BB_l'],   color=C['bb'], lw=0.8, ls=':', alpha=0.8)
    ax0.fill_between(xs, plot_df['BB_u'], plot_df['BB_l'], color=C['bb'], alpha=0.07)
    bi = plot_df.index[plot_df['sig']== 1].tolist()
    si = plot_df.index[plot_df['sig']==-1].tolist()
    if bi: ax0.scatter(bi, plot_df.loc[bi,'Low']*0.995,
                       marker='^', color=C['buy'],  s=65, zorder=6, label='Buy')
    if si: ax0.scatter(si, plot_df.loc[si,'High']*1.005,
                       marker='v', color=C['sell'], s=65, zorder=6, label='Sell')
    ax0.set_title(title, color=C['text'], fontsize=11, pad=6)
    ax0.legend(loc='upper left', fontsize=7, facecolor=C['bg'],
               edgecolor=C['grid'], labelcolor=C['text'], ncol=5)
    ax0.set_xlim(-1,n); ax0.set_xticks([])
    ax0.set_ylabel('Price', color=C['sub'], fontsize=8)

    ax1 = axes[1]
    vcols = [C['cup'] if plot_df['Close'].iloc[i]>=plot_df['Open'].iloc[i]
             else C['cdn'] for i in range(n)]
    ax1.bar(xs, plot_df['Volume'], color=vcols, width=0.7, alpha=0.8, zorder=3)
    ax1.plot(xs, plot_df['VMA'], color=C['neutral'], lw=1.0, zorder=4)
    ax1.set_xlim(-1,n); ax1.set_xticks([])
    ax1.set_ylabel('Vol', color=C['sub'], fontsize=8)

    ax2 = axes[2]
    ax2.plot(xs, plot_df['MACD'], color=C['macd'], lw=1.2, label='MACD', zorder=3)
    ax2.plot(xs, plot_df['MSIG'], color=C['msig'], lw=1.0, ls='--', label='Sig', zorder=3)
    hist  = plot_df['MHST'].values
    hcols = [C['hup'] if v>=0 else C['hdn'] for v in hist]
    ax2.bar(xs, hist, color=hcols, width=0.7, alpha=0.75, zorder=2)
    ax2.axhline(0, color=C['grid'], lw=0.8)
    ax2.set_xlim(-1,n); ax2.set_xticks([])
    ax2.set_ylabel('MACD', color=C['sub'], fontsize=8)
    ax2.legend(loc='upper left', fontsize=7, facecolor=C['bg'],
               edgecolor=C['grid'], labelcolor=C['text'])

    ax3 = axes[3]
    ax3.plot(xs, plot_df['RSI'],   color=C['rsi'],  lw=1.2, label='RSI',   zorder=3)
    ax3.plot(xs, plot_df['RSI_s'], color='#79c0ff', lw=0.9, ls='--', label='RSI(s)', zorder=3)
    ax3.axhline(60, color=C['sell'], lw=0.8, ls='--', alpha=0.7)
    ax3.axhline(40, color=C['buy'],  lw=0.8, ls='--', alpha=0.7)
    ax3.axhline(50, color=C['grid'], lw=0.6)
    ax3.fill_between(xs, plot_df['RSI'], 60,
                     where=plot_df['RSI']>=60, color=C['sell'], alpha=0.15)
    ax3.fill_between(xs, plot_df['RSI'], 40,
                     where=plot_df['RSI']<=40, color=C['buy'],  alpha=0.15)
    ax3.set_ylim(0,100); ax3.set_xlim(-1,n); ax3.set_xticks([])
    ax3.set_ylabel('RSI', color=C['sub'], fontsize=8)
    ax3.legend(loc='upper left', fontsize=7, facecolor=C['bg'],
               edgecolor=C['grid'], labelcolor=C['text'])

    ax4 = axes[4]
    ax4.plot(xs, plot_df['SK'], color=C['buy'],  lw=1.1, label='%K', zorder=3)
    ax4.plot(xs, plot_df['SD'], color=C['msig'], lw=1.0, ls='--', label='%D', zorder=3)
    ax4.axhline(70, color=C['sell'], lw=0.8, ls='--', alpha=0.7)
    ax4.axhline(30, color=C['buy'],  lw=0.8, ls='--', alpha=0.7)
    ax4.set_ylim(0,100); ax4.set_xlim(-1,n)
    ax4.set_ylabel('Stoch', color=C['sub'], fontsize=8)
    ax4.legend(loc='upper left', fontsize=7, facecolor=C['bg'],
               edgecolor=C['grid'], labelcolor=C['text'])

    step = max(1, n//10)
    tpos = list(range(0, n, step))
    dcol = plot_df.columns[0]
    try:    labs = [str(plot_df.iloc[i][dcol])[:10] for i in tpos]
    except: labs = [str(i) for i in tpos]
    ax4.set_xticks(tpos)
    ax4.set_xticklabels(labs, rotation=30, ha='right', fontsize=6, color=C['sub'])
    plt.tight_layout(pad=0.5)
    return fig

def make_bt_chart(bt, title, bt2=None, label='Strategy'):
    eq = bt['equity']; dd = bt['drawdown']; bh = bt.get('bh_series')
    fig, axes = plt.subplots(2,1, figsize=(14,8), facecolor=C['bg'],
                             gridspec_kw={'height_ratios':[3,1],'hspace':0.06})
    for ax in axes: _style_ax(ax)
    ax1, ax2 = axes

    if bh is not None:
        ax1.plot(bh.index, bh.values,
                 color=C['bh'], lw=1.4, ls=':', label='Buy & Hold', zorder=2)
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

    ax2.fill_between(dd.index, dd.values, 0,
                     where=dd.values<0, color=C['sell'], alpha=0.45)
    ax2.axhline(0, color=C['grid'], lw=0.8)
    ax2.set_ylabel('Drawdown%', color=C['sub'], fontsize=9)
    ax2.tick_params(axis='x', colors=C['sub'], labelsize=7)
    plt.tight_layout(pad=0.5)
    return fig

# =========================================================
# Streamlit UI
# =========================================================
st.set_page_config(page_title='Trend Signal PRO', layout='wide')
st.markdown("""<style>
.stApp{background:#0d1117;color:#e6edf3;}
[data-testid="stMetricValue"]{color:#e6edf3!important;font-size:1.2rem!important;}
</style>""", unsafe_allow_html=True)

for k,v in [('result',None),('wf_result',None),
            ('current_code',''),('_quick_ticker','AAPL')]:
    if k not in st.session_state: st.session_state[k] = v

with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("**Quick Select**")
    cols = st.columns(2)
    for i,(ticker,lbl) in enumerate(POPULAR):
        if cols[i%2].button(lbl, key=f'pop_{ticker}', use_container_width=True):
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
        with st.spinner(f"Fetching {code}..."):
            raw = fetch_raw(code, period, interval)
        if raw is not None:
            df_i = compute_indicators(raw.copy())
            df_s = compute_signals(df_i)
            st.session_state['result'] = {
                'df':df_s, 'code':code, 'period':period,
                'interval':interval, 'at':datetime.now().strftime('%H:%M:%S'),
            }
        else:
            st.error(f"❌ データ取得失敗: {code}  ティッカーか期間を確認してください。")

res = st.session_state.get('result')
if res is None:
    st.info("👈 サイドバーでティッカーを選択して Analyze を押してください。")
    st.stop()

df = res['df']; disp_code = res['code']
st.markdown(f"## {disp_code}  –  Updated: {res['at']}")

tab1, tab2, tab3 = st.tabs(['📈 Live Chart','🧪 Backtest','🔬 Walk-Forward Optimization'])

with tab1:
    last = df.iloc[-1]
    sig_label = {1:'🟢 BUY',-1:'🔴 SELL',0:'⚪ NEUTRAL'}.get(int(last['sig']),'⚪ NEUTRAL')
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Signal", sig_label)
    c2.metric("Trend",  last['trend'])
    c3.metric("RSI",    f"{last['RSI']:.1f}")
    c4.metric("ADX",    f"{last['ADX']:.1f}")
    fig_c = make_chart(df, f"{disp_code} – Signal Analysis")
    st.pyplot(fig_c, use_container_width=True); plt.close(fig_c)

with tab2:
    bt = run_backtest(df, atr_mult=ATR_MULT_DEFAULT, use_trail=USE_TRAIL_DEFAULT)
    if bt:
        s = bt['stats']
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Strategy Return", f"{s['sr']:.1f}%")
        c2.metric("Buy & Hold",      f"{s['bh']:.1f}%")
        c3.metric("Max Drawdown",    f"{s['mdd']:.1f}%")
        c4.metric("Sharpe",          f"{s['sharpe']:.2f}")
        c5.metric("Trades",           s['n'])
        c6,c7,c8,c9 = st.columns(4)
        c6.metric("Win Rate", f"{s['wr']:.1f}%")
        c7.metric("Avg Win",  f"{s['aw']:.2f}%")
        c8.metric("Avg Loss", f"{s['al']:.2f}%")
        c9.metric("Calmar",   f"{s['calmar']:.2f}")
        fig_bt = make_bt_chart(bt, f"{disp_code} – Backtest", label='Strategy')
        st.pyplot(fig_bt, use_container_width=True); plt.close(fig_bt)
        if bt['trades']:
            st.markdown("#### Trade Log")
            tdf = pd.DataFrame(bt['trades'])
            tdf['entry_date'] = pd.to_datetime(tdf['entry_date']).dt.strftime('%Y-%m-%d')
            tdf['exit_date']  = pd.to_datetime(tdf['exit_date']).dt.strftime('%Y-%m-%d')
            for col in ['ret','entry','exit']: tdf[col] = tdf[col].round(2)
            st.dataframe(tdf, use_container_width=True, height=250)

with tab3:
    n_combos_display = 1
    for v in PARAM_GRID.values(): n_combos_display *= len(v)
    st.markdown(f"""
    **Expanding Window Walk-Forward Optimization**
    - グリッド: **{n_combos_display}通り**（重みは固定、閾値のみ最適化）
    - Train: Expanding（累積拡大）→ 後半Foldほど安定した選択
    - スコア: Calmar×0.6 + Sharpe×0.4
    - 推定時間: **{n_combos_display * 4 // 200}〜{n_combos_display * 4 // 80}秒**（銘柄・期間による）
    """)
    n_splits = st.slider("Splits", 2, 6, 4)
    if st.button("🚀 Run Walk-Forward Optimization"):
        st.session_state['wf_result'] = None
        wf = walk_forward_optimize(disp_code, res['period'], res['interval'],
                                   n_splits=n_splits)
        st.session_state['wf_result'] = wf

    wf = st.session_state.get('wf_result')
    if wf:
        st.success(f"✅ Walk-Forward 完了（{wf['n_combos']}通り × {n_splits} folds）")
        s = wf['full_bt']['stats']
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("WF Return",    f"{s['sr']:.1f}%")
        c2.metric("Buy & Hold",   f"{s['bh']:.1f}%")
        c3.metric("Max Drawdown", f"{s['mdd']:.1f}%")
        c4.metric("Sharpe",       f"{s['sharpe']:.2f}")
        c5.metric("Trades",        s['n'])

        fig_wf = make_bt_chart(wf['full_bt'],
                               f"{disp_code} – WF Out-of-Sample",
                               bt2=wf['default_bt'], label='WF Strategy')
        st.pyplot(fig_wf, use_container_width=True); plt.close(fig_wf)

        st.markdown("#### 最適パラメータ（最終Fold）")
        st.json(wf['best_p'])

        st.markdown("#### Fold Details")
        for r in wf['fold_results']:
            with st.expander(
                f"Fold {r['fold']}: {r['test_start']} → {r['test_end']}"
                f"  |  Train: {r['train_n']}本  |  Score: {r['score']}"
            ):
                st.write("**Best Params:**", r['best_p'])
                if r['tbt']:
                    rs = r['tbt']['stats']
                    fc1,fc2,fc3,fc4 = st.columns(4)
                    fc1.metric("Test Return",  f"{rs['sr']:.1f}%")
                    fc2.metric("Max Drawdown", f"{rs['mdd']:.1f}%")
                    fc3.metric("Calmar",       f"{rs['calmar']:.2f}")
                    fc4.metric("Trades",        rs['n'])
