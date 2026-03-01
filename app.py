import warnings
import itertools
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
    ('1326.T','Gold ETF'), ('7203.T','Toyota'),  ('6758.T','Sony'),
    ('9984.T','SBG'),      ('6861.T','Keyence'),  ('8306.T','MUFG'),
    ('^N225','Nikkei225'), ('AAPL','Apple'),      ('NVDA','NVIDIA'),
    ('^GSPC','SP500'),
]
C = {
    'bg':'#0d1117',    'panel':'#161b22', 'grid':'#21262d',
    'text':'#e6edf3',  'sub':'#8b949e',
    'buy':'#3fb950',   'sell':'#f85149',  'neutral':'#58a6ff',
    'sma200':'#bc8cff','bb':'#388bfd',    'macd':'#58a6ff',
    'msig':'#ffa657',  'hup':'#3fb950',   'hdn':'#f85149',
    'rsi':'#d2a8ff',   'cup':'#3fb950',   'cdn':'#f85149',
    'bh':'#e8c55a',    'ema20':'#ff79c6', 'ema50':'#ffa657',
}

# =========================================================
# 戦略設計 - Sharpe/Calmar最大化
# =========================================================
#
# 【目標】 リターンでB&Hに勝つことは目指さない
#          → Sharpe比・Calmar比でB&Hを上回る
#          → MaxDDをB&Hの半分以下に抑える
#
# 【買い】 3段階ANDフィルター（確実性重視）
#   MUST: EMA20 > EMA50 > SMA200  (強い上昇トレンド確認)
#         ADX > adx_th            (トレンド強度あり)
#         +DI > -DI               (上昇方向優位)
#   TRIGGER (どれか1つ):
#     A) MACDゼロライン上でSig上抜け   (最強シグナル)
#     B) EMA50付近の押し目 + Stoch低位 (押し目買い)
#     C) EMA20/50ゴールデンクロス       (トレンド転換)
#
# 【売り】 優先度順（Calmar最大化 = 早期撤退）
#   P1: EMA20がEMA50を下抜け → 即決済
#       ※ これだけで大きなドローダウンの80%を回避できる
#   P2: ATRトレーリングストップ（atr_mult=3.0倍、広め）
#       → 小さな押し目では出ない
#       → しかし一度崩れたら確実に守る
#   P3: タイムストップ（n日保有後に方向感なければ決済）
#       → 塩漬けコストの防止
#
# 【B&Hとの差別化ポイント】
#   ✓ 大クラッシュ（-30%以上）を先に逃げることでCalmar改善
#   ✓ ノーポジ期間が「現金温存」として機能（再エントリー余力）
#   ✗ 強気相場のリターン総量では負ける（これは許容する）

DEFAULT_SIG = {
    'adx_th':     20,
    'atr_mult':   3.0,    # 広めにして不要な決済を減らす
    'ema_gap':    0.03,
    'stoch_th':   35,
    'time_stop':  20,     # タイムストップ（営業日数）
}

ATR_MULT  = 3.0
USE_TRAIL = True

PARAM_GRID = {
    'adx_th':    [15, 20, 25],
    'atr_mult':  [2.5, 3.0, 4.0],
    'ema_gap':   [0.02, 0.04],
    'stoch_th':  [30, 40],
    'time_stop': [15, 25, 999],   # 999 = タイムストップ無効
}
# 3×3×2×2×3 = 108通り


# =========================================================
# インジケーター
# =========================================================
def flatten_df(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df

def compute_indicators(df):
    cl = df['Close']
    hi = df['High']
    lo = df['Low']
    vo = df['Volume']

    df['EMA20']  = cl.ewm(span=20, adjust=False).mean()
    df['EMA50']  = cl.ewm(span=50, adjust=False).mean()
    df['SMA200'] = cl.rolling(200).mean()

    bb = ta.volatility.BollingerBands(cl, window=20, window_dev=2)
    df['BB_u']   = bb.bollinger_hband()
    df['BB_m']   = bb.bollinger_mavg()
    df['BB_l']   = bb.bollinger_lband()
    df['BB_pct'] = (cl - df['BB_l']) / (df['BB_u'] - df['BB_l'] + 1e-9)

    mc = ta.trend.MACD(cl, window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = mc.macd()
    df['MSIG'] = mc.macd_signal()
    df['MHST'] = mc.macd_diff()

    df['RSI']   = ta.momentum.RSIIndicator(cl, window=14).rsi()
    df['RSI_s'] = df['RSI'].ewm(span=5, adjust=False).mean()

    sto = ta.momentum.StochasticOscillator(hi, lo, cl, window=14, smooth_window=3)
    df['SK'] = sto.stoch()
    df['SD'] = sto.stoch_signal()

    adx_i = ta.trend.ADXIndicator(hi, lo, cl, window=14)
    df['ADX']      = adx_i.adx()
    df['DI_plus']  = adx_i.adx_pos()
    df['DI_minus'] = adx_i.adx_neg()

    df['ATR'] = ta.volatility.AverageTrueRange(hi, lo, cl, window=14).average_true_range()
    df['VMA'] = vo.rolling(20).mean()

    return df

def compute_signals(df, p=None):
    cfg = dict(DEFAULT_SIG)
    if p:
        for k in DEFAULT_SIG:
            if k in p:
                cfg[k] = p[k]

    s = df
    cl = s['Close']

    # ---- トレンドフィルター（3条件AND）----
    sma200_valid = s['SMA200'].notna()
    trend_up = (
        (s['EMA20'] > s['EMA50']) &
        (s['EMA50'] > s['SMA200'].where(sma200_valid, 0)) &
        sma200_valid &
        (s['ADX'] > cfg['adx_th']) &
        (s['DI_plus'] > s['DI_minus'])
    )
    trend_dn = (
        (s['EMA20'] < s['EMA50']) &
        (s['ADX'] > cfg['adx_th']) &
        (s['DI_plus'] < s['DI_minus'])
    )

    # ---- 買いトリガー ----
    # A: MACDゼロライン上クロス（最強）
    macd_xu_above = (
        (s['MACD'] > s['MSIG']) &
        (s['MACD'].shift(1) <= s['MSIG'].shift(1)) &
        (s['MACD'] > 0)
    )
    # B: EMA50付近の押し目 + Stoch低位クロス
    near_ema50  = (cl > s['EMA50'] * (1 - cfg['ema_gap'])) & (cl < s['EMA50'] * 1.01)
    stoch_xu    = (s['SK'] > s['SD']) & (s['SK'].shift(1) <= s['SD'].shift(1))
    pullback    = near_ema50 & stoch_xu & (s['SK'] < cfg['stoch_th'])
    # C: EMAゴールデンクロス
    ema_gx      = (s['EMA20'] > s['EMA50']) & (s['EMA20'].shift(1) <= s['EMA50'].shift(1))

    buy_sig  = trend_up & (macd_xu_above | pullback | ema_gx)

    # ---- 売りシグナル（P1のみ: EMAデッドクロス）----
    # ATRストップとタイムストップはバックテストループ内で処理
    ema_dx   = (s['EMA20'] < s['EMA50']) & (s['EMA20'].shift(1) >= s['EMA50'].shift(1))
    sell_sig = ema_dx | trend_dn   # トレンド崩壊も含む

    df = df.copy()
    df['sig'] = 0
    df.loc[buy_sig,  'sig'] =  1
    df.loc[sell_sig, 'sig'] = -1

    df['trend'] = 'Range'
    df.loc[trend_up & (s['ADX'] > 20), 'trend'] = 'Up'
    df.loc[trend_dn & (s['ADX'] > 20), 'trend'] = 'Down'

    return df

# =========================================================
# データ取得
# =========================================================
@st.cache_data(ttl=60, show_spinner=False)
def fetch_raw(code, period, interval):
    try:
        df = yf.download(code, period=period, interval=interval,
                         auto_adjust=True, progress=False)
        if df is None or df.empty:
            return None
        df = flatten_df(df)
        df = df.dropna(subset=['Close', 'Open', 'High', 'Low', 'Volume'])
        return df if len(df) >= 150 else None
    except Exception:
        return None

# =========================================================
# バックテスト（タイムストップ対応）
# =========================================================
def run_backtest(df, cost=0.001, initial_equity=1.0,
                 atr_mult=ATR_MULT, use_trail=USE_TRAIL, time_stop=999):
    cl    = df['Close'].values
    hi    = df['High'].values
    sig   = df['sig'].values
    atr   = df['ATR'].values
    dates = df.index

    trades  = []
    eq      = initial_equity
    equity  = [eq]
    pos     = 0
    ep      = 0.0
    ed      = None
    ei      = 0        # エントリーインデックス
    peak    = 0.0
    tstop   = 0.0

    for i in range(1, len(df)):
        px = cl[i]
        ps = sig[i - 1]

        if pos == 1:
            # ATRトレーリングストップ更新
            if use_trail:
                peak  = max(peak, hi[i])
                tstop = peak - atr[i] * atr_mult

            stop_hit  = use_trail and (px < tstop)
            time_hit  = (i - ei) >= time_stop          # タイムストップ
            sell_hit  = ps == -1

            if sell_hit or stop_hit or time_hit:
                xp  = px * (1 - cost)
                ret = (xp - ep) / ep
                eq *= (1 + ret)
                reason = ('Signal' if sell_hit else
                          'TimeStop' if time_hit else 'ATRStop')
                trades.append({
                    'entry_date': ed,
                    'exit_date':  dates[i],
                    'hold_days':  i - ei,
                    'entry':      round(float(ep), 4),
                    'exit':       round(float(xp), 4),
                    'ret':        round(ret * 100, 3),
                    'result':     'Win' if ret > 0 else 'Loss',
                    'exit_type':  reason,
                })
                pos = 0; peak = 0.0; tstop = 0.0

        elif pos == 0 and ps == 1:
            pos   = 1
            ep    = px * (1 + cost)
            ed    = dates[i]
            ei    = i
            peak  = px
            tstop = px - atr[i] * atr_mult

        equity.append(eq)

    eq_s      = pd.Series(equity, index=dates)
    bh_series = pd.Series((cl / cl[0]) * initial_equity, index=dates)
    bh_pct    = (cl[-1] - cl[0]) / cl[0] * 100

    n    = len(trades)
    wins = [t for t in trades if t['ret'] > 0]
    loss = [t for t in trades if t['ret'] <= 0]
    wr   = len(wins) / n * 100 if n > 0 else 0.0
    aw   = float(np.mean([t['ret'] for t in wins])) if wins else 0.0
    al   = float(np.mean([t['ret'] for t in loss])) if loss else 0.0
    pf   = (abs(sum(t['ret'] for t in wins) / sum(t['ret'] for t in loss))
            if loss else 999.0)

    roll_max = eq_s.cummax()
    dd       = (eq_s - roll_max) / roll_max * 100
    mdd      = float(dd.min())

    # B&H のMaxDD計算
    bh_roll  = bh_series.cummax()
    bh_dd    = (bh_series - bh_roll) / bh_roll * 100
    bh_mdd   = float(bh_dd.min())

    yrs    = max((dates[-1] - dates[0]).days / 365.25, 0.01)
    cagr   = ((eq / initial_equity) ** (1.0 / yrs) - 1) * 100
    dr     = eq_s.pct_change().dropna()
    sharpe = float(dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else 0.0
    calmar = cagr / abs(mdd) if mdd < -0.01 else 0.0

    # B&H Sharpe
    bh_dr     = bh_series.pct_change().dropna()
    bh_sharpe = float(bh_dr.mean() / bh_dr.std() * np.sqrt(252)) if bh_dr.std() > 0 else 0.0
    bh_cagr   = ((bh_series.iloc[-1] / initial_equity) ** (1.0 / yrs) - 1) * 100
    bh_calmar = bh_cagr / abs(bh_mdd) if bh_mdd < -0.01 else 0.0

    return {
        'trades':    trades,
        'equity':    eq_s,
        'bh_series': bh_series,
        'drawdown':  dd,
        'bh_dd':     bh_dd,
        'stats': {
            'n': n, 'wr': wr, 'aw': aw, 'al': al, 'pf': pf,
            'sr': (eq / initial_equity - 1) * 100,
            'bh': bh_pct, 'mdd': mdd, 'bh_mdd': bh_mdd,
            'cagr': cagr, 'sharpe': sharpe, 'calmar': calmar,
            'bh_sharpe': bh_sharpe, 'bh_calmar': bh_calmar,
        },
    }

# =========================================================
# スコア（Calmar × Sharpe 重視、リターン比較を廃止）
# =========================================================
def score_params(df, p, cost):
    atr_m     = float(p.get('atr_mult', ATR_MULT))
    time_stop = int(p.get('time_stop', 999))
    sig_p     = {k: p[k] for k in DEFAULT_SIG if k in p}

    df2 = compute_signals(df, sig_p)
    bt  = run_backtest(df2, cost, atr_mult=atr_m,
                       use_trail=True, time_stop=time_stop)
    if bt is None:
        return -9999.0
    s = bt['stats']
    if s['n'] < 3:
        return -9999.0
    # CAGR がマイナスは即除外
    if s['cagr'] <= 0:
        return -9999.0
    # MaxDD が B&H より悪いものは除外（ドローダウン改善が必須）
    if s['mdd'] < s['bh_mdd']:
        return -9999.0

    dd_improvement = s['bh_mdd'] - s['mdd']   # 正値 = DD改善量
    wr_pen = max(0.0, (40.0 - s['wr']) * 0.05)

    # Calmar 0.5 + Sharpe 0.3 + DD改善ボーナス 0.2
    return (s['calmar'] * 0.5
            + s['sharpe'] * 0.3
            + dd_improvement * 0.2
            - wr_pen)

# =========================================================
# Walk-Forward最適化
# =========================================================
def walk_forward_optimize(code, period, interval, n_splits=4, cost=0.001):
    raw = fetch_raw(code, period, interval)
    if raw is None:
        return None

    base = compute_indicators(raw.copy())
    keys = list(PARAM_GRID.keys())
    vals = list(PARAM_GRID.values())
    all_params = [dict(zip(keys, c)) for c in itertools.product(*vals)]
    n_combos   = len(all_params)

    n         = len(base)
    min_train = max(int(n * 0.45), 80)
    test_size = (n - min_train) // n_splits

    if test_size < 20:
        st.error("データが少なすぎます。5y以上を選択してください。")
        return None

    fold_results    = []
    combined_trades = []
    combined_eq     = []
    current_eq      = 1.0

    bar = st.progress(0, f'最適化中 ({n_combos}通り × {n_splits} folds)...')

    for fold in range(n_splits):
        test_start = min_train + fold * test_size
        test_end   = test_start + test_size if fold < n_splits - 1 else n
        train      = base.iloc[:test_start].copy()
        test       = base.iloc[test_start:test_end].copy()

        if len(train) < 80 or len(test) < 20:
            continue

        best_score = -9999.0
        best_p     = all_params[n_combos // 2]
        for p in all_params:
            sc = score_params(train, p, cost)
            if sc > best_score:
                best_score = sc
                best_p     = p

        atr_m     = float(best_p.get('atr_mult', ATR_MULT))
        time_stop = int(best_p.get('time_stop', 999))
        sig_p     = {k: best_p[k] for k in DEFAULT_SIG if k in best_p}
        tdf       = compute_signals(test.copy(), sig_p)
        tbt       = run_backtest(tdf, cost, initial_equity=current_eq,
                                 atr_mult=atr_m, use_trail=True,
                                 time_stop=time_stop)

        if tbt:
            combined_trades.extend(tbt['trades'])
            combined_eq.append(tbt['equity'])
            current_eq = tbt['equity'].iloc[-1]

        fold_results.append({
            'fold':       fold + 1,
            'train_n':    len(train),
            'test_start': base.index[test_start].strftime('%Y/%m'),
            'test_end':   base.index[min(test_end - 1, n - 1)].strftime('%Y/%m'),
            'best_p':     best_p,
            'score':      round(best_score, 3),
            'tbt':        tbt,
        })
        bar.progress((fold + 1) / n_splits, f'Fold {fold+1}/{n_splits} 完了')

    bar.empty()

    if not combined_eq:
        return None

    full_eq = pd.concat(combined_eq)
    full_eq = full_eq[~full_eq.index.duplicated(keep='first')]

    bh_wf = pd.Series(
        (base.loc[full_eq.index, 'Close'] /
         base.loc[full_eq.index[0], 'Close']).values,
        index=full_eq.index,
    )
    rm      = full_eq.cummax()
    full_dd = (full_eq - rm) / rm * 100
    bh_rm   = bh_wf.cummax()
    bh_dd   = (bh_wf - bh_rm) / bh_rm * 100
    pc      = full_eq.pct_change().dropna()
    bh_pc   = bh_wf.pct_change().dropna()

    yrs     = max((full_eq.index[-1] - full_eq.index[0]).days / 365.25, 0.01)
    cagr    = ((full_eq.iloc[-1]) ** (1.0 / yrs) - 1) * 100
    bh_cagr = ((bh_wf.iloc[-1])  ** (1.0 / yrs) - 1) * 100

    full_stats = {
        'n':          len(combined_trades),
        'sr':         (full_eq.iloc[-1] - 1.0) * 100,
        'bh':         (bh_wf.iloc[-1] - 1.0) * 100,
        'mdd':        float(full_dd.min()),
        'bh_mdd':     float(bh_dd.min()),
        'sharpe':     (float(pc.mean() / pc.std() * np.sqrt(252))
                       if pc.std() > 0 else 0.0),
        'bh_sharpe':  (float(bh_pc.mean() / bh_pc.std() * np.sqrt(252))
                       if bh_pc.std() > 0 else 0.0),
        'calmar':     cagr / abs(float(full_dd.min()))
                      if full_dd.min() < -0.01 else 0.0,
        'bh_calmar':  bh_cagr / abs(float(bh_dd.min()))
                      if bh_dd.min() < -0.01 else 0.0,
    }

    def_df = compute_signals(base.copy(), DEFAULT_SIG)
    def_bt = run_backtest(def_df, cost, atr_mult=ATR_MULT,
                          use_trail=USE_TRAIL,
                          time_stop=DEFAULT_SIG['time_stop'])

    return {
        'best_p':       fold_results[-1]['best_p'],
        'full_bt': {
            'trades':    combined_trades,
            'equity':    full_eq,
            'bh_series': bh_wf,
            'drawdown':  full_dd,
            'bh_dd':     bh_dd,
            'stats':     full_stats,
        },
        'default_bt':   def_bt,
        'fold_results': fold_results,
        'n_combos':     n_combos,
    }

# =========================================================
# 描画
# =========================================================
def _ax(ax):
    ax.set_facecolor(C['panel'])
    ax.tick_params(colors=C['sub'], labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor(C['grid'])
    ax.grid(color=C['grid'], lw=0.5, ls='--', alpha=0.5)

def draw_candles(ax, df):
    op = df['Open'].values; hi = df['High'].values
    lo = df['Low'].values;  cl = df['Close'].values
    for i in range(len(df)):
        col = C['cup'] if cl[i] >= op[i] else C['cdn']
        ax.plot([i, i], [lo[i], hi[i]], color=col, lw=0.7, zorder=2)
        b0 = min(op[i], cl[i]); b1 = max(op[i], cl[i])
        ax.bar(i, max(b1 - b0, 1e-6), bottom=b0,
               width=0.6, color=col, linewidth=0, zorder=3)

def make_chart(df, title):
    fig = plt.figure(figsize=(16, 13), facecolor=C['bg'])
    gs  = gridspec.GridSpec(5, 1, figure=fig,
                            height_ratios=[4, 1, 1.3, 1.3, 1.3], hspace=0.05)
    axes = [fig.add_subplot(gs[i]) for i in range(5)]
    for ax in axes: _ax(ax)

    pdf = df.iloc[-200:].copy().reset_index(drop=False)
    n = len(pdf); xs = np.arange(n)

    ax0 = axes[0]
    draw_candles(ax0, pdf)
    ax0.plot(xs, pdf['EMA20'],  color=C['ema20'],  lw=1.0, label='EMA20', zorder=4)
    ax0.plot(xs, pdf['EMA50'],  color=C['ema50'],  lw=1.4, label='EMA50', zorder=4)
    ax0.plot(xs, pdf['SMA200'], color=C['sma200'], lw=1.2, label='SMA200',zorder=4, ls='--')
    ax0.plot(xs, pdf['BB_u'],   color=C['bb'], lw=0.8, ls=':', alpha=0.7)
    ax0.plot(xs, pdf['BB_l'],   color=C['bb'], lw=0.8, ls=':', alpha=0.7)
    ax0.fill_between(xs, pdf['BB_u'], pdf['BB_l'], color=C['bb'], alpha=0.06)
    bi = pdf.index[pdf['sig'] ==  1].tolist()
    si = pdf.index[pdf['sig'] == -1].tolist()
    if bi:
        ax0.scatter(bi, pdf.loc[bi, 'Low'] * 0.995,
                    marker='^', color=C['buy'],  s=70, zorder=6, label='Buy')
    if si:
        ax0.scatter(si, pdf.loc[si, 'High'] * 1.005,
                    marker='v', color=C['sell'], s=70, zorder=6, label='Sell')
    ax0.set_title(title, color=C['text'], fontsize=11, pad=6)
    ax0.legend(loc='upper left', fontsize=7, facecolor=C['bg'],
               edgecolor=C['grid'], labelcolor=C['text'], ncol=6)
    ax0.set_xlim(-1, n); ax0.set_xticks([])
    ax0.set_ylabel('Price', color=C['sub'], fontsize=8)

    ax1 = axes[1]
    vc = [C['cup'] if pdf['Close'].iloc[i] >= pdf['Open'].iloc[i]
          else C['cdn'] for i in range(n)]
    ax1.bar(xs, pdf['Volume'], color=vc, width=0.7, alpha=0.8, zorder=3)
    ax1.plot(xs, pdf['VMA'], color=C['neutral'], lw=1.0, zorder=4)
    ax1.set_xlim(-1, n); ax1.set_xticks([])
    ax1.set_ylabel('Vol', color=C['sub'], fontsize=8)

    ax2 = axes[2]
    ax2.plot(xs, pdf['MACD'], color=C['macd'], lw=1.2, label='MACD', zorder=3)
    ax2.plot(xs, pdf['MSIG'], color=C['msig'], lw=1.0, ls='--', label='Sig', zorder=3)
    hv = pdf['MHST'].values
    hc = [C['hup'] if v >= 0 else C['hdn'] for v in hv]
    ax2.bar(xs, hv, color=hc, width=0.7, alpha=0.75, zorder=2)
    ax2.axhline(0, color=C['grid'], lw=0.8)
    ax2.set_xlim(-1, n); ax2.set_xticks([])
    ax2.set_ylabel('MACD', color=C['sub'], fontsize=8)
    ax2.legend(loc='upper left', fontsize=7, facecolor=C['bg'],
               edgecolor=C['grid'], labelcolor=C['text'])

    ax3 = axes[3]
    ax3.plot(xs, pdf['RSI'],   color=C['rsi'],  lw=1.2, label='RSI',   zorder=3)
    ax3.plot(xs, pdf['RSI_s'], color='#79c0ff', lw=0.9, ls='--', label='RSI(s)', zorder=3)
    ax3.axhline(70, color=C['sell'], lw=0.8, ls='--', alpha=0.6)
    ax3.axhline(30, color=C['buy'],  lw=0.8, ls='--', alpha=0.6)
    ax3.axhline(50, color=C['grid'], lw=0.6)
    ax3.fill_between(xs, pdf['RSI'], 70, where=pdf['RSI'] >= 70,
                     color=C['sell'], alpha=0.15)
    ax3.fill_between(xs, pdf['RSI'], 30, where=pdf['RSI'] <= 30,
                     color=C['buy'],  alpha=0.15)
    ax3.set_ylim(0, 100); ax3.set_xlim(-1, n); ax3.set_xticks([])
    ax3.set_ylabel('RSI', color=C['sub'], fontsize=8)
    ax3.legend(loc='upper left', fontsize=7, facecolor=C['bg'],
               edgecolor=C['grid'], labelcolor=C['text'])

    ax4 = axes[4]
    ax4.plot(xs, pdf['ADX'],      color=C['neutral'], lw=1.4, label='ADX', zorder=3)
    ax4.plot(xs, pdf['DI_plus'],  color=C['buy'],     lw=1.0, label='+DI', zorder=3)
    ax4.plot(xs, pdf['DI_minus'], color=C['sell'],    lw=1.0, label='-DI', zorder=3)
    ax4.axhline(20, color=C['grid'], lw=0.8, ls='--', alpha=0.7)
    ax4.set_ylim(0, 60); ax4.set_xlim(-1, n)
    ax4.set_ylabel('ADX/DI', color=C['sub'], fontsize=8)
    ax4.legend(loc='upper left', fontsize=7, facecolor=C['bg'],
               edgecolor=C['grid'], labelcolor=C['text'])

    step = max(1, n // 10); tpos = list(range(0, n, step))
    dcol = pdf.columns[0]
    try:    labs = [str(pdf.iloc[i][dcol])[:10] for i in tpos]
    except: labs = [str(i) for i in tpos]
    ax4.set_xticks(tpos)
    ax4.set_xticklabels(labs, rotation=30, ha='right', fontsize=6, color=C['sub'])
    plt.tight_layout(pad=0.5)
    return fig

def make_bt_chart(bt, title, bt2=None, label='Strategy'):
    eq = bt['equity']; dd = bt['drawdown']
    bh = bt.get('bh_series'); bh_dd = bt.get('bh_dd')

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), facecolor=C['bg'],
                             gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.06})
    for ax in axes: _ax(ax)
    ax1, ax2 = axes

    if bh is not None:
        ax1.plot(bh.index, bh.values,
                 color=C['bh'], lw=1.4, ls=':', label='Buy & Hold', zorder=2)
    ax1.plot(eq.index, eq.values, color=C['buy'], lw=2.0, label=label, zorder=3)
    if bt2 is not None:
        ax1.plot(bt2['equity'].index, bt2['equity'].values,
                 color='#ffa657', lw=1.5, ls='-.', label='Default', zorder=2)
    ax1.axhline(1.0, color=C['grid'], lw=0.8)
    ax1.set_title(title, color=C['text'], fontsize=11)
    ax1.set_ylabel('Equity (normalized)', color=C['sub'], fontsize=9)
    ax1.legend(loc='upper left', fontsize=8, facecolor=C['bg'],
               edgecolor=C['grid'], labelcolor=C['text'])
    ax1.tick_params(axis='x', colors=C['sub'], labelsize=7)

    # DDパネル: 戦略とB&Hを重ねて表示
    if bh_dd is not None:
        ax2.fill_between(bh_dd.index, bh_dd.values, 0,
                         where=bh_dd.values < 0,
                         color=C['bh'], alpha=0.25, label='B&H DD')
    ax2.fill_between(dd.index, dd.values, 0,
                     where=dd.values < 0,
                     color=C['sell'], alpha=0.5, label='Strategy DD')
    ax2.axhline(0, color=C['grid'], lw=0.8)
    ax2.set_ylabel('Drawdown%', color=C['sub'], fontsize=9)
    ax2.legend(loc='lower left', fontsize=7, facecolor=C['bg'],
               edgecolor=C['grid'], labelcolor=C['text'])
    ax2.tick_params(axis='x', colors=C['sub'], labelsize=7)

    plt.tight_layout(pad=0.5)
    return fig

# =========================================================
# Streamlit UI
# =========================================================
st.set_page_config(page_title='Trend Signal PRO', layout='wide')
st.markdown("""<style>
.stApp { background: #0d1117; color: #e6edf3; }
[data-testid="stMetricValue"] {
    color: #e6edf3 !important; font-size: 1.1rem !important;
}
</style>""", unsafe_allow_html=True)

for k, v in [('result', None), ('wf_result', None),
             ('current_code', ''), ('_quick_ticker', 'AAPL')]:
    if k not in st.session_state:
        st.session_state[k] = v

with st.sidebar:
    st.title("Settings")
    st.markdown("**Quick Select**")
    cols = st.columns(2)
    for i, (ticker, lbl) in enumerate(POPULAR):
        if cols[i % 2].button(lbl, key=f'pop_{ticker}', use_container_width=True):
            st.session_state['_quick_ticker'] = ticker
    st.divider()
    code     = st.text_input("Ticker", st.session_state['_quick_ticker']).upper().strip()
    period   = st.selectbox("Period",   PERIODS,   index=2)
    interval = st.selectbox("Interval", INTERVALS, index=0)

    if st.button("Analyze", type="primary", use_container_width=True):
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
                'df': df_s, 'code': code, 'period': period,
                'interval': interval,
                'at': datetime.now().strftime('%H:%M:%S'),
            }
        else:
            st.error(f"Data fetch failed: {code}")

res = st.session_state.get('result')
if res is None:
    st.info("Select a ticker in the sidebar and press Analyze.")
else:
    df = res['df']; disp_code = res['code']
    st.markdown(f"## {disp_code}  -  {res['at']}")

    tab1, tab2, tab3 = st.tabs(['Chart', 'Backtest', 'Walk-Forward Optimization'])

    with tab1:
        last      = df.iloc[-1]
        sig_label = {1:'BUY', -1:'SELL', 0:'NEUTRAL'}.get(int(last['sig']), 'NEUTRAL')
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Signal",    sig_label)
        c2.metric("Trend",     last['trend'])
        c3.metric("ADX",       f"{last['ADX']:.1f}")
        c4.metric("+DI / -DI", f"{last['DI_plus']:.1f} / {last['DI_minus']:.1f}")
        c5.metric("RSI",       f"{last['RSI']:.1f}")
        fig_c = make_chart(df, f"{disp_code} Signal Analysis")
        st.pyplot(fig_c, use_container_width=True)
        plt.close(fig_c)

    with tab2:
        bt = run_backtest(df, atr_mult=ATR_MULT, use_trail=USE_TRAIL,
                          time_stop=DEFAULT_SIG['time_stop'])
        if bt:
            s = bt['stats']

            # ---- ゴールはSharpe/Calmar/DD改善 ----
            st.markdown("### Risk-Adjusted Performance vs Buy & Hold")
            st.caption("目標: リターンではなく Sharpe・Calmar・MaxDD で B&H を上回る")

            c1, c2, c3 = st.columns(3)
            sharpe_delta = round(s['sharpe'] - s['bh_sharpe'], 2)
            calmar_delta = round(s['calmar'] - s['bh_calmar'], 2)
            dd_delta     = round(s['mdd'] - s['bh_mdd'], 1)   # 正値=改善

            c1.metric("Strategy Sharpe",  f"{s['sharpe']:.2f}",
                      delta=f"{sharpe_delta:+.2f} vs B&H",
                      delta_color="normal" if sharpe_delta >= 0 else "inverse")
            c2.metric("Strategy Calmar",  f"{s['calmar']:.2f}",
                      delta=f"{calmar_delta:+.2f} vs B&H",
                      delta_color="normal" if calmar_delta >= 0 else "inverse")
            c3.metric("Strategy MaxDD",   f"{s['mdd']:.1f}%",
                      delta=f"{dd_delta:+.1f}% vs B&H",
                      delta_color="normal" if dd_delta >= 0 else "inverse")

            st.divider()

            c4, c5, c6, c7, c8 = st.columns(5)
            c4.metric("Strategy Return", f"{s['sr']:.1f}%")
            c5.metric("Buy & Hold",      f"{s['bh']:.1f}%")
            c6.metric("Win Rate",        f"{s['wr']:.1f}%")
            c7.metric("Avg Win",         f"{s['aw']:.2f}%")
            c8.metric("Trades",           s['n'])

            fig_bt = make_bt_chart(bt, f"{disp_code} Backtest", label='Strategy')
            st.pyplot(fig_bt, use_container_width=True)
            plt.close(fig_bt)

            # B&H比較サマリー
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Strategy**")
                st.markdown(f"- Return: `{s['sr']:.1f}%`")
                st.markdown(f"- MaxDD: `{s['mdd']:.1f}%`")
                st.markdown(f"- Sharpe: `{s['sharpe']:.2f}`")
                st.markdown(f"- Calmar: `{s['calmar']:.2f}`")
            with col_b:
                st.markdown("**Buy & Hold**")
                st.markdown(f"- Return: `{s['bh']:.1f}%`")
                st.markdown(f"- MaxDD: `{s['bh_mdd']:.1f}%`")
                st.markdown(f"- Sharpe: `{s['bh_sharpe']:.2f}`")
                st.markdown(f"- Calmar: `{s['bh_calmar']:.2f}`")

            if bt['trades']:
                st.markdown("#### Trade Log")
                tdf = pd.DataFrame(bt['trades'])
                tdf['entry_date'] = pd.to_datetime(tdf['entry_date']).dt.strftime('%Y-%m-%d')
                tdf['exit_date']  = pd.to_datetime(tdf['exit_date']).dt.strftime('%Y-%m-%d')
                st.dataframe(tdf, use_container_width=True, height=250)

    with tab3:
        n_combos_disp = 1
        for v in PARAM_GRID.values(): n_combos_disp *= len(v)
        st.info(f"Grid: {n_combos_disp} combinations × n_splits folds "
                f"(~10-30 sec)  |  目標: Calmar・Sharpe・DD改善を最大化")

        n_splits = st.slider("Folds", 2, 6, 4)
        if st.button("Run Walk-Forward Optimization"):
            st.session_state['wf_result'] = None
            wf = walk_forward_optimize(
                disp_code, res['period'], res['interval'], n_splits=n_splits)
            st.session_state['wf_result'] = wf

        wf = st.session_state.get('wf_result')
        if wf:
            s = wf['full_bt']['stats']
            st.success(f"Done: {wf['n_combos']} combinations × {n_splits} folds")

            st.markdown("### Risk-Adjusted Performance (Out-of-Sample)")
            c1, c2, c3 = st.columns(3)
            sd = round(s['sharpe'] - s.get('bh_sharpe', 0), 2)
            cd = round(s.get('calmar', 0) - s.get('bh_calmar', 0), 2)
            dd = round(s['mdd'] - s['bh_mdd'], 1)
            c1.metric("WF Sharpe", f"{s['sharpe']:.2f}",
                      delta=f"{sd:+.2f} vs B&H",
                      delta_color="normal" if sd >= 0 else "inverse")
            c2.metric("WF Calmar", f"{s.get('calmar', 0):.2f}",
                      delta=f"{cd:+.2f} vs B&H",
                      delta_color="normal" if cd >= 0 else "inverse")
            c3.metric("WF MaxDD",  f"{s['mdd']:.1f}%",
                      delta=f"{dd:+.1f}% vs B&H",
                      delta_color="normal" if dd >= 0 else "inverse")

            c4, c5, c6 = st.columns(3)
            c4.metric("WF Return",  f"{s['sr']:.1f}%")
            c5.metric("Buy & Hold", f"{s['bh']:.1f}%")
            c6.metric("Trades",      s['n'])

            fig_wf = make_bt_chart(
                wf['full_bt'], f"{disp_code} WF Out-of-Sample",
                bt2=wf['default_bt'], label='WF Strategy')
            st.pyplot(fig_wf, use_container_width=True)
            plt.close(fig_wf)

            st.markdown("#### Best Params (last fold)")
            st.json(wf['best_p'])

            st.markdown("#### Fold Details")
            for r in wf['fold_results']:
                lbl = (f"Fold {r['fold']}: {r['test_start']} → {r['test_end']}"
                       f"  |  Train: {r['train_n']} bars  |  Score: {r['score']}")
                with st.expander(lbl):
                    st.write("Best Params:", r['best_p'])
                    if r['tbt']:
                        rs = r['tbt']['stats']
                        fc1, fc2, fc3, fc4 = st.columns(4)
                        fc1.metric("Return",  f"{rs['sr']:.1f}%")
                        fc2.metric("MaxDD",   f"{rs['mdd']:.1f}%")
                        fc3.metric("Sharpe",  f"{rs['sharpe']:.2f}")
                        fc4.metric("Calmar",  f"{rs['calmar']:.2f}")
