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
PERIODS = ['2y', '3y', '5y', '10y']
INTERVALS = ['1d', '1wk']
POPULAR = [
    ('1326.T', 'SPDR Gold'), ('7203.T', 'Toyota'), ('6758.T', 'Sony'),
    ('9984.T', 'SBG'), ('6861.T', 'Keyence'), ('8306.T', 'MUFG'),
    ('^N225', 'Nikkei225'), ('AAPL', 'Apple'), ('NVDA', 'NVIDIA'), ('^GSPC', 'SP500'),
]
C = {
    'bg': '#0d1117', 'panel': '#161b22', 'grid': '#21262d',
    'text': '#e6edf3', 'sub': '#8b949e',
    'buy': '#3fb950', 'sell': '#f85149', 'neutral': '#58a6ff',
    'sma25': '#ffa657', 'sma75': '#58a6ff', 'sma200': '#bc8cff',
    'bb': '#388bfd', 'macd': '#58a6ff', 'msig': '#ffa657',
    'hup': '#3fb950', 'hdn': '#f85149', 'rsi': '#d2a8ff',
    'cup': '#3fb950', 'cdn': '#f85149',
}

DEFAULT_PARAMS = {
    'w_trend': 1, 'w_macd': 2,
    'rsi_buy_th': 30, 'rsi_sell_th': 70,
    'adx_th': 25,
    'stoch_buy_th': 25, 'stoch_sell_th': 75,
    'buy_th': 3, 'sell_th': 3,
}

PARAM_GRID = {
    'w_trend': [1, 2],
    'w_macd': [1, 2, 3],
    'rsi_buy_th': [25, 30, 35],
    'rsi_sell_th': [65, 70, 75],
    'adx_th': [20, 25],
    'stoch_buy_th': [20, 25, 30],
    'stoch_sell_th': [70, 75, 80],
    'buy_th': [2, 3, 4],
    'sell_th': [2, 3, 4],
}

# --- 共通関数 ---
def flatten_df(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df

def compute_indicators(df):
    cl = df['Close']; hi = df['High']; lo = df['Low']; vo = df['Volume']
    df['SMA25']  = cl.rolling(25).mean()
    df['SMA75']  = cl.rolling(75).mean()
    df['SMA200'] = cl.rolling(200).mean()
    bb = ta.volatility.BollingerBands(cl, 20, 2)
    df['BB_u'] = bb.bollinger_hband()
    df['BB_m'] = bb.bollinger_mavg()
    df['BB_l'] = bb.bollinger_lband()
    df['BB_w'] = (df['BB_u'] - df['BB_l']) / df['BB_m']
    mc = ta.trend.MACD(cl, 26, 12, 9)
    df['MACD'] = mc.macd(); df['MSIG'] = mc.macd_signal(); df['MHST'] = mc.macd_diff()
    df['RSI'] = ta.momentum.RSIIndicator(cl, 14).rsi()
    df['ATR'] = ta.volatility.AverageTrueRange(hi, lo, cl, 14).average_true_range()
    adx = ta.trend.ADXIndicator(hi, lo, cl, 14)
    df['ADX'] = adx.adx()
    sto = ta.momentum.StochasticOscillator(hi, lo, cl, 14, 3)
    df['SK'] = sto.stoch(); df['SD'] = sto.stoch_signal()
    df['VMA'] = vo.rolling(20).mean()
    return df

def compute_signals(df, p):
    s = df
    mxu = (s['MACD'] > s['MSIG']) & (s['MACD'].shift(1) <= s['MSIG'].shift(1))
    mxd = (s['MACD'] < s['MSIG']) & (s['MACD'].shift(1) >= s['MSIG'].shift(1))
    sxu = (s['SK'] > s['SD']) & (s['SK'].shift(1) <= s['SD'].shift(1))
    sxd = (s['SK'] < s['SD']) & (s['SK'].shift(1) >= s['SD'].shift(1))

    bsc = (
        (s['SMA25'] > s['SMA75']).astype(int) * p['w_trend'] +
        mxu.astype(int) * p['w_macd'] +
        ((s['RSI'] > p['rsi_buy_th']) & (s['RSI'].shift(1) <= p['rsi_buy_th'])).astype(int) +
        (s['Close'] <= s['BB_l'] * 1.01).astype(int) +
        (s['ADX'] > p['adx_th']).astype(int) +
        (sxu & (s['SK'] < p['stoch_buy_th'])).astype(int)
    )
    ssc = (
        (s['SMA25'] < s['SMA75']).astype(int) * p['w_trend'] +
        mxd.astype(int) * p['w_macd'] +
        ((s['RSI'] < p['rsi_sell_th']) & (s['RSI'].shift(1) >= p['rsi_sell_th'])).astype(int) +
        (s['Close'] >= s['BB_u'] * 0.99).astype(int) +
        (s['ADX'] > p['adx_th']).astype(int) +
        (sxd & (s['SK'] > p['stoch_sell_th'])).astype(int)
    )
    df = df.copy()
    df['bsc'] = bsc; df['ssc'] = ssc; df['sig'] = 0
    df.loc[bsc >= p['buy_th'], 'sig'] = 1
    df.loc[ssc >= p['sell_th'], 'sig'] = -1
    df['trend'] = 'Range'
    df.loc[(s['SMA25'] > s['SMA75']) & (s['SMA75'] > s['SMA200']) & (s['ADX'] > 20), 'trend'] = 'Up'
    df.loc[(s['SMA25'] < s['SMA75']) & (s['SMA75'] < s['SMA200']) & (s['ADX'] > 20), 'trend'] = 'Down'
    return df

@st.cache_data(ttl=60, show_spinner=False)
def fetch_raw(code, period, interval):
    try:
        df = yf.download(code, period=period, interval=interval, auto_adjust=True, progress=False)
        if df is None or df.empty: return None
        df = flatten_df(df)
        df = df.dropna(subset=['Close', 'Open', 'High', 'Low', 'Volume'])
        if len(df) < 120: return None
        return df
    except:
        return None

def run_backtest(df, cost=0.001, initial_equity=1.0):
    cl = df['Close'].values; sig = df['sig'].values; dates = df.index
    trades = []; eq = initial_equity; equity = [eq]; pos = 0; ep = 0.0; ed = None

    for i in range(1, len(df)):
        p = cl[i]; ps = sig[i - 1]
        if pos == 0 and ps == 1:
            pos = 1; ep = p * (1 + cost); ed = dates[i]
        elif pos == 1 and ps == -1:
            xp = p * (1 - cost); ret = (xp - ep) / ep
            eq *= (1 + ret)
            trades.append({'entry_date': ed, 'exit_date': dates[i],
                           'entry': ep, 'exit': xp, 'ret': ret * 100,
                           'result': 'Win' if ret > 0 else 'Loss'})
            pos = 0
        equity.append(eq)

    eq_s = pd.Series(equity, index=dates)
    bh = (cl[-1] - cl[0]) / cl[0] * 100
    n = len(trades)
    wins = [t for t in trades if t['ret'] > 0]
    loss = [t for t in trades if t['ret'] <= 0]
    wr = len(wins) / n * 100 if n > 0 else 0
    aw = np.mean([t['ret'] for t in wins]) if wins else 0
    al = np.mean([t['ret'] for t in loss]) if loss else 0
    pf = abs(sum(t['ret'] for t in wins) / sum(t['ret'] for t in loss)) if loss else 999.0
    roll_max = eq_s.cummax()
    dd = (eq_s - roll_max) / roll_max * 100
    mdd = dd.min()
    yrs = (dates[-1] - dates[0]).days / 365.25
    cagr = ((eq / initial_equity) ** (1 / yrs) - 1) * 100 if yrs > 0 else 0
    dr = eq_s.pct_change().dropna()
    sharpe = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0

    return {'trades': trades, 'equity': eq_s, 'drawdown': dd,
            'stats': {'n': n, 'wr': wr, 'aw': aw, 'al': al, 'pf': pf,
                      'sr': (eq / initial_equity - 1) * 100, 'bh': bh, 'mdd': mdd, 'cagr': cagr, 'sharpe': sharpe}}

def score_params(df, p, cost):
    # DEFAULT_PARAMSをベースにマージ → PARAM_GRIDに不足キーがあってもKeyErrorにならない
    full_p = {**DEFAULT_PARAMS, **p}
    df2 = compute_signals(df, full_p)
    bt = run_backtest(df2, cost)
    if bt is None or bt["stats"]["n"] < 2: return -999
    s = bt["stats"]
    return s["sr"] - max(0, -s["mdd"]) * 0.5 + s["sharpe"] * 3

# --- ウォークフォワード最適化 ---
@st.cache_data(ttl=300, show_spinner=False)
def walk_forward_optimize(code, period, interval, n_splits=4, cost=0.001):
    raw = fetch_raw(code, period, interval)
    if raw is None: return None
    base = compute_indicators(raw.copy())

    keys = list(PARAM_GRID.keys())
    vals = list(PARAM_GRID.values())
    combos = list(itertools.product(*vals))
    all_params = [dict(zip(keys, c)) for c in combos]

    n = len(base)
    split_size = n // (n_splits + 1)

    fold_results = []
    combined_trades = []
    combined_equity_series = []
    current_initial_eq = 1.0

    progress = st.progress(0, 'Walk-forward optimization...')

    for fold in range(n_splits):
        train_end = (fold + 1) * split_size
        test_start = train_end
        test_end = test_start + split_size if fold < n_splits - 1 else n

        train = base.iloc[:train_end].copy()
        test  = base.iloc[test_start:test_end].copy()

        if len(train) < 60 or len(test) < 20: continue

        best_score_fold = -999
        best_idx_fold = 0
        for idx, p in enumerate(all_params):
            sc = score_params(train, p, cost)
            if sc > best_score_fold:
                best_score_fold = sc
                best_idx_fold = idx

        best_p = all_params[best_idx_fold]
        test_df = compute_signals(test.copy(), best_p)
        test_bt = run_backtest(test_df, cost, initial_equity=current_initial_eq)

        if test_bt:
            combined_trades.extend(test_bt['trades'])
            combined_equity_series.append(test_bt['equity'])
            current_initial_eq = test_bt['equity'].iloc[-1]

        fold_results.append({
            'fold': fold + 1,
            'test_start': base.index[test_start].strftime('%Y/%m'),
            'test_end': base.index[min(test_end - 1, n - 1)].strftime('%Y/%m'),
            'best_params': best_p,
            'test_bt': test_bt,
        })
        progress.progress((fold + 1) / n_splits, f'Fold {fold + 1}/{n_splits} done')

    progress.empty()

    if not combined_equity_series: return None

    full_equity = pd.concat(combined_equity_series)
    full_equity = full_equity[~full_equity.index.duplicated(keep='first')]

    roll_max = full_equity.cummax()
    full_dd = (full_equity - roll_max) / roll_max * 100

    full_stats = {
        'n': len(combined_trades),
        'sr': (full_equity.iloc[-1] - 1.0) * 100,
        'bh': ((base['Close'].iloc[-1] - base['Close'].iloc[0]) / base['Close'].iloc[0]) * 100,
        'mdd': full_dd.min(),
        'sharpe': full_equity.pct_change().mean() / full_equity.pct_change().std() * np.sqrt(252)
                  if full_equity.pct_change().std() > 0 else 0
    }

    default_df = compute_signals(base.copy(), DEFAULT_PARAMS)
    default_bt = run_backtest(default_df, cost)

    return {
        'best_params': fold_results[-1]['best_params'],
        'full_bt': {'trades': combined_trades, 'equity': full_equity, 'drawdown': full_dd, 'stats': full_stats},
        'default_bt': default_bt,
        'fold_results': fold_results,
        'base': base,
    }

# --- 描画関数 ---

def _style_ax(ax):
    """共通のAxスタイリング"""
    ax.set_facecolor(C['panel'])
    ax.tick_params(colors=C['sub'], labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor(C['grid'])
    ax.grid(color=C['grid'], lw=0.5, ls='--', alpha=0.5)

def draw_candles(ax, df):
    """ローソク足描画 (修正済: C['cdn'] typo fix)"""
    op = df['Open'].values; hi = df['High'].values
    lo = df['Low'].values; cl = df['Close'].values
    xs = np.arange(len(df))
    for i in range(len(df)):
        # 【バグ修正】C[['cdn']] → C['cdn']
        col = C['cup'] if cl[i] >= op[i] else C['cdn']
        ax.plot([i, i], [lo[i], hi[i]], color=col, lw=0.7, zorder=2)
        b0 = min(op[i], cl[i]); b1 = max(op[i], cl[i])
        ax.bar(i, max(b1 - b0, 1e-6), bottom=b0, width=0.6, color=col, linewidth=0, zorder=3)
    return xs

def make_chart(df, title, mobile=False):
    """
    メインチャート作成
    5パネル構成:
      [0] ローソク足 + SMA + BB + シグナルマーカー
      [1] 出来高
      [2] MACD
      [3] RSI
      [4] Stochastic
    """
    w, h = (9, 14) if mobile else (16, 13)
    fig = plt.figure(figsize=(w, h), facecolor=C['bg'])
    gs = gridspec.GridSpec(5, 1, figure=fig,
                           height_ratios=[4, 1, 1.3, 1.3, 1.3], hspace=0.05)
    axes = [fig.add_subplot(gs[i]) for i in range(5)]
    for ax in axes:
        _style_ax(ax)

    # 表示する直近N本に絞る（見やすさのため）
    display_n = 200
    plot_df = df.iloc[-display_n:].copy().reset_index(drop=False)
    n = len(plot_df)
    xs = np.arange(n)

    # ---- パネル0: ローソク足 + SMA + BB + シグナル ----
    ax0 = axes[0]
    draw_candles(ax0, plot_df)

    # SMA
    ax0.plot(xs, plot_df['SMA25'],  color=C['sma25'],  lw=1.2, label='SMA25',  zorder=4)
    ax0.plot(xs, plot_df['SMA75'],  color=C['sma75'],  lw=1.2, label='SMA75',  zorder=4)
    ax0.plot(xs, plot_df['SMA200'], color=C['sma200'], lw=1.2, label='SMA200', zorder=4)

    # ボリンジャーバンド
    ax0.plot(xs, plot_df['BB_u'], color=C['bb'], lw=0.8, ls='--', alpha=0.7)
    ax0.plot(xs, plot_df['BB_l'], color=C['bb'], lw=0.8, ls='--', alpha=0.7)
    ax0.fill_between(xs, plot_df['BB_u'], plot_df['BB_l'],
                     color=C['bb'], alpha=0.07)

    # シグナルマーカー
    buy_idx  = plot_df.index[plot_df['sig'] == 1].tolist()
    sell_idx = plot_df.index[plot_df['sig'] == -1].tolist()
    if buy_idx:
        ax0.scatter(buy_idx, plot_df.loc[buy_idx, 'Low'] * 0.995,
                    marker='^', color=C['buy'], s=60, zorder=6, label='Buy')
    if sell_idx:
        ax0.scatter(sell_idx, plot_df.loc[sell_idx, 'High'] * 1.005,
                    marker='v', color=C['sell'], s=60, zorder=6, label='Sell')

    # タイトル・凡例
    ax0.set_title(title, color=C['text'], fontsize=11, pad=6)
    ax0.legend(loc='upper left', fontsize=7, facecolor=C['bg'],
               edgecolor=C['grid'], labelcolor=C['text'], ncol=4)
    ax0.set_xlim(-1, n)
    ax0.set_xticks([])
    ax0.set_ylabel('Price', color=C['sub'], fontsize=8)

    # ---- パネル1: 出来高 ----
    ax1 = axes[1]
    vol_colors = [C['cup'] if plot_df['Close'].iloc[i] >= plot_df['Open'].iloc[i]
                  else C['cdn'] for i in range(n)]
    ax1.bar(xs, plot_df['Volume'], color=vol_colors, width=0.7, alpha=0.8, zorder=3)
    ax1.plot(xs, plot_df['VMA'], color=C['neutral'], lw=1.0, zorder=4)
    ax1.set_xlim(-1, n)
    ax1.set_xticks([])
    ax1.set_ylabel('Vol', color=C['sub'], fontsize=8)

    # ---- パネル2: MACD ----
    ax2 = axes[2]
    ax2.plot(xs, plot_df['MACD'], color=C['macd'], lw=1.2, label='MACD', zorder=3)
    ax2.plot(xs, plot_df['MSIG'], color=C['msig'], lw=1.0, ls='--', label='Signal', zorder=3)
    hist = plot_df['MHST'].values
    hcols = [C['hup'] if v >= 0 else C['hdn'] for v in hist]
    ax2.bar(xs, hist, color=hcols, width=0.7, alpha=0.75, zorder=2)
    ax2.axhline(0, color=C['grid'], lw=0.8)
    ax2.set_xlim(-1, n)
    ax2.set_xticks([])
    ax2.set_ylabel('MACD', color=C['sub'], fontsize=8)
    ax2.legend(loc='upper left', fontsize=7, facecolor=C['bg'],
               edgecolor=C['grid'], labelcolor=C['text'])

    # ---- パネル3: RSI ----
    ax3 = axes[3]
    ax3.plot(xs, plot_df['RSI'], color=C['rsi'], lw=1.2, zorder=3)
    ax3.axhline(70, color=C['sell'], lw=0.8, ls='--', alpha=0.7)
    ax3.axhline(30, color=C['buy'],  lw=0.8, ls='--', alpha=0.7)
    ax3.axhline(50, color=C['grid'], lw=0.6)
    ax3.fill_between(xs, plot_df['RSI'], 70,
                     where=plot_df['RSI'] >= 70, color=C['sell'], alpha=0.2)
    ax3.fill_between(xs, plot_df['RSI'], 30,
                     where=plot_df['RSI'] <= 30, color=C['buy'],  alpha=0.2)
    ax3.set_ylim(0, 100)
    ax3.set_xlim(-1, n)
    ax3.set_xticks([])
    ax3.set_ylabel('RSI', color=C['sub'], fontsize=8)

    # ---- パネル4: Stochastic ----
    ax4 = axes[4]
    ax4.plot(xs, plot_df['SK'], color=C['buy'],  lw=1.1, label='%K', zorder=3)
    ax4.plot(xs, plot_df['SD'], color=C['msig'], lw=1.0, ls='--', label='%D', zorder=3)
    ax4.axhline(75, color=C['sell'], lw=0.8, ls='--', alpha=0.7)
    ax4.axhline(25, color=C['buy'],  lw=0.8, ls='--', alpha=0.7)
    ax4.set_ylim(0, 100)
    ax4.set_xlim(-1, n)
    ax4.set_ylabel('Stoch', color=C['sub'], fontsize=8)
    ax4.legend(loc='upper left', fontsize=7, facecolor=C['bg'],
               edgecolor=C['grid'], labelcolor=C['text'])

    # X軸に日付ラベルを設定（パネル4のみ）
    tick_step = max(1, n // 10)
    tick_positions = list(range(0, n, tick_step))
    date_col = plot_df.columns[0]  # index列（reset_index後の'Date'またはindex名）
    try:
        date_labels = [str(plot_df.iloc[i][date_col])[:10] for i in tick_positions]
    except Exception:
        date_labels = [str(i) for i in tick_positions]
    ax4.set_xticks(tick_positions)
    ax4.set_xticklabels(date_labels, rotation=30, ha='right',
                        fontsize=6, color=C['sub'])

    plt.tight_layout(pad=0.5)
    return fig


def make_bt_chart(bt, title, bt2=None):
    """バックテスト結果チャート（エクイティカーブ + ドローダウン）"""
    eq = bt['equity']; dd = bt['drawdown']
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), facecolor=C['bg'],
                             gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.06})
    for ax in axes:
        _style_ax(ax)

    ax1, ax2 = axes
    ax1.plot(eq.index, eq.values, color=C['buy'], lw=2.0,
             label='Walk-Forward Strategy', zorder=3)
    if bt2 is not None:
        ax1.plot(bt2['equity'].index, bt2['equity'].values,
                 color='#ffa657', lw=1.5, ls='-.', label='Default Strategy', zorder=2)

    ax1.axhline(1.0, color=C['grid'], lw=0.8)
    ax1.set_title(title, color=C['text'], fontsize=11)
    ax1.set_ylabel('Equity (normalized)', color=C['sub'], fontsize=9)
    ax1.legend(loc='upper left', fontsize=8, facecolor=C['bg'],
               edgecolor=C['grid'], labelcolor=C['text'])
    ax1.tick_params(axis='x', colors=C['sub'], labelsize=7, rotation=0)

    ax2.fill_between(dd.index, dd.values, 0,
                     where=dd.values < 0, color=C['sell'], alpha=0.45)
    ax2.axhline(0, color=C['grid'], lw=0.8)
    ax2.set_ylabel('Drawdown%', color=C['sub'], fontsize=9)
    ax2.tick_params(axis='x', colors=C['sub'], labelsize=7)

    plt.tight_layout(pad=0.5)
    return fig

# =========================================================
# --- Streamlit アプリ本体 ---
# =========================================================
st.set_page_config(page_title='Trend Signal PRO', layout='wide')
st.markdown(f"""<style>
.stApp {{background:#0d1117; color:#e6edf3;}}
[data-testid="stMetricValue"] {{color:#e6edf3!important; font-size:1.2rem!important;}}
.bb{{background:#1a3d24;color:#3fb950;border:1px solid #3fb950;padding:5px 15px;border-radius:5px;font-weight:bold;}}
.bs{{background:#3d1a1a;color:#f85149;border:1px solid #f85149;padding:5px 15px;border-radius:5px;font-weight:bold;}}
</style>""", unsafe_allow_html=True)

# セッションステート初期化
if 'active_params' not in st.session_state:
    st.session_state['active_params'] = DEFAULT_PARAMS
if 'wf_result' not in st.session_state:
    st.session_state['wf_result'] = None
if 'result' not in st.session_state:
    st.session_state['result'] = None
if 'current_code' not in st.session_state:
    st.session_state['current_code'] = ''

with st.sidebar:
    st.title("⚙️ Settings")

    # ---- ポピュラー銘柄クイック選択 ----
    st.markdown("**Quick Select**")
    cols = st.columns(2)
    for i, (ticker, label) in enumerate(POPULAR):
        if cols[i % 2].button(label, key=f'pop_{ticker}', use_container_width=True):
            st.session_state['_quick_ticker'] = ticker

    st.divider()

    # クイック選択で上書きされた場合に反映
    default_ticker = st.session_state.pop('_quick_ticker', 'AAPL')
    code     = st.text_input("Ticker", default_ticker).upper().strip()
    period   = st.selectbox("Period",   PERIODS,    index=2)
    interval = st.selectbox("Interval", INTERVALS,  index=0)

    if st.button("🔍 Analyze", type="primary", use_container_width=True):
        # 【バグ修正】銘柄変更時に古い結果を必ずクリアする
        st.session_state['result']      = None
        st.session_state['wf_result']   = None
        st.session_state['current_code'] = code

        with st.spinner(f"Fetching {code} ..."):
            raw = fetch_raw(code, period, interval)

        if raw is not None:
            df_ind = compute_indicators(raw.copy())
            df_sig = compute_signals(df_ind, st.session_state['active_params'])
            st.session_state['result'] = {
                'df':   df_sig,
                'code': code,
                'at':   datetime.now().strftime('%H:%M:%S'),
            }
        else:
            st.error(f"❌ データ取得失敗: {code}\n期間を短くするか、ティッカーを確認してください。")

# ---- メインエリア ----
res = st.session_state.get('result')

if res is None:
    st.info("👈 左のサイドバーからティッカーを入力して **Analyze** を押してください。")
else:
    df   = res['df']
    disp_code = res['code']

    st.markdown(f"## {disp_code}  –  Updated: {res['at']}")

    tab1, tab2, tab3 = st.tabs(['📈 Live Chart', '🧪 Backtest', '🔬 Walk-Forward Optimization'])

    # ==== Tab 1: ライブチャート ====
    with tab1:
        # 最新シグナル表示
        last = df.iloc[-1]
        sig_val   = last['sig']
        trend_val = last['trend']
        sig_label = {1: '🟢 BUY', -1: '🔴 SELL', 0: '⚪ NEUTRAL'}.get(sig_val, '⚪ NEUTRAL')
        trend_color = {'Up': C['buy'], 'Down': C['sell'], 'Range': C['neutral']}.get(trend_val, C['neutral'])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Signal",      sig_label)
        c2.metric("Trend",       trend_val)
        c3.metric("RSI",         f"{last['RSI']:.1f}")
        c4.metric("ADX",         f"{last['ADX']:.1f}")

        fig_chart = make_chart(df, f"{disp_code} – Signal Analysis")
        st.pyplot(fig_chart, use_container_width=True)
        plt.close(fig_chart)

    # ==== Tab 2: バックテスト ====
    with tab2:
        bt = run_backtest(df)
        if bt:
            s = bt['stats']
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Return",        f"{s['sr']:.2f}%")
            c2.metric("Buy & Hold",    f"{s['bh']:.2f}%")
            c3.metric("Max Drawdown",  f"{s['mdd']:.2f}%")
            c4.metric("Sharpe",        f"{s['sharpe']:.2f}")
            c5.metric("Trades",        s['n'])

            c6, c7, c8 = st.columns(3)
            c6.metric("Win Rate",      f"{s['wr']:.1f}%")
            c7.metric("Avg Win",       f"{s['aw']:.2f}%")
            c8.metric("Avg Loss",      f"{s['al']:.2f}%")

            fig_bt = make_bt_chart(bt, f"{disp_code} – Backtest Result")
            st.pyplot(fig_bt, use_container_width=True)
            plt.close(fig_bt)

            if bt['trades']:
                st.markdown("#### Trade Log")
                tdf = pd.DataFrame(bt['trades'])
                tdf['entry_date'] = pd.to_datetime(tdf['entry_date']).dt.strftime('%Y-%m-%d')
                tdf['exit_date']  = pd.to_datetime(tdf['exit_date']).dt.strftime('%Y-%m-%d')
                tdf['ret']        = tdf['ret'].round(2)
                tdf['entry']      = tdf['entry'].round(2)
                tdf['exit']       = tdf['exit'].round(2)
                st.dataframe(tdf, use_container_width=True, height=250)

    # ==== Tab 3: ウォークフォワード最適化 ====
    with tab3:
        st.markdown("各Foldでトレーニング期間のみでパラメータ最適化し、未知のテスト期間に適用します。")
        n_splits = st.slider("Splits (Folds)", 2, 6, 4)

        if st.button("🚀 Run Walk-Forward Optimization"):
            # 銘柄が変わっていたら古いWF結果もクリア
            st.session_state['wf_result'] = None
            wf = walk_forward_optimize(disp_code, period, interval, n_splits=n_splits)
            st.session_state['wf_result'] = wf

        wf = st.session_state.get('wf_result')
        if wf:
            st.success("✅ Walk-Forward 完了 – 以下はアウト・オブ・サンプルの連結結果です。")
            s = wf['full_bt']['stats']
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Concat Return",  f"{s['sr']:.1f}%")
            c2.metric("Max Drawdown",   f"{s['mdd']:.1f}%")
            c3.metric("Sharpe",         f"{s['sharpe']:.2f}")
            c4.metric("Trades",          s['n'])

            fig_wf = make_bt_chart(wf['full_bt'],
                                   f"{disp_code} – WF Out-of-Sample Performance",
                                   bt2=wf['default_bt'])
            st.pyplot(fig_wf, use_container_width=True)
            plt.close(fig_wf)

            st.markdown("#### 最適パラメータ (最終Fold)")
            st.json(wf['best_params'])

            st.markdown("#### Fold Details")
            for r in wf['fold_results']:
                with st.expander(f"Fold {r['fold']}: {r['test_start']} → {r['test_end']}"):
                    st.write("**Best Params:**", r['best_params'])
                    if r['test_bt']:
                        rs = r['test_bt']['stats']
                        fc1, fc2, fc3 = st.columns(3)
                        fc1.metric("Test Return",   f"{rs['sr']:.2f}%")
                        fc2.metric("Max Drawdown",  f"{rs['mdd']:.2f}%")
                        fc3.metric("Trades",         rs['n'])
