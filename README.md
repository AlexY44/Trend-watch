# 📈 トレンドシグナル分析ツール

西山孝四郎式 複合テクニカルシグナルシステム  
Streamlit Community Cloud で無料公開 → iPhone で閲覧可能

---

## 🚀 クラウドへのデプロイ手順（無料）

### ステップ 1: GitHub リポジトリを作成

1. [https://github.com](https://github.com) にログイン（なければ無料で登録）
2. 右上「+」→「New repository」をクリック
3. Repository name に **trend-signal** など好きな名前を入力
4. **Public** を選択（Streamlit Cloud の無料枠は Public のみ）
5. 「Create repository」をクリック

---

### ステップ 2: ファイルをアップロード

GitHub のリポジトリページで：

1. 「Add file」→「Upload files」をクリック
2. このフォルダにある **2ファイル** をドラッグ＆ドロップ：
   - `trend_app.py`
   - `requirements.txt`
3. 「Commit changes」をクリック

---

### ステップ 3: Streamlit Cloud にデプロイ

1. [https://streamlit.io/cloud](https://streamlit.io/cloud) を開く
2. 「Sign up for free」→ **GitHub アカウントでログイン**
3. 「New app」をクリック
4. 以下を選択：
   - **Repository**: 先ほど作ったリポジトリ
   - **Branch**: main
   - **Main file path**: `trend_app.py`
5. 「Deploy!」をクリック

⏳ 3〜5分でデプロイ完了。自動的に URL が発行されます。

---

### ステップ 4: iPhone で開く

発行された URL（例: `https://あなたの名前-trend-signal-xxxxx.streamlit.app`）を  
iPhone の Safari で開くだけ！

**ホーム画面に追加する方法（アプリっぽく使える）:**
1. Safari でページを開く
2. 下部の「共有」ボタン（四角＋矢印）をタップ
3. 「ホーム画面に追加」をタップ
4. 「追加」をタップ

---

## 📱 使い方

1. 銘柄コードを入力（例: `7203.T` `AAPL` `^N225`）
2. 「🔍 分析」ボタンをタップ
3. サイドバー（☰ マーク）から自動更新間隔を設定して「▶ 開始」

### 銘柄コード例

| 種類 | コード |
|------|--------|
| トヨタ | `7203.T` |
| ソニー | `6758.T` |
| ソフトバンクG | `9984.T` |
| 日経225 | `^N225` |
| Apple | `AAPL` |
| NVIDIA | `NVDA` |
| S&P500 | `^GSPC` |
| ドル円 | `USDJPY=X` |

---

## 🔧 シグナルロジック

複合スコア方式（3点以上でシグナル発生・最大7点）

| 条件 | 点数 |
|------|:---:|
| SMA25 ＞/＜ SMA75 | 1 |
| MACDゴールデン/デッドクロス | **2** |
| RSI 30超え / 70割れ | 1 |
| ボリンジャーバンド下限/上限付近 | 1 |
| ADX ＞ 25（トレンド強度） | 1 |
| ストキャスGC/DC + 過熱ゾーン | 1 |

---

## ⚠️ 免責事項

このツールは情報提供のみを目的としており、投資の助言ではありません。  
投資判断はご自身の責任で行ってください。
