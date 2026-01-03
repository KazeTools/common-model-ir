# Common Model IR (`.cm`) 言語仕様（README）

## 目的

`.cm`（Common Model IR）は

> **論文・数式・モデルの「世界構造」を、人間とAIが共通に読める正規形で記述するための仕様**

である。

これは **プログラミング言語ではない**。
Stan / PyTorch / JuMP / FEniCS の代替でもない。

`.cm` は、

> **論文のモデル節（生成過程・制約・方程式・目的関数）を、そのまま写す中間表現（IR）**

である。

実装・数値解法・アルゴリズム選択は **Common Model IR の責務外**とする。

---

## 基本思想

`.cm` は常に次の 7 ブロックから構成される：

```
model
index
given
unknown
define
rules
want
```

この 7 ブロックのみで、

* 統計モデル
* 数理最適化
* 常微分方程式 / 偏微分方程式
* 物理法則
* ニューラルネットワーク
* Physics-Informed NN

を **同一の抽象構造**で表現する。

---

## 各ブロックの意味

### `model`

モデル名。論文名・節名・理論名。

```yaml
model LinearRegression:
model SIR:
model MaxwellFDTD:
model GeneralRelativity:
```

識別子であり、意味論は持たない。

---

### `index`

添字の **意味** を定義する。

```yaml
index:
  i : sample
  t : time
  x : spacetime
  d : feature
  l : layer
```

* 数値範囲・サイズは書かない
* **意味のみを書く**
* for による量化の単位になる

---

### `given`

外部から与えられる量。

* 観測データ
* 定数
* 既知の場
* 境界条件
* モデル構造（層数など）

```yaml
given:
  x@train[i,d] : real
  y@train[i]   : real
  beta         : real
  N            : real
```

`@train / @pred` は **観測・非観測の区別を示す注釈**であり、
意味論の補助である（必須ではないが推奨）。

---

### `unknown`

この world で **解かれる量・推定される量**。

* パラメータ
* 状態変数
* フィールド
* 設計変数

```yaml
unknown:
  w[d]
  sigma
  g[x,a,b]
```

#### 原則

* **unknown は最小にする**
* 中間量・派生量は入れない
* 「自由に選べる量」のみを書く

---

### `define`

`unknown` と `given` から **派生する量**。

* 中間テンソル
* attention, Ricci tensor
* softmax
* 逆行列
* 数式として定義される関数

```yaml
define:
  mu[i] = sum_d x[i,d] * w[d] + b
  g_inv[x,a,b] = inverse(g[x,a,b])
```

#### define の意味論

* `=` を使用可
* 制約ではない（破ってもよいものではなく、**定義**）
* マクロ的定義・数式展開のための場所

加えて、`define` は **意味が一意に定まらない量を明示的に露出させる場所**でもある。

* 論文中で参照されるが定義されない量
* 慣習・実装・付録・コードなどから暗黙に補われる量
* 複数の自然な解釈が存在する量

これらは **`define` から外に逃がさず**、未確定性として記述する。

##### 未確定な定義の表現

```yaml
define:
  a(x) = ?          # 未確定（候補不明）
  b(x) = ?{d(x), e(x)}    # 未確定だが候補が列挙できる
  c(x) = ?{f(x)}    # 暗黙的に f(x) が有力候補
```

* `?`
  未確定な量。論文・前提から一意に定まらない。
* `?{...}`
  未確定だが、候補が列挙できる場合。
  デフォルトでは **暗黙候補（implicit）** を意味する。

これらは **評価・実装上の選択を強制しない**ための表現であり、
制約（rules）ではない。

##### define の粒度

define の展開粒度は **厳密に規定しない**。

目的は：

* 論文のモデル構造が追えること
* ブラックボックスを避けること
* rules を読みやすくすること
* **世界がどこで閉じていないかを可視化すること**

そのため、

* 解法
* 数値近似
* 最適化アルゴリズム
* 実装依存の詳細

は原則として `define` に閉じるか、もしくは明示的に省略する。

---

### `rules`

この世界が従う **法則・制約・方程式・尤度**。

ここでは **代入は禁止**。

使用できる記法は：

* 等式制約：`==`
* 不等式：`< <= > >=`
* 確率的関係：`~`

```yaml
rules:
  sigma > 0
  for i:
    y[i] ~ Normal(mu[i], sigma)
```

以下はすべて rules に書く：

* ODE / PDE
* 物理法則
* 制約条件
* 尤度
* 最適化制約

---

### `for`（量化）

`for` は **量化（∀）** を表す。

```yaml
for t:
  ∂S[t]/∂t == ...
```

意味：

> 任意の t について、この制約が成り立つ

* ループではない
* 実行順序を持たない
* 時間発展のアルゴリズムを意味しない

---

### 数学記号（ホワイトリスト）

使用可能な基本演算：

* 四則演算 `+ - * / ^`
* `sum`
* `exp`
* `sqrt`
* `max`
* `min`
* `∂`
* `∇`

それ以外の関数は **直接使わない**。
必要なら `define` に展開して書く。

---

### `want`

この world から **何を求めるか**。

使用可能な動詞は **3つだけ**：

```yaml
want:
  fit ...
  predict ...
  optimize ...
```

#### 意味

* `fit`
  given があるとき、unknown をデータに合わせて推定する

* `predict`
  rules に従って unknown や派生量を生成する

* `optimize`
  目的関数を最適化する

---

## Common Model IR の意味論（要約）

`.cm` は、

> **unknown と define の上に、rules という制約集合を張り、
> want でそれをどう使うかを指定する**

という言語である。

Stan / JuMP / PDE / NN / 物理モデルは
すべてこの構造に落ちる。

---

## 論文から `.cm` への変換手順

1. 添字を洗い出す → `index`
2. 外部から与えられる量 → `given`
3. 解きたい量 → `unknown`
4. 中間定義・テンソル → `define`
5. 方程式・尤度・制約 → `rules`
6. 論文の目的 → `want`

---

## `.cm` が書けない論文とは

`.cm` に落とせない論文は、

> **モデル構造が曖昧で、再現可能な定義を持たない**

可能性があり、単独で読むには留意が必要である。

---

## これは何の仕様か

Common Model IR は

> **科学・物理・統計・機械学習における
> 「モデル層」の共通仕様**

である。

実装は各言語の仕事であり、
Common Model IR は **意味と構造を書く**。

---

## 記述例

ここからはいくつかの例を挙げる。

```yaml
model LinearRegression:
  # 古典的な線形回帰モデル
  # y = x·w + b + noise

index:
  i : sample      # データ点の添字
  d : feature     # 特徴量の次元

given:
  # 観測された入力と出力（学習データ）
  x@train[i,d] : real
  y@train[i]   : real

unknown:
  # 推定したいパラメータ
  w[d]          # 重みベクトル
  b             # バイアス
  sigma         # 観測ノイズの標準偏差

  # 予測したい入力
  x@pred[i,d]  : real

define:
  # 線形予測値（決定論的）
  mu[i] = sum_d x@train[i,d] * w[d] + b

rules:
  # ノイズ分散は正
  sigma > 0

  # 各データ点について尤度を課す
  for i:
    y@train[i] ~ Normal(mu[i], sigma)

want:
  # データに合わせてパラメータを推定
  fit w, b, sigma
  predict y@pred
```

```yaml
model SIR:
  # 感染症の基本的な区画モデル
  # S: susceptible, I: infected, R: recovered

index:
  t : time   # 時間

given:
  # 総人口（定数）
  N     : real
  beta  : real   # 感染率
  gamma : real   # 回復率

unknown:
  # 各時刻の状態変数
  S[t]
  I[t]
  R[t]

rules:
  # 人口保存則
  for t:
    S[t] + I[t] + R[t] == N

  # 微分方程式（力学法則）
  for t:
    ∂S[t]/∂t == -beta * S[t] * I[t] / N
    ∂I[t]/∂t ==  beta * S[t] * I[t] / N - gamma * I[t]
    ∂R[t]/∂t ==  gamma * I[t]

want:
  # 初期条件が与えられたときの時間発展を求めたい
  predict S, I, R
```

```yaml
model QuadraticOptimization:
  # 二次目的関数の制約付き最小化問題

index:
  i : variable

given:
  # 二次形式の係数
  Q[i,i] : real
  c[i]   : real

unknown:
  # 最適化したい変数
  x[i]

define:
  # 目的関数（スカラー）
  f = sum_i sum_j x[i] * Q[i,j] * x[j] + sum_i c[i] * x[i]

rules:
  # 制約条件
  for i:
    x[i] >= 0

want:
  # 目的関数を最小化
  optimize min f
```

```yaml
model TransformerSequenceModel:
  # 単純化した Transformer による系列予測
  # 論文のモデル構造を追えることを優先

index:
  t : time        # 系列の時刻
  d : feature     # 入力・出力次元
  h : hidden      # 隠れ次元
  l : layer       # 層番号

given:
  # 観測された系列（教師データ）
  X@train[t,d] : real

unknown:
  # 各層のパラメータ
  W_q[l,h,h]
  W_k[l,h,h]
  W_v[l,h,h]
  W_o[l,h,h]
  W_ff1[l,h,h]
  W_ff2[l,h,h]

  # 各時刻の隠れ状態
  Z[l,t,h]

  # 出力系列
  Y@pred[t,d]

define:
  # Attention の中間量
  Q[l,t,h] = sum_k Z[l-1,t,k] * W_q[l,k,h]
  K[l,t,h] = sum_k Z[l-1,t,k] * W_k[l,k,h]
  V[l,t,h] = sum_k Z[l-1,t,k] * W_v[l,k,h]

rules:
  # 各層・各時刻について Transformer の規則を課す
  for l, t:
    Z[l,t] == sum_s softmax(Q[l,t]·K[l,s]) * V[l,s]

  # 出力は最終層の線形写像
  for t:
    Y@pred[t] == sum_h Z[L,t,h] * W_o[L,h,d]

  # 教師信号との誤差モデル
  for t:
    X@train[t+1] ~ Normal(Y@pred[t], sigma)

want:
  # パラメータ学習と予測
  fit W_q, W_k, W_v, W_o, W_ff1, W_ff2
  predict Y@pred
```

```yaml
model PINN_Burgers_1D_ExpandedNN_Collocation
# ------------------------------------------------------------
# 1D Burgers 方程式を対象とした Physics-Informed Neural Network (PINN)
#
# 目的:
#   未知関数 u(x,t) を、Burgers 方程式・初期条件・境界条件を
#   できるだけ満たすように、ニューラルネットで近似する。
#
# 特徴:
#   - PDE は「厳密制約」ではなく、残差の二乗和を最小化する
#   - PDE 残差はコロケーション点集合で評価される
#   - NN はブラックボックスではなく、数式として完全展開
# ------------------------------------------------------------


index:
  k : data_point
  # 観測データ点（教師ありデータがある場合）

  n : collocation_point
  # PDE 残差を評価するコロケーション点

  l : layer
  # ニューラルネットの層インデックス（0 ... L-1 を想定）

  h : hidden_unit
  # 各層の隠れユニット

  i : input_dim
  # 入力次元（この例では x, t の2次元）


given:
  nu : real
  # Burgers 方程式の粘性係数（既知定数）

  # --- 教師ありデータ（存在すれば使う） ---
  x_data[k] : real
  t_data[k] : real
  u_data[k] : real
  # 一部の (x,t) における u の観測値
  # PINN では必須ではないが、使える場合が多い

  # --- コロケーション点 ---
  x_col[n]  : real
  t_col[n]  : real
  # PDE 残差を評価する点集合
  # これが事実上、最適化の「測度」を定める

  # --- NN 構造 ---
  L : integer
  # 隠れ層の数
  # 構造は given、学習対象ではない


unknown:
  # --- NN の学習パラメータ ---
  W[l,h,i]
  # 各層の重み行列
  # 入力次元 i → 隠れユニット h

  b[l,h]
  # 各層のバイアス

  W_out[h]
  b_out
  # 出力層の重みとバイアス


define:
  # ==========================================================
  # ニューラルネットの順伝播（数式展開）
  # ==========================================================

  z_in[0] = x
  z_in[1] = t
  # 入力ベクトル (x, t)

  a[0,h] = sum_j W[0,h,j] * z_in[j] + b[0,h]
  z[1,h] = tanh(a[0,h])
  # 第1層

  for l:
    a[l,h]   = sum_q W[l,h,q] * z[l,q] + b[l,h]
    z[l+1,h] = tanh(a[l,h])
  # 中間層（l = 1 ... L-1 を想定）
  # for は量化（∀l）であり、ループや順序ではない

  u(x,t) = sum_h W_out[h] * z[L,h] + b_out
  # NN が定義する近似解 u(x,t)

  # ==========================================================
  # 微分量（自動微分の方法はここでは問わない）
  # ==========================================================

  u_t(x,t)  = ∂u(x,t)/∂t
  u_x(x,t)  = ∂u(x,t)/∂x
  u_xx(x,t) = ∂^2 u(x,t)/∂x^2

  # ==========================================================
  # Burgers 方程式の残差
  # ==========================================================

  r(x,t) = u_t(x,t)
           + u(x,t) * u_x(x,t)
           - nu * u_xx(x,t)
  # PDE を満たしていれば r(x,t) = 0 になる


rules:
  # ==========================================================
  # 厳密に満たすべき制約
  # ==========================================================

  for k:
    u(x_data[k], t_data[k]) == u_data[k]
  # 観測データとの一致（あれば）

  for x:
    u(x, 0) == u0(x)
  # 初期条件

  for t:
    u(-1, t) == 0
    u( 1, t) == 0
  # 境界条件


want:
  # ==========================================================
  # 最適化目的
  # ==========================================================

  optimize min sum_n r(x_col[n], t_col[n])^2
  # コロケーション点集合における
  # PDE 残差の二乗和を最小化
```