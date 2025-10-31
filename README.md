
# README: 世界モデル(Dreamer系 RSSM)で学習する Multiwalker 協調制御（共有方策）

**目的**: PettingZoo Multiwalker を対象に、世界モデルあり(Dreamer系)と世界モデルなし(モデルフリーPPO)のAB比較を再現可能に実施する。
**前提**: Python 3.10.12。主要ライブラリはユーザー提示のバージョン（`gymnasium==0.29.1`, `pettingzoo==1.24.3`, `stable_baselines3==2.6.0`, `torch==2.6.0`, `moviepy==2.2.1`, `matplotlib==3.10.1`, `pandas==2.2.3`, `scikit-learn==1.7.2`, `SuperSuit==3.9.3` など）を使用。
**学習方針**:

* 世界モデルあり: Dreamer系 RSSM + 共有方策(全エージェント同一ネットをパラメータ共有)、オンライン学習のみ、PyTorch `.pt` で保存
* 世界モデルなし: モデルフリー共有PPO（比較用ベースライン）
* 可視化: ①動画(mp4/gif)、②latent t-SNE、③行動分布・価値推移
* 再現性: seed 42 固定
* OS想定: Linux系またはWSL2。Windowsネイティブでも可。FFmpegが必要

---

## 0. フォルダ構成

```
multiwalker_worldmodel/
├─ README.md  ← 本ファイル
├─ pyproject.toml or requirements.txt  ← 依存管理どちらか
├─ configs/
│   └─ default.yaml
├─ src/
│  ├─ envs/
│  │   └─ multiwalker_wrapped.py
│  ├─ models/
│  │   ├─ rssm.py
│  │   ├─ encoders.py
│  │   ├─ decoders.py
│  │   └─ heads.py
│  ├─ agents/
│  │   ├─ dreamer_shared.py
│  │   └─ ppo_shared.py
│  ├─ utils/
│  │   ├─ replay_buffer.py
│  │   ├─ video.py
│  │   ├─ tsne.py
│  │   └─ plotting.py
│  ├─ train_dreamer.py
│  ├─ train_baseline_ppo.py
│  ├─ evaluate.py
│  └─ rollout_record.py
└─ scripts/
   ├─ train_dreamer.sh
   ├─ train_baseline_ppo.sh
   ├─ eval_video.sh
   ├─ eval_latent_tsne.sh
   └─ eval_plots.sh
```

---

## 1. 依存関係と環境

ユーザー提示のバージョン群を尊重する。FFmpegのみ未導入なら入れる。

### 1.1 必須パッケージ

* 既に揃っているもの: `gymnasium`, `pettingzoo`, `stable_baselines3`, `torch`, `moviepy`, `matplotlib`, `pandas`, `scikit-learn`, `SuperSuit`, `imageio`, `imageio-ffmpeg` など
* OS依存: FFmpeg

  * Ubuntu/WSL2: `sudo apt-get update && sudo apt-get install -y ffmpeg`
  * Windows: `choco install ffmpeg` など

### 1.2 追加ラッパーと注意

* PettingZoo MPE の `sisl/multiwalker_v9` を使用
* SuperSuit で環境正規化とベクトル化
* 乱数は `torch`, `numpy`, `random`, `gymnasium` で seed 42 を設定

---

## 2. Multiwalker 環境の要点

* 環境: `pettingzoo.sisl.multiwalker_v9.env()`
* エージェント数: 3
* 報酬: 既定のチーム報酬を使用
* 通信: 無し（観測は各エージェントのローカル）
* ステップ長: 500
* 共有方策: 3体とも同一の Actor-Critic を使用し、学習も共有

### 2.1 ラップ方針

* 観測空間・行動空間をGymnasium互換のベクトル化APIにブリッジ
* `agent_id` を one-hot で観測に付与して識別性を担保（共有方策に必須）

---

## 3. 世界モデルあり: Dreamer系 RSSM 概要

**構成**:

* 観測エンコーダ: MLP
* RSSM: Deterministic GRU h_t と Stochastic z_t（カテゴリカルまたはガウス）
* デコーダ: 観測復元ヘッド、報酬予測ヘッド、終端予測ヘッド
* Actor-Critic: Latent上での方策π(a|h,z)と価値V(h,z)。学習はImagined rollout上で

**学習ループ**:

1. 実環境で収集 → リプレイに保存
2. バッチをサンプルして世界モデル損失で更新

   * 再構成損失 + KL正則化 + 報酬・終端予測損失
3. RSSMで潜在ロールアウトを生成
4. 潜在ロールアウト上でActor-Critic更新（λ-return, entropy, value bootstrap）
5. 方策で再び環境へ

**損失の擬似式**:

* 世界モデル:

  * `L_world = recon(x|h,z) + β * KL(q(z|h,enc(x)) || p(z|h)) + CE(done|h,z) + MSE(r|h,z)`
* Actor: `L_actor = - E_{imagined} [ Σ γ^t * r_t^λ ] + α * entropy(π)`
* Critic: `L_value = MSE(V(h,z), target_λ)`

---

## 4. ABテスト設計

* **A系**: Dreamer系RSSM + Shared Policy
* **B系**: モデルフリー共有PPO（SB3のPPOを拡張して共有方策でMultiwalkerに対応）
* 同一の環境シード、同一の学習ステップ数、同一の評価プロトコル
* 指標: エピソード総報酬、成功率、隊列崩壊率、落下回数
* 可視化: 動画、latent t-SNE、行動分布・価値推移
* 保存: `runs/{exp_name}/checkpoints/*.pt`, `runs/{exp_name}/videos/*.mp4`, `runs/{exp_name}/figs/*.png`, `runs/{exp_name}/latents/*.npz`

---

## 5. 設定ファイル例（`configs/default.yaml`）

```yaml
seed: 42
device: "cuda"         # "cpu" でも可
env:
  name: "multiwalker_v9"
  n_agents: 3
  episode_len: 500
  vector_envs: 8
  frame_stack: 1
  normalize_obs: true
  add_agent_id: true
world_model:
  rssm:
    deter_size: 200
    stoch_size: 32
    stoch_dist: "gauss"  # "categorical" も可
    gru_hidden: 200
    horizon: 15
  enc:
    hidden: [256, 256]
  dec:
    hidden: [256, 256]
  loss:
    beta_kl: 1.0
    recon_coef: 1.0
    reward_coef: 1.0
    done_coef: 1.0
actor_critic:
  actor_hidden: [256, 256]
  critic_hidden: [256, 256]
  entropy_coef: 0.001
  gamma: 0.99
  lambda: 0.95
train:
  steps: 2_000_000
  batch_size: 256
  seq_len: 50
  update_world_model_every: 1
  update_actor_every: 1
  init_random_steps: 10_000
  eval_interval: 50_000
  save_interval: 50_000
  log_interval: 5_000
eval:
  episodes: 10
  render: true
  save_video: true
output:
  dir: "runs/exp_dreamer_shared"
baseline:
  steps: 2_000_000
  output_dir: "runs/exp_ppo_shared"
```

---

## 6. Codexへの指示（ファイル生成）

### 6.1 環境ラッパー

**指示**:

```
Create file src/envs/multiwalker_wrapped.py with:
- A PettingZoo Multiwalker wrapper that exposes a Gymnasium-like vectorized API using SuperSuit.
- Support n_agents=3, episode_len=500, add_agent_id=True to append one-hot agent id to observation.
- Implement make_env(seed, vector_envs, normalize_obs, frame_stack).
- Provide reset(seed), step(action_dict or stacked actions), observe() returning stacked obs per agent, and a bridge to single shared policy (concatenate per-agent obs into batch).
- Ensure team reward handling and done truncation consistent with PettingZoo.
```

### 6.2 モデル群

**指示**:

```
Create file src/models/encoders.py with:
- MLPEncoder(in_dim, hidden: List[int], out_dim) -> latent features for RSSM input.

Create file src/models/decoders.py with:
- MLPDecoder(in_dim, hidden: List[int], out_dim) -> reconstruct observation.
- RewardHead(in_dim) -> scalar reward prediction.
- DoneHead(in_dim) -> Bernoulli or logits for terminal prediction.

Create file src/models/heads.py with:
- ActorHead(in_dim, act_dim) outputs action distribution params (DiagGaussian for continuous; Categorical for discrete).
- ValueHead(in_dim) outputs state value.

Create file src/models/rssm.py with:
- RSSM with deterministic GRU h_t and stochastic z_t.
- Functions: init_state(batch), observe(enc_obs_seq) -> posteriors, imagine(policy, horizon) -> latent rollout.
- KL loss calc supporting "free nats" optional.
```

### 6.3 エージェント

**指示**:

```
Create file src/agents/dreamer_shared.py with:
- DreamerSharedAgent holding: Encoder, RSSM, Decoder/Reward/Done heads, ActorHead, ValueHead.
- ReplayBuffer for sequences (import from utils).
- train_step() that:
  1) samples sequences (B, T, obs, act, rew, done)
  2) updates world model via L_world
  3) runs RSSM.imagine() with current actor to produce imagined trajectories of length horizon
  4) computes lambda-returns on imagined reward and updates actor/value
- act(obs_batch) returns actions for all 3 agents using shared policy.
- save/load checkpoints to .pt.
```

**ベースラインPPO**:

```
Create file src/agents/ppo_shared.py with:
- Shared-parameter PPO for multi-agent:
  - Single policy network conditioned on agent_id one-hot.
  - Collects trajectories across agents into one buffer.
  - Standard PPO update (clip, value loss, entropy).
- Compatible with Gymnasium-like vector env wrapper.
```

### 6.4 ユーティリティ

**指示**:

```
Create file src/utils/replay_buffer.py with:
- Sequence-capable replay for multi-agent shared-policy.
- API: add(transition per agent), sample(batch_size, seq_len), save/load.

Create file src/utils/video.py with:
- record_episode_video(env, policy, path_mp4 or gif) using imageio-ffmpeg/moviepy.

Create file src/utils/tsne.py with:
- collect_latents(agent, eval_episodes) to dump npz of {h, z} sequences.
- run_tsne(npz) -> 2D embedding, save png.

Create file src/utils/plotting.py with:
- plot_action_distribution(log_dir) over time by dimension.
- plot_value_curves(log_dir) for predicted V vs returns.
```

### 6.5 トレーニングと評価スクリプト

**指示**:

```
Create file src/train_dreamer.py with:
- Load configs/default.yaml.
- Set seeds.
- Build env via envs/multiwalker_wrapped.make_env.
- Instantiate DreamerSharedAgent on device.
- Online loop:
  - init_random_steps: take random actions to fill replay.
  - Main: interact -> store -> if step % update_every == 0: train_step()
  - Periodic eval: run N episodes, log metrics, save video if configured.
  - Save checkpoints at intervals.

Create file src/train_baseline_ppo.py with:
- Similar structure, but using PPOShared agent.
- Train for baseline.steps.

Create file src/evaluate.py with:
- Load checkpoint of Dreamer or PPO.
- Run episodes, compute metrics, optionally save video.
- Option to dump latents and call tsne/plotting utilities.

Create file src/rollout_record.py with:
- Simple tool to roll out a fixed policy and save transitions to npz for analysis.
```

---

## 7. 実装ガイド: 重要部分の擬似コード

### 7.1 RSSM 観測更新と想像ロールアウト

```text
# Observe (posterior)
h0, z0 = init_state(B)
for t in 1..T:
  e_t = enc(x_t)
  h_t = GRU(h_{t-1}, concat(z_{t-1}, e_t))
  z_t ~ q(z | h_t, e_t)             # posterior
  losses += recon(dec(h_t, z_t), x_t) + β * KL(q || p(z|h_t))
  r_hat = RewardHead(h_t, z_t)
  d_hat = DoneHead(h_t, z_t)

# Imagine (prior)
h0, z0 from last posterior
for τ in 1..H:
  a_τ ~ π(a | h_{τ-1}, z_{τ-1})
  h_τ = GRU(h_{τ-1}, concat(z_{τ-1}, a_τ))
  z_τ ~ p(z | h_τ)                   # prior
  r_τ = RewardHead(h_τ, z_τ)
  d_τ = DoneHead(h_τ, z_τ)
compute λ-returns on {r_τ, d_τ} → targets
update actor/value on imagined trajectory
```

### 7.2 共有方策の観測

```text
obs_per_agent = [obs_i]  # i=1..3
obs_i_aug = concat(obs_i, one_hot(agent_id=i))
batch = stack(obs_i_aug over i)
actions = actor(batch)
step env with per-agent actions
```

---

## 8. コマンド一覧

### 8.1 Dreamer 学習

```bash
# 例: CUDA使用
python -m src.train_dreamer --config configs/default.yaml
```

### 8.2 ベースラインPPO 学習

```bash
python -m src.train_baseline_ppo --config configs/default.yaml
```

### 8.3 評価と動画生成

```bash
# Dreamerチェックポイントで評価し動画保存
python -m src.evaluate \
  --config configs/default.yaml \
  --ckpt runs/exp_dreamer_shared/checkpoints/step_2000000.pt \
  --save_video --out runs/exp_dreamer_shared/videos/eval_ep.mp4

# PPOチェックポイント
python -m src.evaluate \
  --config configs/default.yaml \
  --ckpt runs/exp_ppo_shared/checkpoints/step_2000000.pt \
  --save_video --out runs/exp_ppo_shared/videos/eval_ep.mp4
```

### 8.4 latent t-SNE 可視化

```bash
# 潜在を収集
python -m src.evaluate \
  --config configs/default.yaml \
  --ckpt runs/exp_dreamer_shared/checkpoints/step_2000000.pt \
  --dump_latents --latents_out runs/exp_dreamer_shared/latents/latents_eval.npz

# t-SNE実行と保存
python -c "from src.utils.tsne import run_tsne; run_tsne('runs/exp_dreamer_shared/latents/latents_eval.npz')"
# 出力: runs/exp_dreamer_shared/figs/tsne.png
```

### 8.5 行動分布・価値推移

```bash
python -c "from src.utils.plotting import plot_action_distribution; plot_action_distribution('runs/exp_dreamer_shared')"
python -c "from src.utils.plotting import plot_value_curves; plot_value_curves('runs/exp_dreamer_shared')"
```

---

## 9. ABテスト実施手順

1. Dreamerを所定ステップまで学習し、`runs/exp_dreamer_shared/` に成果物を保存
2. 同条件でPPO共有方策を学習し、`runs/exp_ppo_shared/` に保存
3. 評価を各10エピソード以上で実施し、以下を比較

   * 平均エピソード報酬、標準誤差
   * 成功率
   * 隊列崩壊率、落下回数
   * 学習曲線の収束速度
   * 動画の質的比較
   * latent t-SNE のクラスタリングや遷移の滑らかさ
   * 行動分布の安定度、価値推定のバイアス
4. 結果を `runs/compare/` にCSVで集約し、グラフを保存
5. 結論を簡潔にまとめる

---

## 10. ロギングと保存

* すべて `.pt` で保存
* ログはCSVまたはTensorBoard。既存環境に合わせ、最低限CSVを標準とする
* `save_interval` ごとに `checkpoints/step_{N}.pt` を保存

---

## 11. 再現性

* `seed=42` を `random`, `numpy`, `torch`, `gymnasium` に設定
* 学習回数や評価回数を固定
* 乱数の漂いを抑えるため、評価時は`eval()`モード、探索ノイズ無効

---

## 12. 既知の落とし穴と対策

* Multiwalkerは初期難度が高い。`init_random_steps` を十分に確保
* 観測のスケールと正規化は重要。`normalize_obs: true` を推奨
* 共有方策では `agent_id` を必ず観測へ付与
* 動画生成はFFmpegが必要。出力FPSを大きくしすぎると重い
* RSSMの`horizon`が短すぎると長期予測が弱くなる。15〜25で調整
* KLのスケール`beta_kl`で表現力と正則化をバランス

---

## 13. スクリプト雛形

`scripts/train_dreamer.sh`

```bash
#!/usr/bin/env bash
set -e
python -m src.train_dreamer --config configs/default.yaml
```

`scripts/train_baseline_ppo.sh`

```bash
#!/usr/bin/env bash
set -e
python -m src.train_baseline_ppo --config configs/default.yaml
```

`scripts/eval_video.sh`

```bash
#!/usr/bin/env bash
set -e
python -m src.evaluate --config configs/default.yaml --ckpt "$1" --save_video --out "$2"
```

`scripts/eval_latent_tsne.sh`

```bash
#!/usr/bin/env bash
set -e
python -m src.evaluate --config configs/default.yaml --ckpt "$1" --dump_latents --latents_out "$2"
python -c "from src.utils.tsne import run_tsne; run_tsne('$2')"
```

`scripts/eval_plots.sh`

```bash
#!/usr/bin/env bash
set -e
python -c "from src.utils.plotting import plot_action_distribution; plot_action_distribution('$1')"
python -c "from src.utils.plotting import plot_value_curves; plot_value_curves('$1')"
```

---

## 14. 実行クイックスタート

```bash
# 1) Dreamer
bash scripts/train_dreamer.sh

# 2) PPOベースライン
bash scripts/train_baseline_ppo.sh

# 3) 評価動画
bash scripts/eval_video.sh runs/exp_dreamer_shared/checkpoints/step_2000000.pt runs/exp_dreamer_shared/videos/eval_ep.mp4

# 4) latent t-SNE
bash scripts/eval_latent_tsne.sh runs/exp_dreamer_shared/checkpoints/step_2000000.pt runs/exp_dreamer_shared/latents/latents_eval.npz

# 5) 各種プロット
bash scripts/eval_plots.sh runs/exp_dreamer_shared
```

---

## 15. 補足メモ

* 行動空間が連続か離散かは`multiwalker_v9`の設定に依存する。連続ならActorHeadは正規分布、離散ならCategorical
* t-SNEは`sklearn`を使用。対象は`[h_t, z_t]`連結ベクトル
* 学習安定化のため、観測エンコーダとRSSMの初期化に`orthogonal_/xavier_`を採用
* ロスのナナりやすさ対策として`grad clip`を導入

---