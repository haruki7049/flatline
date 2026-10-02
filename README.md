# flatline

超低遅延・高応答性を志向する Voice-to-Voice AI システム。ペルソナは冷淡・無感情で、極めて簡潔・単調に応答する「無感情な頭脳」を目標とする（将来的にペルソナを切り替えられる余地は残す）。

## 設計思想

- **一時的な推論スキャフォールドとしての Python**: 重いモデル推論（Bark-small による応答音声の事前生成など）は、実行時パイプラインの外側、ビルド時ツール (`python/experiments/build_response_assets.py`) に限定する。
- **コアは Zig**: 音声 I/O（マイク入力・スピーカー再生）、VAD（発話区間検出）、状態遷移、プロセス間同期を全て Zig (`src/`) で実装する。
- **最終目標**: Python スキャフォールドを排除し、Apple Silicon 上でネイティブ動作する単一バイナリに統合する。EnCodec のエンコード/デコード自体は既に Zig (`src/encoder.zig`, `src/decoder.zig`, `src/model.zig` の SafeTensors ローダ経由) に移植済み。マイク入力 (`src/capture.zig`) と応答再生 (`src/player.zig`) は [zaudio](https://github.com/zig-gamedev/zaudio) によるネイティブ実装で、実行時パイプライン (`capture | receiver | player`) に Python プロセスは一つも存在しない。Python が必要なのはビルド時ツール (`python/experiments/*`) のみ。

## アーキテクチャ

現在組み上がっているパイプラインは、3 つの **Zig ネイティブバイナリ**を UNIX パイプで接続した構成。Python プロセスは実行時に一つも存在しない。各フレームは共通のバイナリヘッダ付きパケットでやり取りされる（詳細は [docs/architecture.md](docs/architecture.md) を参照）。

```
[マイク入力]
     │ zaudio capture device (24kHz, mono, 16-bit)
     ▼
./capture (src/capture.zig, zig build) — ネイティブ実装、Python なし
  - zaudio のキャプチャデバイスから直接取得。40ms (960サンプル) 単位でバッファリング
  - 各チャンクの RMS エネルギーで is_speech フラグを算出し、ヘッダの reserved に格納
  - 0xAA55 ヘッダ + 生 PCM(int16) を stdout へ書き込み
     │ stdout → stdin (パイプ)
     ▼
./receiver (src/receiver.zig, zig build-exe)
  - VAD ステートマシン (Idle / Listening)。判定はヘッダの is_speech フラグで行う
  - 無音/ノイズのフレームを捨て、確定した発話区間の生 PCM だけを
    一括りにまとめて次段へ転送
     │ stdout → stdin (パイプ)
     ▼
./player (src/player.zig, zig build) — ネイティブ実装、Python なし
  - 受け取った生 PCM の中身は使わず、「発話確定」を応答トリガーとしてのみ
    扱う（エコーバックは廃止）
  - 起動時に assets/responses/manifest.json 記載の .wav を全て読み込み、
    zaudio (miniaudio) の AudioBuffer/Sound としてメモリに保持
  - 確定時にランダムに1件選んで zaudio 経由でそのまま再生。
    コミットから再生開始(Sound.start())までのレイテンシを計測して stderr に出力
```

40ms チャンク単位で EnCodec 推論をかけていた旧方式はチャンク境界で歪み・かすれ音が生じていたため、「発話が確定するまでは生 PCM のまま中継する」方式に変更済み（詳細は [docs/architecture.md](docs/architecture.md) §1.3, §5.1）。応答生成も、受け取った音声をそのまま読み上げるエコーバック → Bark-small によるリアルタイム合成 → ビルド時に事前生成した応答音声を即時再生する「超低遅延トークンルーター」、という順で簡略化してきた（§4）。パイプライン後半（`stream_decoder.py`）はまず `src/player.zig` + zaudio に置き換わり、続いて前半のマイク入力（`stream_mic_encoder.py`）も `src/capture.zig` + zaudio のネイティブ実装に置き換わったことで、実行時パイプラインから Python が完全に姿を消した（§4.2, §5.3）。将来的には、文字起こし・自己回帰的な応答テキスト生成（カテゴリ選択ロジック）を組み込み、最終的には単一バイナリへ統合することを目指す。

## ディレクトリ構成

```
.
├── build.zig / build.zig.zon   # Zig ビルド定義 (zig 0.16)
├── shell.nix / .envrc          # Nix 開発環境 (direnv)
├── docs/
│   ├── memo.md                 # 当初の企画仕様書（無感情化のコンセプト案）
│   └── architecture.md         # 通信プロトコル・VAD仕様・技術的課題
├── python/
│   ├── pipeline/                # 実行時パイプラインには含まれない補助スクリプトのみ
│   │   └── stream_encoder.py       # (補助) stdin WAV風入力のストリームエンコード
│   └── experiments/             # 検証用の使い捨てスクリプト群 / ビルド時ツール
│       ├── encode_speech.py / generate_tokens.py / export_bark_tokens.py 等
│       └── build_response_assets.py  # 定型応答の EnCodec トークンを事前生成し assets/responses/ へ出力
├── assets/
│   └── responses/               # build_response_assets.py が生成する応答アセット
│       ├── <category>_<id>.bin     # EnCodec RVQ トークン [T, 8] uint16
│       ├── <category>_<id>.wav     # 検証用 24kHz モノラル PCM
│       └── manifest.json           # カテゴリ・テキスト・フレーム数・ファイルパスの対応表
├── src/
│   ├── main.zig                 # CLI エントリポイント (`hoge`): encode/decode/stream サブコマンド
│   ├── receiver.zig             # VAD ステートマシン。パイプ間の発話区間検出・中継担当
│   ├── capture.zig              # マイクキャプチャ + RMS判定 (zaudio)。stream_mic_encoder.py の後継
│   ├── player.zig               # 応答アセットの即時再生 (zaudio)。stream_decoder.py の後継
│   ├── model.zig                # SafeTensors (.safetensors) のパース・mmap ビュー
│   ├── encoder.zig / decoder.zig # EnCodec 24kHz の SEANet エンコーダ/デコーダ (Zig 実装)
│   ├── rvq.zig                   # 残差ベクトル量子化 (RVQ) のフレーム形式・エンコード/デコード
│   ├── seanet.zig                # SEANet 用レイヤーローダ (Conv1d/ConvTranspose1d/LSTM/ResNet)
│   ├── nn.zig, nn/               # Conv1d, ConvTranspose1d, LSTM, ResNet, 活性化関数の実装
│   └── wav.zig                  # モノラル WAV 読み書き
└── scripts/
    ├── download_weights.nu       # HuggingFace から EnCodec 24kHz の重みを取得 (固定リビジョン)
    └── test-gpu.sh               # ROCm/HIP 環境での GPU テンソル確認 (Linux 専用)
```

## 動かし方

### 1. 開発環境

Nix + direnv を使用する（`shell.nix` が zig 0.16 / zls / python3.11 等を用意する）。

```sh
direnv allow
```

実行時パイプライン (`capture | receiver | player`) は全て Zig ネイティブバイナリで、Python は一切不要。`torch`, `transformers`, `encodec` などの Python 依存は、ビルド時ツール `python/experiments/build_response_assets.py`（および他の `python/experiments/*` スクリプト）を動かすときにだけ venv 等で用意すればよい。

`zaudio`（[zig-gamedev/zaudio](https://github.com/zig-gamedev/zaudio)、内部で miniaudio を使用）は Zig の依存関係として `build.zig.zon` に登録済みで、`zig build` が自動的にフェッチする。

### 2. EnCodec の重みを取得

```sh
nu scripts/download_weights.nu
# -> src/weights/encodec_24khz.safetensors に保存される
```

### 3. Zig バイナリのビルド

```sh
zig build-exe src/receiver.zig
zig build          # src/main.zig (CLI "hoge"), src/capture.zig, src/player.zig をビルド
                    # -> zig-out/bin/{hoge,capture,player}
```

初回の `zig build` は `zaudio` パッケージ（と、そのさらに依存する `system_sdk`）をネットワークから取得する。macOS では `zaudio`/`miniaudio` のビルドに CoreAudio 等のフレームワークが必要。Nix 環境でこれらのフレームワーク検索パスが自動解決できない場合に備え、`build.zig` は `xcrun --show-sdk-path` の結果をフォールバックのフレームワーク検索パスとして明示的に追加している。

`hoge` の主なサブコマンド（`zig build-exe`/`zig build` 後、`-h` でも確認可能）:

```
hoge                           # tokens.bin (省略時は空) を output.wav にデコード
hoge --encode input.wav        # input.wav (24kHz) を tokens.bin にエンコード
hoge --stream                  # stdin の RVQ トークンを読み、stdout に raw f32 PCM を書く
```

### 4. 応答アセットの事前ビルド（初回のみ）

`player` は実行時に Bark を呼ばず、事前生成済みの応答音声を再生するだけなので、先に一度だけビルドしておく。

```sh
python3 python/experiments/build_response_assets.py
# -> assets/responses/{ack,status,reject,complete}_{0,1}.{bin,wav} と manifest.json を生成
```

### 5. 応答プロトタイプ・パイプラインの実行

Zig 製のマイクキャプチャ (`capture`) がマイク入力を 40ms 単位で生 PCM 化して送り出し、Zig 製 VAD (`receiver`) がそれを発話区間として確定し、Zig 製の `player` がその発話確定をトリガーに、§4 で事前ビルドした固定・短文の冷淡な応答（例: "Acknowledged."）からランダムに1件選んで zaudio 経由で即座に再生する導通確認構成（ユーザーの発話内容そのものは応答に反映されない）。3プロセス全てが Zig ネイティブバイナリで、Python は一切登場しない。

```sh
./zig-out/bin/capture | ./receiver | ./zig-out/bin/player
```

## 現在の進捗とロードマップ

- **Phase 1〜7: 完了**
  - マイクキャプチャ → VAD・生 PCM バッファリング → 発話確定をトリガーにした事前ビルド済み応答アセットの即時再生まで、3段とも Zig ネイティブバイナリ（`capture | receiver | player`）のみで構成。実行時パイプラインに Python プロセスは存在しない。
- **解消済みの課題**
  - 40ms 単位のチャンク分割による境界歪み・かすれ音: 都度 EnCodec 推論をかけていたことが原因だったため、生 PCM バッファリング方式に移行して解消した。
  - エコーバック（オウム返し）からの卒業: 受け取った PCM を復元する代わりに、発話確定をトリガーとして固定の冷淡な応答を再生するようになった。
  - 実行時 Bark 推論（数秒かかる）の撤廃: `python/experiments/build_response_assets.py` でビルド時に一度だけ Bark-small を実行し、応答音声を `assets/responses/` にアセット化。これにより実行時に重い推論は一切発生しなくなった。
  - パイプライン全体の脱 Python: `python/pipeline/stream_decoder.py` を `src/player.zig` に、続いて `python/pipeline/stream_mic_encoder.py` を `src/capture.zig` に置き換えた（いずれも [zaudio](https://github.com/zig-gamedev/zaudio) 使用）。`player` は起動時に `assets/responses/manifest.json` の `.wav` を全てロードして zaudio の `Sound`/`AudioBuffer` として保持し、発話確定時はその場から選んで `Sound.start()` するだけ。`capture` はマイクの `Device`（capture モード）のコールバックで直接 RMS 判定とパケット送出を行う。`torch`/`transformers`/Bark はもちろん Python インタプリタ自体への実行時依存が完全に無くなった。コミットから再生開始までのレイテンシは `player` が stderr に出力する。
- **次のマイルストーン**
  - 入力音声の文字起こし（ASR）と、聞いた内容に基づくカテゴリ選択（ルーティング）ロジック。現状は発話内容に関わらず固定フレーズからランダムに応答するのみ。
  - `hoge` / `receiver` / `capture` / `player` という複数の Zig バイナリを単一バイナリに統合する。

詳細な通信プロトコル仕様・VAD ステートマシン仕様・技術的課題については [docs/architecture.md](docs/architecture.md) を参照。企画段階の思想・アルゴリズム案は [docs/memo.md](docs/memo.md) を参照（一部は現行実装と異なる設計案を含む）。
