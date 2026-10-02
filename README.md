# flatline

超低遅延・高応答性を志向する Voice-to-Voice AI システム。ペルソナは冷淡・無感情で、極めて簡潔・単調に応答する「無感情な頭脳」を目標とする（将来的にペルソナを切り替えられる余地は残す）。

## 設計思想

- **一時的な推論スキャフォールドとしての Python**: 重いモデル推論（Bark-small による応答音声の事前生成など）は、実行時パイプラインの外側、ビルド時ツール (`python/experiments/build_response_assets.py`) に限定する。
- **コアは Zig**: 音声 I/O、VAD（発話区間検出）、状態遷移、プロセス間同期、そして応答再生そのものを Zig (`src/`) で実装する。
- **最終目標**: Python スキャフォールドを排除し、Apple Silicon 上でネイティブ動作する単一バイナリに統合する。EnCodec のエンコード/デコード自体は既に Zig (`src/encoder.zig`, `src/decoder.zig`, `src/model.zig` の SafeTensors ローダ経由) に移植済み。実行時パイプラインのマイク入力 (`python/pipeline/stream_mic_encoder.py`) は依然 Python だが、応答再生側 (`./player`, `src/player.zig`) は [zaudio](https://github.com/zig-gamedev/zaudio) によるネイティブ実装に置き換わり、Python/torch への実行時依存は無くなった。

## アーキテクチャ

現在組み上がっているパイプラインは、3 プロセスを UNIX パイプで接続した構成。各フレームは共通のバイナリヘッダ付きパケットでやり取りされる（詳細は [docs/architecture.md](docs/architecture.md) を参照）。

```
[マイク入力]
     │ sounddevice (24kHz, 40ms チャンク)
     ▼
python/pipeline/stream_mic_encoder.py
  - EnCodec には依存しない。生 PCM (int16) をそのまま送るだけ
  - RMS エネルギーで is_speech フラグを算出し、ヘッダの reserved に格納
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

40ms チャンク単位で EnCodec 推論をかけていた旧方式はチャンク境界で歪み・かすれ音が生じていたため、「発話が確定するまでは生 PCM のまま中継する」方式に変更済み（詳細は [docs/architecture.md](docs/architecture.md) §1.3, §5.1）。応答生成も、受け取った音声をそのまま読み上げるエコーバック → Bark-small によるリアルタイム合成 → ビルド時に事前生成した応答音声を即時再生する「超低遅延トークンルーター」、という順で簡略化してきたが（§4）、最終段の実行時プロセスは当初 Python (`stream_decoder.py`) だった。これを `src/player.zig` + [zaudio](https://github.com/zig-gamedev/zaudio) によるネイティブ実装に置き換え、パイプライン後半の Python プロセスを完全に廃止した。将来的には、文字起こし・自己回帰的な応答テキスト生成（カテゴリ選択ロジック）を組み込み、最終的には `stream_mic_encoder.py` 側も Zig 側に統合して単一バイナリ化する。

## ディレクトリ構成

```
.
├── build.zig / build.zig.zon   # Zig ビルド定義 (zig 0.16)
├── shell.nix / .envrc          # Nix 開発環境 (direnv)
├── docs/
│   ├── memo.md                 # 当初の企画仕様書（無感情化のコンセプト案）
│   └── architecture.md         # 通信プロトコル・VAD仕様・技術的課題
├── python/
│   ├── pipeline/                # 現行の実動パイプライン
│   │   ├── stream_mic_encoder.py   # マイク入力 → 生PCM送信 (is_speechフラグ付き) → stdout
│   │   └── stream_encoder.py       # (補助) stdin WAV風入力のストリームエンコード
│   └── experiments/             # 検証用の使い捨てスクリプト群
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

Python 側の依存関係（`sounddevice`, `numpy` 等）は別途 venv などで用意する。これが必要なのは `stream_mic_encoder.py`（マイク入力側）のみで、応答再生側はネイティブの `./player` に置き換わったため Python 不要。`torch`, `transformers`, `encodec` が必要なのはビルド時ツール `python/experiments/build_response_assets.py`（および他の `python/experiments/*` スクリプト）のみで、実行時パイプラインには一切関係しない。

`zaudio`（[zig-gamedev/zaudio](https://github.com/zig-gamedev/zaudio)、内部で miniaudio を使用）は Zig の依存関係として `build.zig.zon` に登録済みで、`zig build` が自動的にフェッチする。

### 2. EnCodec の重みを取得

```sh
nu scripts/download_weights.nu
# -> src/weights/encodec_24khz.safetensors に保存される
```

### 3. Zig バイナリのビルド

```sh
zig build-exe src/receiver.zig
zig build          # src/main.zig (CLI "hoge") と src/player.zig (応答再生) をビルド
                    # -> zig-out/bin/hoge, zig-out/bin/player
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

マイクで喋った音声を Zig 製 VAD (`receiver`) が生 PCM のまま発話区間として確定し、Zig 製の `player` がその発話確定をトリガーに、§4 で事前ビルドした固定・短文の冷淡な応答（例: "Acknowledged."）からランダムに1件選んで zaudio 経由で即座に再生する導通確認構成（ユーザーの発話内容そのものは応答に反映されない）。このパイプラインに Python プロセスは `stream_mic_encoder.py` の1つだけで、応答側に Python/torch は一切登場しない。

```sh
python/pipeline/stream_mic_encoder.py | ./receiver | ./zig-out/bin/player
```

## 現在の進捗とロードマップ

- **Phase 1〜6: 完了**
  - マイク入力 → UNIX パイプ通信 → Zig 側 VAD・生 PCM バッファリング → 発話確定をトリガーにした、事前ビルド済み応答アセットのネイティブ即時再生（Zig + zaudio）まで成立。
- **解消済みの課題**
  - 40ms 単位のチャンク分割による境界歪み・かすれ音: `stream_mic_encoder.py` が都度 EnCodec 推論をかけていたことが原因だったため、生 PCM バッファリング方式に移行して解消した。
  - エコーバック（オウム返し）からの卒業: 受け取った PCM を復元する代わりに、発話確定をトリガーとして固定の冷淡な応答を再生するようになった。
  - 実行時 Bark 推論（数秒かかる）の撤廃: `python/experiments/build_response_assets.py` でビルド時に一度だけ Bark-small を実行し、応答音声を `assets/responses/` にアセット化。これにより実行時に重い推論は一切発生しなくなった。
  - パイプライン後半の Python プロセスの撤廃: `python/pipeline/stream_decoder.py` を削除し、`src/player.zig`（[zaudio](https://github.com/zig-gamedev/zaudio) 使用）に置き換えた。起動時に `assets/responses/manifest.json` の `.wav` を全てロードして zaudio の `Sound`/`AudioBuffer` として保持し、発話確定時はその場から選んで `Sound.start()` するだけになり、`torch`/`transformers`/Bark はもちろん Python インタプリタ自体への実行時依存も無くなった。コミットから再生開始までのレイテンシは `player` が stderr に出力する。
- **次のマイルストーン**
  - 入力音声の文字起こし（ASR）と、聞いた内容に基づくカテゴリ選択（ルーティング）ロジック。現状は発話内容に関わらず固定フレーズからランダムに応答するのみ。
  - 残る Python プロセス `stream_mic_encoder.py`（マイク入力・EnCodec不使用）も Zig 側に統合し、最終的に単一バイナリ化する。

詳細な通信プロトコル仕様・VAD ステートマシン仕様・技術的課題については [docs/architecture.md](docs/architecture.md) を参照。企画段階の思想・アルゴリズム案は [docs/memo.md](docs/memo.md) を参照（一部は現行実装と異なる設計案を含む）。
