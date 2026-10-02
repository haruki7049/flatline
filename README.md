# flatline

超低遅延・高応答性を志向する Voice-to-Voice AI システム。ペルソナは冷淡・無感情で、極めて簡潔・単調に応答する「無感情な頭脳」を目標とする（将来的にペルソナを切り替えられる余地は残す）。

## 設計思想

- **一時的な推論スキャフォールドとしての Python**: EnCodec のようなモデル推論は、現段階では PyTorch (MPS 加速) 上の Python スクリプトで済ませる。
- **コアは Zig**: 音声 I/O、VAD（発話区間検出）、状態遷移、プロセス間同期は Zig (`src/`) で実装する。
- **最終目標**: Python スキャフォールドを排除し、Apple Silicon 上でネイティブ動作する単一バイナリに統合する。EnCodec のエンコード/デコード自体は既に Zig (`src/encoder.zig`, `src/decoder.zig`, `src/model.zig` の SafeTensors ローダ経由) に移植済みで、現状 Python 側に残っているのは主にマイク入出力 (`sounddevice`) とストリーミング制御。

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
python/pipeline/stream_decoder.py
  - 受け取った生 PCM の中身は使わず、「発話確定」を応答生成のトリガー
    としてのみ扱う（エコーバックは廃止）
  - suno/bark-small (`bark_model.generate()`) を temperature 0.7 で駆動し、
    話者を固定(voice_preset)した上で定型の冷淡な短文から PCM を直接生成
  - sounddevice でスピーカーへ再生
```

40ms チャンク単位で EnCodec 推論をかけていた旧方式はチャンク境界で歪み・かすれ音が生じていたため、「発話が確定するまでは生 PCM のまま中継する」方式に変更済み（詳細は [docs/architecture.md](docs/architecture.md) §1.3, §5.1）。さらに `stream_decoder.py` は、受け取った音声をそのまま読み上げるエコーバックから、発話確定をトリガーに固定・短文の冷淡な応答を Bark-small で合成するプロトタイプへ置き換えた（§4）。将来的には、文字起こし・自己回帰的な応答テキスト生成を組み込み、最終的には思考層 (LLM) も Zig 側に統合して単一バイナリ化する。

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
│   │   ├── stream_decoder.py       # stdin (発話確定トリガー) → Bark-small応答合成 → スピーカー再生
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

Python 側の依存関係（`torch`, `sounddevice`, `transformers` 等）は別途 venv などで用意する。`stream_decoder.py` は初回実行時に Hugging Face Hub から `suno/bark-small` をダウンロードする（`encodec` パッケージはこのパイプラインではもう使用しないが、`python/experiments/*` の一部スクリプトは依然依存する）。

### 2. EnCodec の重みを取得

```sh
nu scripts/download_weights.nu
# -> src/weights/encodec_24khz.safetensors に保存される
```

### 3. Zig バイナリのビルド

```sh
zig build-exe src/receiver.zig
zig build          # src/main.zig (CLI "hoge") のビルド。zig-out/bin/hoge に生成
```

`hoge` の主なサブコマンド（`zig build-exe`/`zig build` 後、`-h` でも確認可能）:

```
hoge                           # tokens.bin (省略時は空) を output.wav にデコード
hoge --encode input.wav        # input.wav (24kHz) を tokens.bin にエンコード
hoge --stream                  # stdin の RVQ トークンを読み、stdout に raw f32 PCM を書く
```

### 4. 応答プロトタイプ・パイプラインの実行

マイクで喋った音声を Zig 製 VAD (`receiver`) が生 PCM のまま発話区間として確定し、`stream_decoder.py` がその発話確定をトリガーに、固定・短文の冷淡な応答（例: "Acknowledged."）を Bark-small で合成して読み上げる導通確認構成（ユーザーの発話内容そのものは応答に反映されない）。

```sh
python/pipeline/stream_mic_encoder.py | ./receiver | python/pipeline/stream_decoder.py
```

## 現在の進捗とロードマップ

- **Phase 1〜4: 完了**
  - マイク入力 → UNIX パイプ通信 → Zig 側 VAD・生 PCM バッファリング → 発話確定をトリガーにした Bark-small による固定・短文応答の合成・再生まで成立。
- **解消済みの課題**
  - 40ms 単位のチャンク分割による境界歪み・かすれ音: `stream_mic_encoder.py` が都度 EnCodec 推論をかけていたことが原因だったため、生 PCM バッファリング方式に移行して解消した。
  - エコーバック（オウム返し）からの卒業: `stream_decoder.py` は受け取った PCM を復元する代わりに、発話確定をトリガーとして Bark-small (temperature 0.2) で生成した定型の冷淡な応答を再生するようになった。トレードオフとして、Bark-small の 3 段階生成（Semantic → Coarse → Fine）のぶん応答再生開始までの遅延が増えた。
- **進行中**
  - 実行時の Bark 自己回帰生成（数秒かかる）を廃止し、「埋め込み駆動型トークンルーター」（カテゴリ別の定型応答から即時選択・再生）へ移行するための準備として、`python/experiments/build_response_assets.py` で応答音声の EnCodec トークンを事前ビルドする仕組みを追加した（`assets/responses/`）。`stream_decoder.py` 側をこのプリビルド済みトークンの再生に切り替える作業はまだ未着手。
- **次のマイルストーン**
  - 入力音声の文字起こし（ASR）と、聞いた内容に基づくカテゴリ選択（ルーティング）ロジック。現状は発話内容に関わらず固定フレーズからランダムに応答するのみ。
  - 最終的に Python スキャフォールド (`python/pipeline/*`) を Zig 実装に置き換え、単一バイナリ化する。

詳細な通信プロトコル仕様・VAD ステートマシン仕様・技術的課題については [docs/architecture.md](docs/architecture.md) を参照。企画段階の思想・アルゴリズム案は [docs/memo.md](docs/memo.md) を参照（一部は現行実装と異なる設計案を含む）。
