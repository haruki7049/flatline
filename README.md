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
  - 受け取った一続きの生 PCM に対して EnCodec で encode → decode を
    1 回だけ実行（40ms ごとの再推論はしない）
  - sounddevice でスピーカーへ再生
```

40ms チャンク単位で EnCodec 推論をかけていた旧方式はチャンク境界で歪み・かすれ音が生じていたため、現在は上記の通り「発話が確定するまでは生 PCM のまま中継し、確定後に一括で EnCodec へ通す」方式に変更済み（詳細は [docs/architecture.md](docs/architecture.md) §1.3, §3.1）。将来的には、`stream_decoder.py` が担っている EnCodec 処理を `src/encoder.zig` / `src/decoder.zig` に置き換え、思考層 (LLM) も Zig 側に統合して単一バイナリ化する。

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
│   │   ├── stream_mic_encoder.py   # マイク入力 → EnCodec エンコード → stdout
│   │   ├── stream_decoder.py       # stdin → EnCodec デコード → スピーカー再生
│   │   └── stream_encoder.py       # (補助) stdin WAV風入力のストリームエンコード
│   └── experiments/             # 検証用の使い捨てスクリプト群
│       ├── encode_speech.py / generate_tokens.py / export_bark_tokens.py 等
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

Python 側の依存関係（`torch`, `sounddevice`, `encodec` 等）は別途 venv などで用意する。

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

### 4. エコーバック・パイプラインの実行

マイクで喋った音声が EnCodec でトークン化され、Zig 製 VAD (`receiver`) で発話区間を確定した上で、デコーダがそのまま読み上げ（エコーバック）する導通確認構成。

```sh
python/pipeline/stream_mic_encoder.py | ./receiver | python/pipeline/stream_decoder.py
```

## 現在の進捗とロードマップ

- **Phase 1〜4: 完了**
  - マイク入力 → UNIX パイプ通信 → Zig 側 VAD・生 PCM バッファリング → 確定発話の一括 EnCodec デコードによるエコーバックの導通確認まで成立。
- **解消済みの課題**
  - 40ms 単位のチャンク分割による境界歪み・かすれ音: `stream_mic_encoder.py` が都度 EnCodec 推論をかけていたことが原因だったため、生 PCM バッファリング方式（発話確定後に一括で EnCodec へ通す）に移行して解消した。トレードオフとして、発話が長いほど再生開始までの遅延が伸びる。
- **次のマイルストーン**
  - 思考層 (LLM) の結合。現状は録音内容をそのまま復元するエコーバックのみで、応答生成は未実装。
  - 最終的に Python スキャフォールド (`python/pipeline/*`) を Zig 実装に置き換え、単一バイナリ化する。

詳細な通信プロトコル仕様・VAD ステートマシン仕様・技術的課題については [docs/architecture.md](docs/architecture.md) を参照。企画段階の思想・アルゴリズム案は [docs/memo.md](docs/memo.md) を参照（一部は現行実装と異なる設計案を含む）。
