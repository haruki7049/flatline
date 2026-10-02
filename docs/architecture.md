# Architecture

現行パイプライン（`python/pipeline/stream_mic_encoder.py | ./receiver | ./player`）の実装に基づく通信プロトコルと各コンポーネントの仕様。企画段階の思想・将来案は [memo.md](memo.md) を参照（本ドキュメントは実装済みの挙動を正とする）。

> **2026-10 改訂 (1)**: 40ms チャンク単位で EnCodec 推論を行う方式（チャンク境界の歪み・かすれ音の原因）を廃止し、パイプラインは「生 PCM をバッファリングし、確定した発話区間のみを一括で EnCodec に通す」方式へ移行した。`stream_mic_encoder.py` は EnCodec/torch に依存しなくなり、マイク入力と VAD 用の RMS 計算のみを行う。
>
> **2026-10 改訂 (2)**: `stream_decoder.py`（当時）はエコーバック（受け取った PCM をそのまま encode → decode して再生する）を廃止した。発話確定は「応答を生成するトリガー」としてのみ扱われ、受信 PCM の内容そのものは破棄する。応答は `suno/bark-small` を低 temperature (0.2) で駆動し、定型の冷淡な短文から生成した EnCodec トークンを再生する（§4）。
>
> **2026-10 改訂 (3)**: 実行時の Bark 推論をビルド時ツール (`build_response_assets.py`) に切り出し、`stream_decoder.py` は事前ビルド済みアセットを即時再生するだけの「トークンルーター」になった（§4）。
>
> **2026-10 改訂 (4)**: その `stream_decoder.py` 自体を撤廃し、`src/player.zig`（[zaudio](https://github.com/zig-gamedev/zaudio) を使用するネイティブ実装）に置き換えた。パイプライン後半の Python プロセスはこれで無くなった（§4.2）。

## 1. プロセス間通信プロトコル

3 プロセスは標準入出力をつないだ UNIX パイプで通信する。各メッセージは「8 byte バイナリヘッダ + ペイロード」の 1 フレーム。

### 1.1 ヘッダ (8 bytes, リトルエンディアン)

Zig 側の定義 (`src/receiver.zig`):

```zig
const Header = extern struct {
    magic: u16,        // 0xAA55 固定
    version: u8,       // 1 固定
    reserved: u8,       // is_speech フラグ (0: 無音, 1: 有音)。送信側が設定する
    payload_len: u32,  // 後続ペイロードのバイト長
};
```

Python 側 (`stream_mic_encoder.py`) は `struct.pack("<HBBI", magic, version, reserved, payload_len)` で、Zig 側 (`receiver.zig`, `player.zig`) は上記の `extern struct Header` で、同一の 8 byte レイアウトを生成・解釈する。

| フィールド | サイズ | 値 |
|---|---|---|
| `magic` | u16 | `0xAA55` 固定。不一致は即エラー終了 |
| `version` | u8 | `1` 固定 |
| `reserved` | u8 | `is_speech` フラグ。`stream_mic_encoder.py` → `receiver` 方向は RMS エネルギーがしきい値 `0.015` を超えれば `1`、それ以外は `0`。`receiver` → `player` 方向は常に `1`（確定済み発話のみを送るため） |
| `payload_len` | u32 | ペイロードのバイト数 |

`receiver.zig` はこの `reserved`（`is_speech`）フラグだけで無音/発話を判定する（後述）。旧実装にあった「ペイロード先頭トークン値が無音コード `110` かどうか」による判定は廃止された。

### 1.2 ペイロード: 生 PCM

- サンプルフォーマット: `int16` リトルエンディアン, モノラル, 24,000 Hz
- EnCodec などのコーデック処理は**この段階では行わない**。`stream_mic_encoder.py` はマイクからの `float32` サンプルを `int16` に変換して送るだけで、torch/EnCodec には依存しない。

`stream_mic_encoder.py` → `receiver` 方向は **40ms チャンク** (`CHUNK_SAMPLES = 960` サンプル @ 24kHz) を 1 パケットとして逐次送信する。ペイロードは常に `960 samples × 2 bytes = 1920 bytes`。

`receiver` → `player` 方向のペイロードは、VAD が確定した発話区間ぶんの生 PCM サンプル列をまとめて 1 パケットで送るため、長さはその発話の継続時間に応じて可変（`SAMPLES_PER_FRAME (960) × valid_frames`）。

### 1.3 EnCodec 推論のタイミング

`receiver` から届く「発話確定」パケットは、現在は**応答生成のトリガーとしてのみ**使われる。`player`（および、その前身だった `stream_decoder.py`）はペイロード（ユーザーの発話内容そのもの）を読み捨て、代わりに事前ビルド済みの応答音声を再生する（§4）。実行時プロセスでは EnCodec の推論は一切発生しない。Bark-small と EnCodec によるトークン生成・デコードは `python/experiments/build_response_assets.py`（ビルド時ツール）でのみ行われる。

## 2. Zig 側 VAD ステートマシン (`src/receiver.zig`)

### 2.1 状態

```
idle ⇄ listening
```

- `idle`: 無音待機中。`sample_buffer` は空。
- `listening`: 発話中とみなし、受信した生 PCM フレームをすべて `sample_buffer` に蓄積する。

### 2.2 無音/発話判定

`receiver.zig` は各パケットのヘッダの `reserved`（`is_speech`）フラグだけを見る。

```zig
const is_silence = (header.reserved == 0);
```

判定そのものは送信側（`stream_mic_encoder.py`）の RMS エネルギーしきい値に委ねられており、`receiver` は判定結果に従ってバッファリングと状態遷移のみを行う。

### 2.3 遷移条件

| 定数 | 値 | 意味 |
|---|---|---|
| `SILENCE_THRESHOLD_FRAMES` | 12 | 40ms × 12 = 480ms 連続無音で `listening → idle` に戻る |
| `MIN_SPEECH_FRAMES` | 3 | 有効発話フレーム数がこれ未満ならノイズスパイクとして破棄 (120ms 未満) |

遷移ロジック:

1. `idle` で非無音フレームを受信 → `listening` へ遷移、バッファをクリアして当該フレームを格納、`[SPEECH STARTED]` をログ。
2. `listening` 中は受信フレームを逐次バッファへ追加。
   - 非無音フレームが来るたびに `silence_frames` を 0 にリセット。
   - 無音フレームが来るたびに `silence_frames` をインクリメント。
3. `silence_frames >= SILENCE_THRESHOLD_FRAMES`（480ms 連続無音）に達したら `listening → idle` に戻り、区間を確定する。
   - `valid_frames = speech_frame_count - silence_frames`（末尾の無音ぶんを除いた実発話フレーム数）が `MIN_SPEECH_FRAMES` 未満なら `[IGNORED NOISE]` として**転送せず**破棄。
   - それ以外は `sample_buffer` の先頭 `valid_frames × SAMPLES_PER_FRAME` サンプルぶん（＝末尾の無音を切り捨てた生 PCM）を 1 パケットにまとめ、0xAA55 ヘッダ（`reserved = 1`）を付けて stdout（次段のデコーダ）へ転送し、`[SPEECH COMMITTED]` をログ。
4. いずれの場合もバッファ・カウンタをリセットして `idle` に戻る。

## 4. 冷淡な思考層プロトタイプ: オフライン事前ビルド + ネイティブ実行時トークンルーター

応答生成はまだ「聞いた内容を理解して返す」段階にはなく、**発話確定を単なるトリガーとして固定・短文応答を返す**プロトタイプ。Bark-small による自己回帰生成（数百ms〜数秒かかる）は実行時から完全に排除し、**ビルド時に一度だけ**実行して結果をアセット化する構成に分離した。さらに実行時側（§4.2）は Python を使わず Zig + zaudio のネイティブ実装である。

### 4.1 オフラインビルド (`python/experiments/build_response_assets.py`)

実行コマンド:

```sh
python3 python/experiments/build_response_assets.py
```

- モデル: `suno/bark-small`（`transformers.BarkModel` / `AutoProcessor`）。このスクリプトの実行時にのみロードされ、実行時プロセス（`player`, 元 `stream_decoder.py`）には一切ロードされない。
- 固定フレーズをカテゴリ別に用意し（`ack`/`status`/`reject`/`complete`、各2文）、`bark_model.generate()` を 1 回呼ぶだけで Semantic → Coarse → Fine の 3 段階と EnCodec デコードまで完結させる。

  ```python
  bark_model.generate(
      **inputs,
      semantic_temperature=0.7,
      coarse_temperature=0.7,
      fine_temperature=0.7,
      semantic_max_new_tokens=96,
      min_eos_p=0.05,
  )
  ```

  - `voice_preset = "v2/en_speaker_6"` で話者を固定。
  - `temperature = 0.7`: 当初 `0.2` だったが、極端に低い temperature は coarse/fine acoustics でモード崩壊（同一トークンの反復選択）を起こし、人の声ではなく単一周波数の発振音（ハウリング/ピー音）になる不具合があったため引き上げた。
  - `semantic_max_new_tokens = 96`: Semantic 段のデフォルト `max_new_tokens=768`（10秒超）は短文には過大なため制限。`min_eos_p = 0.05` で EOS を早めに出しやすくし、短文に対して余分な喃語が続かないようにする。
  - トークン抽出は `bark_model.codec_decode` を一時的にフックし、`generate()` が最後に PCM へデコードする直前の EnCodec fine トークン `[1, 8, T]` をそのまま捕捉する（`export_bark_tokens.py` と同じ手法）。手動で `model.semantic.generate` 等を個別に呼ぶ旧方式は、ステージ間の EOS 検出・アテンションマスクの引き継ぎが正しく働かず、単語が終わっても最大長まで破裂音が反復生成される不具合（意味不明な喃語/「ダンダンダン」ループ）があったため廃止した。
- 無音トリムは EnCodec フレーム単位（320 サンプル = 1 フレーム, 75Hz）で行い、トリム後のトークン列を再デコードしたものを `.wav` として保存する（`.bin` を後で単体デコードしたときの音と一致させるため）。
- 出力: `assets/responses/<category>_<id>.bin`（EnCodec RVQ トークン `[T, 8] uint16`、`src/rvq.zig` の `Frame`/`framesFromBytes` と同一レイアウト）、`<category>_<id>.wav`（24kHz mono, 検証用）、`manifest.json`（カテゴリ・テキスト・フレーム数・ファイルパスの対応表）。

### 4.2 実行時トークンルーター (`src/player.zig`、ネイティブ実装)

当初この役割は Python の `stream_decoder.py` が担っていたが、パイプライン後半の Python プロセスを完全に撤廃するため `src/player.zig` に置き換えた。再生バックエンドには [zaudio](https://github.com/zig-gamedev/zaudio)（内部で miniaudio を使用、`build.zig.zon` に依存として追加）を使う。`player` は起動時に `torch`/`transformers`/Bark はおろか Python インタプリタ自体を一切ロードしない。

1. 起動時に `assets/responses/manifest.json`（`std.json.parseFromSlice` で構造体にパース）を読み込み、そこに列挙された各 `.wav` を `wav.readMonoFile`（`src/wav.zig`、既存の WAV パーサをそのまま再利用）でデコードする。
2. 各応答の PCM サンプル列を `zaudio.AudioBuffer`（format `.float32`, サンプルレートは manifest のものをそのまま設定）として保持し、`engine.createSoundFromDataSource(...)` で `zaudio.Sound` を 1 個ずつ事前に作っておく（`ResponseAsset.sound`）。コミットのたびに新規生成はしない。
3. `receiver` から「発話確定」パケットを受信したら、ペイロード（ユーザーの発話内容）は読み捨て、`std.Io.Timestamp.now(io, .awake)` でコミット時刻を記録する。
4. 事前生成済みの `Sound` からランダムに 1 件選び、`sound.seekToPcmFrame(0)` で先頭へ巻き戻してから `sound.start()` を呼ぶだけで再生が始まる（新規デコード・新規アロケーションなし）。
5. コミットからこの `start()` 呼び出しまでのレイテンシをミリ秒単位で計測し、`[flatline-player] ... Commit-to-playback latency: ...ms` として stderr（`std.debug.print`）に出力する。
6. `sound.isAtEnd()` をポーリング（`std.Io.sleep` で 5ms 間隔）して再生完了を待ってから次のパケットを処理する（§1.2 の「発話は1件ずつ、重ならない」前提を再生側でも維持）。

実行時に重い推論・ディスクI/O・Python インタプリタの介在が一切発生しないため、応答レイテンシは Zig 側の配列インデックスとポインタ操作のみ（実測で1ms未満）になる。

macOS でのビルド時の注意: `zaudio`/`miniaudio` は CoreAudio 等のフレームワークが必要で、通常は zaudio 側の `system_sdk`（遅延依存）経由でフレームワーク検索パスが解決される。この Nix 環境ではその解決が安定しなかったため、`build.zig` 側で `xcrun --show-sdk-path` の結果を明示的なフォールバック検索パスとして追加している。

### 4.3 既知の制約

- 入力音声の文字起こしも、聞いた内容に基づくテキスト生成も行っていない。応答は発話内容に関わらず固定候補からのランダム選択であり、「オウム返し」から「固定文の読み上げ」になっただけで、対話としての思考層はまだ存在しない（§5.2 のロードマップ）。
- 応答の多様性はビルド時にあらかじめ用意したフレーズ数（現状カテゴリごとに2文、計8文）に限られる。新しい応答を増やすには `build_response_assets.py` の `RESPONSES` を編集し、再ビルドする必要がある。
- `player` はプロセス終了まで全アセットの `Sound`/`AudioBuffer`/`Engine` を保持し続け、stdin が EOF になったとき（通常は `Ctrl-C` でパイプライン全体を止めたとき）にのみ明示的に `destroy()` して片付ける。

## 5. 現在の技術的課題と次のアプローチ

### 5.1 チャンク境界の歪み・かすれ音（解消済み）

旧実装では `stream_mic_encoder.py` が 40ms (960 サンプル) ごとに独立して EnCodec へ推論をかけていた。EnCodec の SEANet エンコーダ/デコーダは因果的 (causal) な畳み込みで前後の文脈を一部参照するため、40ms 単位で推論セッションを区切るとチャンクの境界で音響特徴が不連続になり、再生時に歪み・かすれが生じていた。

**対応済み**: `stream_mic_encoder.py` は EnCodec 推論を行わず生 PCM を逐次送るだけに変更し、`receiver.zig` が発話区間ぶんの生 PCM をバッファリングするようにした。この改修自体はエコーバック時代に行ったものだが、実行時の応答生成がエコーバック → Bark 直接合成 → 事前ビルド済みアセットの再生（§4、現在は `player.zig`）と変わった現在も、VAD が「発話確定」を1つのまとまった単位で検出する役割は変わらず活きている。40ms ごとの再推論・境界不連続という問題そのものは、その後の応答生成方式の変更とは独立に解消済み。

### 5.2 思考層 (LLM) の未結合（部分的に着手）

§4 で「発話確定 → 事前ビルド済み固定フレーズ群からのランダム選択 → プリビルド済み音声の即時再生」という応答生成プロトタイプを導入したが、これは疎通確認・低遅延化目的の仮実装であり、まだ対話としての思考層ではない。入力音声の文字起こし（ASR）も、聞いた内容に基づくテキスト生成（自己回帰モデルによる応答テキスト生成）も行っていない。`docs/memo.md` に記載の「無感情化（Monotone Control）」や 1B〜2B クラスの軽量 LLM 統合は未着手のフェーズ 3 相当であり、次のマイルストーンとして残っている（将来的には、ASR結果からカテゴリ `ack`/`status`/`reject`/`complete` 等をルーティングする、あるいはLLMが直接カテゴリを選ぶ形が想定される）。

### 5.3 Python スキャフォールドの段階的排除（進行中）

EnCodec のエンコード/デコード処理自体は `src/encoder.zig` / `src/decoder.zig` / `src/rvq.zig` に Zig 実装済みで、`hoge --encode` / `hoge --stream` から呼び出せる状態にある。応答再生側は `python/pipeline/stream_decoder.py` を削除し `src/player.zig`（§4.2）に置き換えたことで、パイプライン中で Python プロセスが必要なのは `stream_mic_encoder.py`（マイク入力・RMSベースのVADフラグ算出のみ、EnCodec/torch は不使用）1つだけになった。重い推論 (`suno/bark-small`) は `python/experiments/build_response_assets.py` というビルド時ツールに切り出されており、実行時パイプラインとは完全に分離されている。残る `stream_mic_encoder.py` を Zig 側 (`src/main.zig` 系、または `receiver.zig`/`player.zig` 同様の新規バイナリ) に統合すれば、パイプライン全体が単一バイナリ（または複数の Zig バイナリのみ）で完結する。
