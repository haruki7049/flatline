# Architecture

現行パイプライン（`python/pipeline/stream_mic_encoder.py | ./receiver | python/pipeline/stream_decoder.py`）の実装に基づく通信プロトコルと各コンポーネントの仕様。企画段階の思想・将来案は [memo.md](memo.md) を参照（本ドキュメントは実装済みの挙動を正とする）。

> **2026-10 改訂 (1)**: 40ms チャンク単位で EnCodec 推論を行う方式（チャンク境界の歪み・かすれ音の原因）を廃止し、パイプラインは「生 PCM をバッファリングし、確定した発話区間のみを一括で EnCodec に通す」方式へ移行した。`stream_mic_encoder.py` は EnCodec/torch に依存しなくなり、マイク入力と VAD 用の RMS 計算のみを行う。
>
> **2026-10 改訂 (2)**: `stream_decoder.py` はエコーバック（受け取った PCM をそのまま encode → decode して再生する）を廃止した。発話確定は「応答を生成するトリガー」としてのみ扱われ、受信 PCM の内容そのものは破棄する。応答は `suno/bark-small` を低 temperature (0.2) で駆動し、定型の冷淡な短文から生成した EnCodec トークンを再生する（§4）。

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

Python 側は `struct.pack("<HBBI", magic, version, reserved, payload_len)` で同一レイアウトを生成・解釈する（`stream_mic_encoder.py`, `stream_decoder.py` 共通）。

| フィールド | サイズ | 値 |
|---|---|---|
| `magic` | u16 | `0xAA55` 固定。不一致は即エラー終了 |
| `version` | u8 | `1` 固定 |
| `reserved` | u8 | `is_speech` フラグ。`stream_mic_encoder.py` → `receiver` 方向は RMS エネルギーがしきい値 `0.015` を超えれば `1`、それ以外は `0`。`receiver` → `stream_decoder.py` 方向は常に `1`（確定済み発話のみを送るため） |
| `payload_len` | u32 | ペイロードのバイト数 |

`receiver.zig` はこの `reserved`（`is_speech`）フラグだけで無音/発話を判定する（後述）。旧実装にあった「ペイロード先頭トークン値が無音コード `110` かどうか」による判定は廃止された。

### 1.2 ペイロード: 生 PCM

- サンプルフォーマット: `int16` リトルエンディアン, モノラル, 24,000 Hz
- EnCodec などのコーデック処理は**この段階では行わない**。`stream_mic_encoder.py` はマイクからの `float32` サンプルを `int16` に変換して送るだけで、torch/EnCodec には依存しない。

`stream_mic_encoder.py` → `receiver` 方向は **40ms チャンク** (`CHUNK_SAMPLES = 960` サンプル @ 24kHz) を 1 パケットとして逐次送信する。ペイロードは常に `960 samples × 2 bytes = 1920 bytes`。

`receiver` → `stream_decoder.py` 方向のペイロードは、VAD が確定した発話区間ぶんの生 PCM サンプル列をまとめて 1 パケットで送るため、長さはその発話の継続時間に応じて可変（`SAMPLES_PER_FRAME (960) × valid_frames`）。

### 1.3 EnCodec 推論のタイミング

`receiver` から届く「発話確定」パケットは、現在は**応答生成のトリガーとしてのみ**使われる。`stream_decoder.py` はペイロード（ユーザーの発話内容そのもの）を読み捨て、代わりに定型の冷淡な短文から Bark-small で生成した EnCodec トークンを `codec.decode` で PCM に復元して再生する（§4）。EnCodec (24kHz, 8 コードブック, 6kbps 相当, `set_target_bandwidth(6.0)`) はこの最終デコード段にのみ使われ、エンコードは行わない。

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

## 4. 冷淡な思考層プロトタイプ (`stream_decoder.py`)

応答生成はまだ「聞いた内容を理解して返す」段階にはなく、**発話確定を単なるトリガーとして固定・短文応答を返す**プロトタイプ。

### 4.1 構成

- モデル: `suno/bark-small`（`transformers.BarkModel` / `AutoProcessor`）。起動時に一度だけロードし、デバイス（CUDA/MPS/CPU）に常駐させる。
- 最終デコード: `BarkModel.generate()` を直接呼び出す高レベル API を使う。Semantic → Coarse → Fine の 3 段階に加え、Fine 段の EnCodec 8-stage トークンから PCM への変換（内部で `self.codec_model.decode` 相当）までを 1 回の呼び出し内で完結させる。§1.3 で使っていた `encodec` パッケージの `EncodecModel` は、このプロセスではもう使用しない。
- 応答テキスト: 以下の固定フレーズからランダムに 1 つを選択する（`RESPONSE_PHRASES`）。

  ```
  Acknowledged.
  Signal received. Stand by.
  Input logged.
  Processing complete.
  State confirmed.
  Noted.
  ```

### 4.2 生成ロジック（Semantic → Coarse → Fine、`bark_model.generate()` に集約）

旧実装では `python/experiments/generate_speech_lm.py` を踏襲し、`model.semantic.generate` → `model.coarse_acoustics.generate` → `model.fine_acoustics.generate` を手動で個別に呼び出していた。この方式は各段の `GenerationConfig` を自前で組み立てる必要がある一方、ステージ間の EOS 検出・アテンションマスクの引き継ぎが正しく働かず、単語が終わっても最大長（`SEMANTIC_MAX_NEW_TOKENS`）まで破裂音が反復生成される不具合（意味不明な喃語/「ダンダンダン」ループ）があった。

**現行実装**: `generate_flat_response_audio` は `bark_model.generate(**inputs, **GENERATE_KWARGS)` を 1 回呼ぶだけで、Semantic → Coarse → Fine の 3 段階と EnCodec デコードまでを `transformers` 側に任せる。`GENERATE_KWARGS` は各段の設定を `semantic_`/`coarse_`/`fine_` を接頭辞に持つキーワード引数として渡す、`transformers` が想定する公式な上書き方法。

```python
GENERATE_KWARGS = dict(
    semantic_temperature=FLAT_TEMPERATURE,
    coarse_temperature=FLAT_TEMPERATURE,
    fine_temperature=FLAT_TEMPERATURE,
    semantic_max_new_tokens=SEMANTIC_MAX_NEW_TOKENS,
)
```

- `voice_preset = "v2/en_speaker_6"` を `bark_processor` に渡して話者埋め込みを固定し、発話ごとに声質が暴れないようにする。
- 手動で `BarkSemanticGenerationConfig` 等を組み立てる構成は廃止したため、`max_length`/`max_new_tokens` の競合や `generation_config` と重複引数を同時に渡すことに起因する `transformers` の警告は発生しない。

> **temperature=0.2 は危険域**: 当初 `FLAT_TEMPERATURE = 0.2` としていたが、極端に低い temperature は Bark の自己回帰的な coarse/fine acoustics 生成でモード崩壊（同一の音響トークンが毎ステップ反復選択される）を引き起こし、デコード結果が人の声ではなく単一周波数の発振音（ハウリング/ピー音）になる不具合があった。`0.7` に引き上げることでサンプリングの多様性を確保し、この崩壊を回避している。

**生成長の上限**: Bark-small の Semantic 段はデフォルトで `max_new_tokens=768`（Semantic レート ~49Hz 換算で 10 秒超）まで生成し得るが、固定応答フレーズは 1〜5 単語の短文しかない。`SEMANTIC_MAX_NEW_TOKENS = 96` でこれを明示的に絞り込み、Semantic 系列長を基準に決まる Coarse/Fine の生成長も連動して短縮する。

**末尾無音のトリム**: `bark_model.generate()` が返す PCM に対し、`trim_trailing_silence`（20ms フレーム単位の RMS がしきい値 `0.01` を下回る末尾を切り捨てる簡易処理）を適用してから再生する。

### 4.3 処理フロー

1. `receiver` から「発話確定」パケットを受信する（ペイロードは読み捨てる）。
2. `RESPONSE_PHRASES` からランダムに 1 文を選び、上記の 3 段階生成で EnCodec トークンを得る。
3. `codec.decode` で PCM に変換し、再生開始時に `[flatline-decoder] Playing response (...)...` を stderr に出力したうえで `sounddevice` で再生する。

### 4.4 既知の制約

- 入力音声の文字起こしも、聞いた内容に基づくテキスト生成も行っていない。応答は発話内容に関わらず固定候補からのランダム選択であり、「オウム返し」から「固定文の読み上げ」になっただけで、対話としての思考層はまだ存在しない（§5.2 のロードマップ）。
- Bark-small の 3 段階生成（Semantic → Coarse → Fine）は数百 ms〜数秒オーダーの推論コストがあり、発話確定から応答再生開始までの遅延（レイテンシ）は旧来のエコーバックより大きい（`SEMANTIC_MAX_NEW_TOKENS` の調整で短縮済みだが、ゼロにはならない）。

## 5. 現在の技術的課題と次のアプローチ

### 5.1 チャンク境界の歪み・かすれ音（解消済み）

旧実装では `stream_mic_encoder.py` が 40ms (960 サンプル) ごとに独立して EnCodec へ推論をかけていた。EnCodec の SEANet エンコーダ/デコーダは因果的 (causal) な畳み込みで前後の文脈を一部参照するため、40ms 単位で推論セッションを区切るとチャンクの境界で音響特徴が不連続になり、再生時に歪み・かすれが生じていた。

**対応済み**: `stream_mic_encoder.py` は EnCodec 推論を行わず生 PCM を逐次送るだけに変更し、`receiver.zig` が発話区間ぶんの生 PCM をバッファリングするようにした。この改修自体はエコーバック時代に行ったものだが、`stream_decoder.py` がエコーバックから応答生成プロトタイプ（§4）に置き換わった現在も、VAD が「発話確定」を1つのまとまった単位で検出する役割は変わらず活きている。40ms ごとの再推論・境界不連続という問題そのものは、エコーバック廃止とは独立に解消済み。

### 5.2 思考層 (LLM) の未結合（部分的に着手）

§4 で「発話確定 → 固定フレーズ群からのランダム選択 → Bark-small による音声合成」という応答生成プロトタイプを導入したが、これは疎通確認目的の仮実装であり、まだ対話としての思考層ではない。入力音声の文字起こし（ASR）も、聞いた内容に基づくテキスト生成（自己回帰モデルによる応答テキスト生成）も行っていない。`docs/memo.md` に記載の「無感情化（Monotone Control）」や 1B〜2B クラスの軽量 LLM 統合は未着手のフェーズ 3 相当であり、次のマイルストーンとして残っている。

### 5.3 Python スキャフォールドの段階的排除

EnCodec のエンコード/デコード処理自体は `src/encoder.zig` / `src/decoder.zig` / `src/rvq.zig` に Zig 実装済みで、`hoge --encode` / `hoge --stream` から呼び出せる状態にある。一方、現行の実運用パイプライン (`python/pipeline/stream_mic_encoder.py`, `stream_decoder.py`) はマイク入出力 (`sounddevice`) 周りとストリーミング制御のために依然として Python に依存している。最終的にはこれらの入出力制御も Zig 側 (`src/main.zig` 系) に統合し、単一バイナリ化することを目指す。
