# Architecture

現行パイプライン（`python/pipeline/stream_mic_encoder.py | ./receiver | python/pipeline/stream_decoder.py`）の実装に基づく通信プロトコルと各コンポーネントの仕様。企画段階の思想・将来案は [memo.md](memo.md) を参照（本ドキュメントは実装済みの挙動を正とする）。

> **2026-10 改訂**: 40ms チャンク単位で EnCodec 推論を行う方式（チャンク境界の歪み・かすれ音の原因）を廃止し、パイプラインは「生 PCM をバッファリングし、確定した発話区間のみを一括で EnCodec に通す」方式へ移行した。`stream_mic_encoder.py` は EnCodec/torch に依存しなくなり、マイク入力と VAD 用の RMS 計算のみを行う。EnCodec のエンコード/デコードは `stream_decoder.py` が発話確定後に 1 回だけ実行する。

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

EnCodec (24kHz, 8 コードブック, 6kbps 相当, `set_target_bandwidth(6.0)`) への入出力は `stream_decoder.py` のみが担う。発話確定後に受け取った一続きの生 PCM に対して `model.encode` → `model.decode` を 1 回の連続推論として実行するため、以前のような 40ms ごとの再推論によるチャンク境界の不連続が発生しない。

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

## 3. 現在の技術的課題と次のアプローチ

### 3.1 チャンク境界の歪み・かすれ音（解消済み）

旧実装では `stream_mic_encoder.py` が 40ms (960 サンプル) ごとに独立して EnCodec へ推論をかけていた。EnCodec の SEANet エンコーダ/デコーダは因果的 (causal) な畳み込みで前後の文脈を一部参照するため、40ms 単位で推論セッションを区切るとチャンクの境界で音響特徴が不連続になり、再生時に歪み・かすれが生じていた。

**対応済み**: `stream_mic_encoder.py` は EnCodec 推論を行わず生 PCM を逐次送るだけに変更し、`receiver.zig` が発話区間ぶんの生 PCM をバッファリング、確定後に `stream_decoder.py` が一続きの PCM に対して 1 回だけ `encode` → `decode` を実行するようにした（§1.3）。これによりチャンクごとの再推論・境界不連続が発生しなくなった。ただし発話全体のエンコード/デコードが発話確定後にまとめて走るため、発話が長いほどそのぶん再生開始までの遅延（レイテンシ）が伸びるトレードオフがある。

### 3.2 思考層 (LLM) の未結合

現状のパイプラインは「マイク入力 → VAD/生PCMバッファリング → EnCodec encode/decode」のみで構成されており、ユーザーの発話をそのまま読み上げるエコーバック止まりである。応答を生成する自己回帰モデル（思考層）はまだ組み込まれていない。`docs/memo.md` に記載の「無感情化（Monotone Control）」や 1B〜2B クラスの軽量 LLM 統合は未着手のフェーズ 3 相当であり、次のマイルストーンとして残っている。

### 3.3 Python スキャフォールドの段階的排除

EnCodec のエンコード/デコード処理自体は `src/encoder.zig` / `src/decoder.zig` / `src/rvq.zig` に Zig 実装済みで、`hoge --encode` / `hoge --stream` から呼び出せる状態にある。一方、現行の実運用パイプライン (`python/pipeline/stream_mic_encoder.py`, `stream_decoder.py`) はマイク入出力 (`sounddevice`) 周りとストリーミング制御のために依然として Python に依存している。最終的にはこれらの入出力制御も Zig 側 (`src/main.zig` 系) に統合し、単一バイナリ化することを目指す。
