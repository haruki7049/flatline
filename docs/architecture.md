# Architecture

現行パイプライン（`python/pipeline/stream_mic_encoder.py | ./receiver | python/pipeline/stream_decoder.py`）の実装に基づく通信プロトコルと各コンポーネントの仕様。企画段階の思想・将来案は [memo.md](memo.md) を参照（本ドキュメントは実装済みの挙動を正とする）。

## 1. プロセス間通信プロトコル

3 プロセスは標準入出力をつないだ UNIX パイプで通信する。各メッセージは「8 byte バイナリヘッダ + ペイロード」の 1 フレーム。

### 1.1 ヘッダ (8 bytes, リトルエンディアン)

Zig 側の定義 (`src/receiver.zig`):

```zig
const Header = extern struct {
    magic: u16,        // 0xAA55 固定
    version: u8,       // 1 固定
    reserved: u8,       // stream_mic_encoder.py → receiver 向けは is_speech フラグ (0/1)
    payload_len: u32,  // 後続ペイロードのバイト長
};
```

Python 側は `struct.pack("<HBBI", magic, version, reserved, payload_len)` で同一レイアウトを生成・解釈する（`stream_mic_encoder.py`, `stream_decoder.py`, `python/experiments/*` 共通）。

| フィールド | サイズ | 値 |
|---|---|---|
| `magic` | u16 | `0xAA55` 固定。不一致は即エラー終了 |
| `version` | u8 | `1` 固定 |
| `reserved` | u8 | `stream_mic_encoder.py` が書き込む場合のみ意味を持つ `is_speech` フラグ（RMS エネルギーがしきい値 `0.015` を超えれば `1`、それ以外は `0`）。`receiver` → `stream_decoder.py` 方向では未使用 (`0`) |
| `payload_len` | u32 | ペイロードのバイト数 |

`receiver.zig` 自体は `reserved`（`is_speech` フラグ）を読まず、ペイロード先頭のトークン値で無音判定する（後述）。VAD 判定とヘッダの `is_speech` フラグは現状二重に存在しており、実際に使われているのは前者のみ。

### 1.2 ペイロード: EnCodec RVQ トークン

- コーデック: Meta EnCodec 24kHz (`EncodecModel.encodec_model_24khz()`, `set_target_bandwidth(6.0)`)
- サンプルレート: 24,000 Hz
- コードブック数: 8 (`num_stages = 8`, 6kbps 相当, `src/rvq.zig`)
- 1 トークン = `int16` (Python 側キャスト) / Zig 側の恒久フォーマットは `u16`（`src/rvq.zig` の `Frame = [8]u16`）
- レイアウト: `[timesteps, codebooks]` の順にフラット化（1 timestep ぶん = 8 トークン = 16 bytes）

`stream_mic_encoder.py` は **40ms チャンク** (`CHUNK_SAMPLES = 960` サンプル @ 24kHz) を 1 回の EnCodec 推論単位としており、EnCodec の畳み込みダウンサンプル比 (2×4×5×8 = 320) から 1 チャンクあたり `timesteps = 960 / 320 = 3`。つまり 1 パケットのペイロードは通常 `3 timesteps × 8 codebooks × 2 bytes = 48 bytes`。

`receiver` → `stream_decoder.py` 方向のペイロードは、VAD が確定した発話区間ぶんのトークンをまとめて 1 パケットで送るため、`timesteps` はその発話の長さに応じて可変。

## 2. Zig 側 VAD ステートマシン (`src/receiver.zig`)

### 2.1 状態

```
idle ⇄ listening
```

- `idle`: 無音待機中。トークンバッファは空。
- `listening`: 発話中とみなし、受信したトークンフレームをすべて `token_buffer` に蓄積する。

### 2.2 無音/発話判定

`receiver.zig` は各パケットの**ペイロード先頭トークン値**だけを見る（ヘッダの `reserved`/`is_speech` フラグは見ない）。

```zig
const SILENCE_TOKEN: i16 = 110;
const is_silence = (first_token == SILENCE_TOKEN);
```

EnCodec の出力上、無音区間の第 1 コードブックが安定して `110` に量子化される挙動を利用した簡易判定（`stream_mic_encoder.py` が送る RMS ベースの `is_speech` とは独立した、受信側だけで完結する判定ロジック）。

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
   - それ以外は `token_buffer` の先頭 `valid_frames` ぶん（＝末尾の無音を切り捨てたトークン列）を 1 パケットにまとめ、0xAA55 ヘッダを付けて stdout（次段のデコーダ）へ転送し、`[SPEECH COMMITTED]` をログ。
4. いずれの場合もバッファ・カウンタをリセットして `idle` に戻る。

## 3. 現在の技術的課題と次のアプローチ

### 3.1 チャンク境界の歪み・かすれ音

`stream_mic_encoder.py` は 40ms (960 サンプル) ごとに独立して EnCodec へ推論をかけている。EnCodec の SEANet エンコーダ/デコーダは因果的 (causal) な畳み込みで前後の文脈を一部参照するが、40ms 単位で推論セッションを区切ると、チャンクの境界で音響特徴が不連続になり、再生時に歪み・かすれが生じる。

**次回アプローチ（未実装）**: チャンクごとに都度エンコードするのではなく、生 PCM をリングバッファに蓄積してから、より長い／連続的な単位でエンコードする方式へ移行する。これにより境界の不連続を減らす。

### 3.2 思考層 (LLM) の未結合

現状のパイプラインは「マイク入力 → トークン化 → VAD → デコード」のみで構成されており、ユーザーの発話をそのまま読み上げるエコーバック止まりである。応答を生成する自己回帰モデル（思考層）はまだ組み込まれていない。`docs/memo.md` に記載の「無感情化（Monotone Control）」や 1B〜2B クラスの軽量 LLM 統合は未着手のフェーズ 3 相当であり、次のマイルストーンとして残っている。

### 3.3 Python スキャフォールドの段階的排除

EnCodec のエンコード/デコード処理自体は `src/encoder.zig` / `src/decoder.zig` / `src/rvq.zig` に Zig 実装済みで、`hoge --encode` / `hoge --stream` から呼び出せる状態にある。一方、現行の実運用パイプライン (`python/pipeline/stream_mic_encoder.py`, `stream_decoder.py`) はマイク入出力 (`sounddevice`) 周りとストリーミング制御のために依然として Python に依存している。最終的にはこれらの入出力制御も Zig 側 (`src/main.zig` 系) に統合し、単一バイナリ化することを目指す。
