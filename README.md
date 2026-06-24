# English Speaking Feedback

英語スピーキング練習用の自動フィードバックアプリです。

ユーザに英語の質問を提示し、マイクで録音した英語回答を `faster-whisper` で文字起こしします。その後、Ollama上のローカルLLMを使って、文字起こしされた英文に対するフィードバックを生成します。

このアプリは発音、イントネーション、流暢さなどの音響的特徴は評価しません。評価対象は、文字起こしされた英文テキストです。

## 機能

- レベルとトピックに応じた英語質問の提示
- 固定質問リストからのランダム出題
- OllamaによるLLM質問生成
- マイク録音または音声ファイルアップロード
- Whisperによる英語音声の文字起こし
- LLMによる回答内容、文法、語彙、自然さの評価
- 文法修正、自然な表現、フィードバック、模範回答の表示

## セットアップ

Pythonの仮想環境を作成します。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Ollamaを起動し、使用するモデルを取得します。

```bash
ollama pull qwen2.5:7b-instruct
ollama serve
```

別のモデルを使う場合は、環境変数で指定できます。

```bash
export OLLAMA_MODEL=llama3.1:8b
```

Whisperモデルも変更できます。

```bash
export WHISPER_MODEL=tiny
```

## 起動

```bash
source .venv/bin/activate
python app.py
```

ブラウザで表示されたGradioのURLを開いて使います。

デフォルトでは `8960` 以降の空きポートを自動で使います。ポートを固定したい場合は、次のように指定します。

```bash
export GRADIO_SERVER_PORT=8961
python app.py
```

## 使い方

1. `Level` と `Topic` を選ぶ。
2. `New Question` を押して質問を表示する。
3. マイクで英語回答を録音する。
4. `Evaluate` を押して文字起こしとフィードバックを表示する。

`Use LLM question generation` をオンにすると、固定質問ではなくOllamaのLLMで質問を生成します。

## 注意

- Ollamaが起動していない場合、質問生成や採点は失敗します。
- 初回のWhisper実行時はモデルの準備に時間がかかることがあります。
- 採点はLLMによる自動評価なので、同じ回答でも点数が少し変わることがあります。
