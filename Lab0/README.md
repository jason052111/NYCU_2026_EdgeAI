# Edge AI Lab0 學習紀錄

這份 Lab0 是我在 Edge AI 課程中的第一個實作練習。整個作業分成兩個主要部分：第一部分是使用 MobileNetV2 在 CIFAR-10 上做影像分類，第二部分是使用 Llama-3.2-1B-Instruct 搭配 `torch.compile` 比較語言模型推論速度。

## 作業目標

我這次主要完成了以下幾件事：

- 使用 PyTorch 訓練 MobileNetV2 影像分類模型。
- 將 CIFAR-10 圖片資料轉成 MobileNetV2 可以接受的輸入格式。
- 修改 pretrained MobileNetV2 的 classifier，讓它可以輸出 CIFAR-10 的 10 個類別。
- 設定 loss function、optimizer 和 learning rate scheduler。
- 訓練模型並讓 test accuracy 超過作業要求的 92.5%。
- 儲存訓練好的模型權重到 `model.pt`。
- 將 PyTorch 模型匯出成 ONNX 格式。
- 重新載入 `model.pt` 並確認模型 accuracy 正常。
- 使用 Llama-3.2-1B-Instruct 測試 `torch.compile` 前後的推論速度差異。

## 執行環境

我一開始使用 Google Colab 進行測試，後來因為 Colab GPU 配額限制，改成在自己的 WSL 環境中使用本機 GPU 執行。我的本機環境如下：

- OS: WSL on Windows
- GPU: NVIDIA GeForce RTX 4070
- Python: 3.11
- PyTorch: 2.11.0+cu128
- torchvision: 0.26.0+cu128

為了讓環境比較乾淨，我在 `Lab0` 資料夾底下建立了獨立的 conda environment，並另外整理了 requirements 檔案：

- `requirements-torch-cu128.txt`: 安裝 CUDA 12.8 版本的 PyTorch / torchvision / torchaudio。
- `requirements.txt`: 安裝 notebook 其餘需要的套件，例如 `torchprofile`、`onnx`、`onnxruntime`、`onnxscript`、`transformers`、`accelerate` 等。

## Part 1: MobileNetV2 + CIFAR-10

第一部分是做影像分類。CIFAR-10 是一個包含 10 個類別的圖片資料集，每張圖片原本大小是 `3 x 32 x 32`。MobileNetV2 則是一個輕量化 CNN，常用在手機或 edge device 這類資源有限的環境。

因為 MobileNetV2 的 pretrained model 通常使用 ImageNet 的輸入格式，所以我先對 CIFAR-10 圖片做以下前處理：

- `Resize((224, 224))`: 將圖片從 `32 x 32` 放大到 `224 x 224`。
- `ToTensor()`: 將圖片轉成 PyTorch tensor。
- `Normalize(...)`: 使用 ImageNet 的 mean/std 進行標準化。

我使用的訓練參數如下：

```python
NUM_CLASSES = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.01
NUM_EPOCH = 10
```

模型部分，我載入 torchvision 提供的 pretrained MobileNetV2，然後把最後的 classifier 從 ImageNet 的 1000 類改成 CIFAR-10 的 10 類：

```python
model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
```

訓練時我使用：

- Loss function: `nn.CrossEntropyLoss()`
- Optimizer: `SGD` with momentum and weight decay
- Scheduler: `CosineAnnealingLR`

其中 scheduler 的 `T_max` 我設定成：

```python
T_max = NUM_EPOCH * len(dataflow["train"])
```

因為我是在每個 batch 後呼叫一次 `scheduler.step()`，所以 `T_max` 使用總 batch 數，而不是只有 epoch 數。

最後訓練結果達到：

```text
final accuracy: 95.22235870361328
```

這已經高於作業要求的 `92.5%`。

## 儲存與載入模型

訓練完成後，我使用以下方式儲存模型權重：

```python
torch.save(model.state_dict(), "model.pt")
```

這裡存的是模型的 weights，不是整個模型架構。因此重新載入時，我需要先重新建立 MobileNetV2 架構，並把 classifier 改成 10 類，接著再載入 `model.pt`：

```python
loaded_model = mobilenet_v2(weights=None)
loaded_model.classifier[1] = nn.Linear(loaded_model.last_channel, NUM_CLASSES)
loaded_model.load_state_dict(torch.load("model.pt"))
loaded_model = loaded_model.cuda()
```

重新載入模型後，我再次執行 evaluation，得到：

```text
accuracy: 95.22235870361328
```

這代表儲存和載入模型的流程是正確的。

## ONNX Export

我也將訓練好的 MobileNetV2 匯出成 ONNX 格式：

```python
dummy_input = torch.zeros(1, 3, 224, 224).cuda()
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=18,
    input_names=["input"],
    output_names=["output"]
)
```

一開始我使用較舊的 opset version 時，新版 PyTorch 會出現版本轉換的 warning。後來我改成 `opset_version=18`，比較符合目前 PyTorch ONNX exporter 的行為。匯出後，我使用 Netron 開啟 `model.onnx`，確認模型結構可以被正常顯示。

## Part 2: Llama + torch.compile

第二部分是比較 LLM 在使用 `torch.compile` 前後的推論速度。我使用的模型是：

```text
meta-llama/Llama-3.2-1B-Instruct
```

這個模型需要 Hugging Face 帳號授權和 token 才能下載。我先在 Hugging Face 上申請 Llama 模型存取權，然後在 WSL terminal 中使用 `hf auth login` 登入。

我使用的 prompt 是：

```python
prompt = "What is Edge AI?"
```

接著我先測試沒有使用 `torch.compile` 時的生成時間，記錄多次 inference time。再將模型設定為 static KV cache，並編譯模型的 forward pass：

```python
model.generation_config.cache_implementation = "static"
model.forward = torch.compile(
    model.forward,
    mode="reduce-overhead",
    fullgraph=True
)
compiled_model = model
```

這裡的 static KV cache 是為了讓 LLM generation 時的 cache shape 比較固定，使 `torch.compile` 更容易最佳化。`torch.compile` 則會嘗試將模型的 forward pass 編譯成較有效率的計算圖，以減少推論時的 overhead。

在我的 WSL + RTX 4070 環境中，我遇到過 CUDA Graphs 和 static KV cache 互動造成的錯誤。錯誤訊息建議在每次 model invocation 前呼叫：

```python
torch.compiler.cudagraph_mark_step_begin()
```

因此我將 compiled inference 包成一個函式：

```python
def compiled_generate():
    torch.compiler.cudagraph_mark_step_begin()
    return compiled_model.generate(
        **inputs,
        max_length=max_token_length,
        pad_token_id=tokenizer.eos_token_id
    )
```

這樣可以明確告訴 PyTorch 每次 `generate()` 是新的 CUDA Graph step，避免前一次執行的 tensor 被後一次覆蓋。

最後測得的結果是：

```text
Original median: 1.7293795776367187
Compile median: 0.9154331970214844
Speedup: 1.8891379330174451x
```

作業要求 speedup 大於 `1.5x`，我的結果約為 `1.89x`，因此 Part 2 也達成要求。

## 我學到的東西

透過這次 Lab0，我學到的不只是怎麼把 notebook 跑完，而是比較完整地理解了一個模型從資料處理到部署格式轉換的流程。

在 Part 1 中，我學到影像模型訓練的基本步驟，包括 dataset、transform、DataLoader、model、loss、optimizer、scheduler、training loop 和 evaluation loop。以前我可能只知道 accuracy 是最後結果，但這次我比較清楚知道 loss 是如何透過 backward 計算 gradient，再由 optimizer 更新 model weights。

我也理解到 pretrained model 的概念。MobileNetV2 原本是為 ImageNet 1000 類訓練的模型，所以要拿來做 CIFAR-10 時，需要改掉最後一層 classifier。這讓我更理解 transfer learning 的實際做法。

在 Part 2 中，我第一次比較完整地接觸 LLM inference 的加速問題。我學到 LLM 生成文字時會使用 KV cache，而 static KV cache 可以讓 tensor shape 比較固定，進而幫助 `torch.compile` 做最佳化。我也學到 GPU 計時不能只靠普通時間函式，因為 CUDA 執行是 asynchronous，所以需要使用 `torch.cuda.Event` 和 `torch.cuda.synchronize()` 才能比較準確地測量 GPU inference time。

另外，這次我也學到環境管理的重要性。Colab 雖然方便，但 GPU 配額不一定穩定；使用本機 WSL + GPU 則需要處理 CUDA、PyTorch、Hugging Face、ONNX exporter 和 `torch.compile` 相關相容性問題。這些問題雖然一開始看起來像錯誤，但實際上也讓我更理解深度學習實作不是只有寫模型，環境和工具鏈也很重要。

## 繳交檔案

最後我整理出的繳交檔案如下：

- `314551147.ipynb`
- `314551147_acc_1.png`
- `314551147_acc_2.png`
- `314551147_onnx.png`
- `314551147_speedup.png`

這些檔案已壓縮成：

```text
314551147.zip
```
