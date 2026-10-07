# Lab 21 — Evaluation Report

**Họ tên**: Tạ Văn Tuấn · **MSSV**: 2A202602806 · **Ngày chạy**: 2026-10-07
**Tier**: `LAPTOP` (giữ nguyên cấu hình phần cứng của tier) · **Base model**: `Qwen/Qwen3.5-0.8B`
**GPU thực tế**: NVIDIA GeForce RTX 3050 Ti Laptop, 4.0 GB VRAM, sm_86, precision `bf16`
*(thông tin môi trường, từ `labkit.device.banner()` — không phải artifact chấm điểm, xem Phụ lục A)*
**Cấu hình tái lập**: `submission/RUN_CONFIG.env`

> **Nguồn số liệu.** Mọi metric dùng để chấm và mọi kết luận định lượng chính trong report
> này đều được đối chiếu với artifacts trong `results/`. Các thông tin môi trường (tên GPU,
> VRAM, capability), metadata trên Hugging Face Hub (kích thước trọng số), và log phụ của
> lần chạy được **ghi rõ nguồn riêng ngay tại chỗ** khi xuất hiện, và **không** được dùng làm
> metric chấm điểm. Phụ lục A ghi nguồn của từng nhóm số.
>
> **Quy ước trình bày.** Số trong report được in với số thập phân cố định (ví dụ `0.910`
> trong khi `results/autopsy.json` ghi `0.91`). Đó là cùng một giá trị, chỉ khác cách hiển
> thị; mọi chỗ khác đều khớp nguyên văn.

---

## 1. Setup — và lý do chọn model

| | |
|---|---|
| Dataset | Corpus kèm lab: 250 ticket CSKH tiếng Việt → JSON triage 4 trường (`intent`, `urgency`, `product`, `sentiment`) |
| Kích thước | `data/train_seed.jsonl` = 250 dòng · `data/eval_target.jsonl` = 50 dòng · `data/eval_regression.jsonl` = 15 dòng |
| Train / val | **225 / 25** (seed 42, `data/split/`) |
| p95 token | **98** (`results/token_stats.json`: p50=93, p95=98, p99=100, max=101) |
| `suggested_max_length` | **256** (làm tròn p95 lên luỹ thừa 2) |
| `max_length` của tier | **1024** |
| `MASK_MODE` | `assistant-only` |
| Epochs | **2.0** → **58 optimizer step** |
| `per_device_batch` × `grad_accum` | 1 × 8 = **8** (< ngưỡng 32 của deck §11.4) |
| `max_steps` (cả 4 run) | **58** |
| Template có giữ khối `think` không? | **CÓ** — xem §2 |

**Nguồn của bảng trên.** `results/runs.csv` (cột `tier`, `model`, `precision`) và
`results/baselines_frozen.json` (`tier`, `model`) là nguồn cho mọi dòng về cấu hình chạy.
Riêng RAM/VRAM của máy và `sm_86` là thông tin môi trường (Phụ lục A).

**Vì sao `BASE_MODEL=Qwen/Qwen3.5-0.8B` thay vì model mặc định của tier.** Máy huấn luyện
là laptop 4 GB VRAM. Model mặc định của tier `T4` là `unsloth/Qwen3.5-4B`, và tier `LAPTOP`
mặc định là `Qwen/Qwen3.5-2B`; cả hai đều lớn hơn 4 GB chỉ tính riêng trọng số (xem bảng
kích thước bên dưới), nên không nạp nổi — chứ chưa nói tới việc train. README cho phép
override model mà **giữ nguyên cấu hình phần cứng của tier** (`BASE_MODEL=<hf-id>`), nên tôi
giữ `COMPUTE_TIER=LAPTOP` (batch 1 × grad_accum 8, `max_length` 1024, precision theo device)
và hạ base model xuống `Qwen/Qwen3.5-0.8B` — đúng model mà tier `CPU` của lab ship, nên chat
template, `eos_token` và hành vi `think` vẫn là của họ Qwen3.5 mà lab thiết kế.

Kích thước trọng số dưới đây là **metadata trên Hugging Face Hub** (đọc qua HF API
`/api/models/<id>?blobs=true`), **không** phải số đo trong `results/` — chép lại để giải
thích quyết định chọn model, không dùng để chấm:

| Model | Trọng số (tổng các shard `.safetensors`) | Nguồn |
|---|---|---|
| `unsloth/Qwen3.5-4B` (mặc định tier T4) | ~9,32 GB (2 shard) | HF Hub metadata |
| `Qwen/Qwen3.5-2B` (mặc định tier LAPTOP) | ~4,55 GB (1 shard) | HF Hub metadata |
| `Qwen/Qwen3.5-0.8B` (**đã dùng**) | ~1,65 GB (1 shard) | HF Hub metadata |

**Đã chạy lại mask khi đổi base model (bắt buộc theo README).**
Output của `python scripts/check_mask_agreement.py` — đây là **stdout của một script tái lập
được**, không phải file trong `results/`:

```
model: Qwen/Qwen3.5-0.8B
chat template exposes {% generation %} markers: False
labkit assistant-only : 9/31 tokens (29.0%)
  supervised text: '{"intent": "doi_tra"}<|im_end|>\n'
TRL assistant_masks   : 0/31 tokens (0.0%)      <- tokenizer path: RỖNG
trainer path: TRL PATCHES the template -> 13/31 tokens
  supervised text: ' thinking\n\n</think>\n\n{"intent": "doi_tra"}<|im_end|>\n'
VERDICT: FAIL — the tokenizer-level mask is EMPTY.
```

Đây chính là lýdo lab **pre-tokenize bằng `labkit.data`** thay vì đặt
`assistant_only_loss=True`: đường tokenizer của TRL trả mask **rỗng** (train trên không
token nào, chỉ có một warning), còn đường trainer của TRL **vá template** và khi đó loss phủ
thêm cả khối `think` rỗng — không phải mask mà NB1 đã chứng minh. Script trả exit code 1 là
*kết quả đo*, không phải lỗi repo.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | **0.3936** (37/94 token) |
| Câu trả lời nằm trong loss (`answer_is_supervised`) | **true** |
| Câu hỏi KHÔNG nằm trong loss (`question_is_masked`) | **true** |

`0.3936 < 0.95` → không rơi vào bẫy "tính loss cả trên prompt". *(Ngưỡng `0.95` là hằng số của
rubric mục 1.1 và của `scripts/verify.py`, không phải một số đo trong `results/`.)*

**Đoạn được tính loss** (`supervised_preview` trong `results/mask_proof.json`):

```
{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

**Đoạn bị che** (`masked_preview`) — toàn bộ system + user turn + khối `think` rỗng:

```
<|im_start|>system
Phân loại ticket sau.<|im_end|>
<|im_start|>user
Alo shop, mình đặt balo laptop mã đơn VN411453. Cho tôi trả lại. Đã 3 ngày rồi. Cho tôi hỏi.<|im_end|>
<|im_start|>assistant
 thinking

</think>

```

So sánh trực tiếp, cùng một mẫu, hai chế độ (in ra ở NB1; dòng `everything` là
**output log của NB1**, không nằm trong `results/` — chỉ dòng `assistant-only` là artifact):

| mode | supervised | đọc được gì từ loss | nguồn |
|---|---|---|---|
| `assistant-only` | 37/94 (39%) | chỉ JSON trả lời + `<|im_end|>` | `results/mask_proof.json` |
| `everything` | 94/94 (100%) | **toàn bộ câu hỏi viết lại được** — đây là bug §22 | log NB1 (`runs/nb1.log`) |

`<|im_end|>` **nằm trong** phần supervised là cố ý: đó là tín hiệu dừng của model.

**Template có giữ `think` không?** — `results/template_check.json`:
`ok=true`, `open_tag_present=true`, `body_present=true`,
`verdict="reasoning preserved — safe to train on traces"`. Chuỗi render đầy đủ:

```
<|im_start|>user
2+2?<|im_end|>
<|im_start|>assistant
 thinking
buoc 1: kiem tra. buoc 2: tra loi.
</think>

4<|im_end|>
```

→ Template Qwen3.5 **không** xoá khối suy luận. Nhưng corpus này gồm 250 câu trả lời JSON
trần, và template đóng khối `think` **rỗng** ngay trong generation prompt — nên trên corpus
này `masked-think` và `response-only` là **no-op** (`labkit.data` cảnh báo đúng điều đó).
Vì vậy `valid_trace_rate` ở NB5 là `0.0` cho **mọi** arm: `generate_batch` được gọi với
`enable_thinking=False`, nên không arm nào phát ra `think`. Chỉ số này bằng 0 do cấu hình
decode, **không** phải do reasoning-trace collapse, và tôi không dùng nó để kết luận gì.

**`max_length` — vì sao 1024 dù p95 gợi ý 256.** Đây là chỗ lệch tier và cần giải thích
(rubric 1.3). Đo được: p95 = 98 token, max = 101 token trên cả 250 mẫu
(`results/token_stats.json`), nên `suggested_max_length` = 256; tier `LAPTOP` đặt 1024.
Tôi **giữ 1024** vì: (a) đây là cấu hình phần cứng của tier, và tôi chỉ được override base
model chứ không được override batch/`max_length`; (b) 1024 > max = 101 nên **không một mẫu
nào bị cắt** — mask, nhãn và câu trả lời nguyên vẹn; (c) hạ xuống 256 sẽ không đổi bất kỳ
con số nào trong `results/`, chỉ đổi giá trị ghi trong `SFTConfig`. Nói cách khác: chênh
lệch này vô hại về mặt đo lường, nhưng tôi không tự ý sửa nó — và ghi lại đây đúng như NB1
yêu cầu.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

Đọc trực tiếp `results/baselines_frozen.json`:

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | **0.000** | 0.6778 | **0.000** | 7666.8 |
| (b) base + optimized prompt | **0.490** | 0.6778 | **1.000** | 2365.0 |
| (c) LoRA fine-tune *(đo ở NB5)* | 0.985 | 0.0667 | 1.000 | 1272.5 |

`n_target=50`, `n_regression=15`, `eval_limit=null`, **`smoke_mode=false`** → **toàn bộ**
tập eval được dùng, không có `EVAL_LIMIT`. `optimized_prompt_sha = 719e74d3b6232053`.

**(b) có thật sự mạnh hơn (a) không?** **CÓ** — 0.490 vs 0.000 target, 1.000 vs 0.000
format. Đây không phải khoảng cách nhỏ: prompt naive khiến model 0.8B trả lời bằng văn xuôi
tiếng Việt thay vì JSON, nên **format = 0** và mọi field đều sai; prompt tối ưu ép được đúng
schema nên format = 1.0 và target nhảy lên 0.490. Lưu ý tôi **không** sửa `OPTIMIZED_PROMPT`:
SHA khớp với bản ship trong `src/labkit/config.py`, nên verify ghi "baseline (b) prompt
unmodified" là PASS. Cả hai baseline được đo **trước** NB3.

Mốc phải vượt ở NB5 là **(b) = 0.490**, không phải (a) = 0.000.

---

## 4. Giải phẫu cấu hình sai (NB4)

Ghép `results/runs.csv` (metadata huấn luyện) với `results/autopsy.json` (điểm target ở NB5 §4).
Cột `training_loss` dưới đây là cột **`final_loss` trong `results/runs.csv`** — xem ghi chú
ngay dưới bảng về việc nó *là gì*:

| Run | vị trí | r | trainable | LR | `training_loss` | **target** | format | latency ms | VRAM GB | giây |
|---|---|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear (12 module) | 16 | 10,822,656 | 1e-4 | 0.393 | **0.985** | 1.000 | 1272.5 | 1.97 | 558.4 |
| `attn_only` | q,v (2 module) | **271** | 10,822,656 | 1e-4 | 0.4356 | **0.910** | 1.000 | 929.1 | 1.98 | 367.0 |
| `wrong_lr` | text-linear | 16 | 10,822,656 | **1e-5** | 1.5424 | **0.330** | 0.995 | 1394.5 | 1.98 | 563.2 |
| `qlora` | text-linear | 16 | 10,822,656 | 1e-4 | 0.4239 | **0.945** | 1.000 | 1537.1 | **1.19** | 583.7 |

> **`training_loss` là gì — và không là gì.** NB3/NB4 ghi `result.training_loss` (NB4:
> `res.training_loss`) vào `runs.csv` dưới tên `final_loss`. Giá trị đó là **run-level
> training loss tổng hợp do `Trainer` trả về**, **không** phải loss đo riêng tại optimizer
> step cuối cùng — không có artifact nào trong `results/` chứng minh nó là "loss ở step 58".
> Vì vậy trong report này nó luôn được gọi là *run-level `training_loss`* (hay *training loss
> ghi trong `runs.csv`*), **không** bao giờ gọi là "final step loss" hay "loss tại step 58".
> Điều tương tự áp cho cột `trainable_params`, `learning_rate`, `peak_vram_gb`, `train_seconds`:
> tất cả đều đọc nguyên văn từ `runs.csv`.

**Công bằng của phép so sánh** (rubric mục 2, `make verify` kiểm tra tự động):

- **Cùng ngân sách tham số**: `attn_only` = 10,822,656 trainable, `correct` = 10,822,656 →
  **sai lệch 0.00%**. `matched_rank()` giải ra **r=271** (alpha=542). Nếu đem so
  `q,v @ r=16` với `all-linear @ r=16` thì `q,v` chỉ có **638.976** tham số — khoảng **1/17**
  ngân sách — và phép so đó không chứng minh gì. *(638.976 là output của bảng
  `modeling.describe_placement()` ở NB4 — log `runs/nb4.log`, không phải `results/`; hai giá
  trị 10,822,656 thì có trong `runs.csv`.)*
- **Cùng số step**: cả bốn run đều `max_steps=58`, lấy từ **một** nguồn duy nhất
  (`train.planned_steps(225, tier, training_epochs())` với `EPOCHS=2`).
- **Mỗi run đổi đúng một biến**: `attn_only` chỉ đổi *vị trí* (kèm rank để giữ ngân sách);
  `wrong_lr` chỉ đổi `learning_rate`; `qlora` chỉ đổi `load_in_4bit` (trong `runs.csv`:
  `load_in_4bit=True` cho `qlora`, `False` cho ba run còn lại). Mask, dataset, seed, số step,
  scheduler và batch hiệu dụng đều giống nhau.
- `qlora` được **chấm trên base 4-bit** (`load_in_4bit=SPECS["qlora"].load_in_4bit` trong
  NB5) — đúng cách nó được train.
- **Hạn chế đã biết:** `autopsy.json` chấm ba contrast trên **target + format** thôi
  (`with_regression=False` trong NB5), nên report này **không** có regression cho
  `attn_only` / `wrong_lr` / `qlora`. Tôi không suy diễn con số đó.

### 4.1 — `attn_only` cùng ngân sách tham số: thắng, thua hay hoà?

`attn_only` **THUA** `correct`: **0.910 vs 0.985 target**, tức **−7.5 điểm**, với ngân sách
tham số **giống hệt** (10,822,656 = 10,822,656, chênh 0.00%) và format bằng nhau (1.000).
Nó thắng ở đúng một chỗ: **latency 929.1 ms vs 1272.5 ms** (nhanh hơn ~27%, vì chỉ 2 module
được gắn adapter thay vì 12). Trên **run-level `training_loss`**, `attn_only` = 0.4356 so với
`correct` = 0.393 — chỉ kém **11%** về loss trong khi kém **7.5 điểm** về target. Cùng với
việc `qlora` (loss 0.4239) và `attn_only` (loss 0.4356) chỉ cách nhau **2.8%** về loss nhưng
cách nhau **3.5 điểm** về target, kết luận là: `training_loss` **không co giãn theo** khoảng
cách năng lực — trong lần chạy này nó chỉ đủ để xếp thứ tự, không đủ để định lượng.

Điều đó nói gì về *rank* so với *vị trí gắn adapter*: **trong phép so sánh này, tăng rank rất
mạnh (16 → 271, ×17) không bù được việc thu hẹp placement xuống `q,v` khi ngân sách trainable
parameter đã được khớp** — placement là đòn bẩy mạnh hơn rank. Tôi cố ý **không** viết "rank
không quan trọng": thí nghiệm này chỉ khớp ngân sách ở **một** điểm (r=16 cho `correct`) và
chỉ ở **một** base model, nên nó chứng minh *thứ tự ưu tiên* giữa placement và rank trong cấu
hình đã đo, không chứng minh một quy luật chung cho mọi rank. Cách diễn đạt đúng của kết quả
này là: "q,v kém không phải vì rank thấp" — đó là chẩn đoán sai cho cấu hình đã đo.

**Về thứ hạng — tôi báo cáo đúng cái đã đo, không dựng ra một nghịch đảo.** Xếp bốn run:

| | hạng 1 | hạng 2 | hạng 3 | hạng 4 |
|---|---|---|---|---|
| theo **target (NB5 §4)** | `correct` 0.985 | `qlora` 0.945 | `attn_only` 0.910 | `wrong_lr` 0.330 |
| theo **run-level `training_loss` (thấp = tốt)** | `correct` 0.393 | `qlora` 0.4239 | `attn_only` 0.4356 | `wrong_lr` 1.5424 |

**Hai thứ hạng này TRÙNG NHAU trong lần chạy này.** Tôi ghi thẳng điều đó ra thay vì tìm một
cách sắp xếp khác để tạo ra một nghịch đảo không tồn tại. Cái mà thí nghiệm này **thật sự**
chứng minh về `training_loss` mạnh hơn một nghịch đảo thứ hạng, và nó nằm ở §5: **`correct`
có training loss tốt nhất *và* target tốt nhất, rồi vẫn FAILED cổng hồi quy** (regression sụp
từ 0.6778 xuống 0.0667). Nếu tôi chỉ nhìn `training_loss` — hoặc chỉ nhìn target — tôi sẽ kết
luận "cấu hình này thắng" và bỏ qua đúng thứ đã hỏng. Đó là bằng chứng mạnh nhất trong bài
này cho việc **không được dùng training loss làm tiêu chuẩn deploy**: nó không nhìn thấy được
sự quên.

### 4.2 — `wrong_lr`: chỉ khác đúng một con số

Chỉ đổi `learning_rate` từ **1e-4** (thang LoRA, §11.3) xuống **1e-5** (thang full-FT); mọi
thứ khác — 12 module, r=16, alpha=32, bf16, mask, dataset, seed, **58 step** — giữ nguyên.

**Bằng chứng có trong `results/` (đây là phần chấm điểm):**

| | `correct` | `wrong_lr` | nguồn |
|---|---|---|---|
| `learning_rate` | 1e-4 | **1e-5** | `results/runs.csv` |
| run-level `training_loss` | **0.393** | **1.5424** | `results/runs.csv` |
| target | **0.985** | **0.330** | `results/autopsy.json` |
| format | 1.000 | 0.995 | `results/autopsy.json` |
| `max_steps` (giống nhau) | 58 | 58 | `results/runs.csv` |

Hai run khác nhau **đúng một con số** trong cấu hình, và kết quả lệch nhau rất xa: run-level
`training_loss` **0.393 so với 1.5424** (tệ hơn **3,9 lần**), còn trên tác vụ thì target
**0.985 so với 0.330**.

Điểm quan trọng nhất, và là lý do `wrong_lr` là một đối chứng đáng giá: nó đạt target **0.330**,
tức **−65,5 điểm** so với `correct` và **−16,0 điểm so với chính baseline (b) = 0.490** — tức
nó **không vượt nổi một prompt tốt**, và là run **duy nhất** trong bốn run nằm dưới mốc (b).
Một sai số 10× ở learning rate không chỉ "làm chậm hội tụ"; trong cùng ngân sách 58 step công
bằng, nó xoá sạch phần lợi mà `correct` giành được.

**Quan sát phụ, không dùng để chấm.** Log huấn luyện local (`runs/nb3.log`, `runs/nb4.log`)
cho thấy đường loss của `correct` giảm gần như đơn điệu theo hàm mũ, còn đường loss của
`wrong_lr` gần như **phẳng** suốt 2 epoch và `mean_token_accuracy` cuối run của nó thấp hơn
hẳn `correct`. Hai chỉ số đó (loss theo step, `mean_token_accuracy`) chỉ có trong **log**,
không có trong `results/`, nên tôi **không** đưa chúng vào bảng chấm điểm và **không** trích
số cụ thể; chúng chỉ giải thích *cơ chế* đằng sau hai con số đã có artifact.

Nếu chỉ nhìn loss mà không biết LR, tôi sẽ kết luận "run này vẫn đang học, chỉ cần thêm step"
— và đó chính là kết luận sai mà rubric 2.5 cảnh báo. Việc *nhiều step hơn* có cứu được nó hay
không thì thí nghiệm này **không** đo, và tôi không tuyên bố.

### 4.3 — `qlora`: tiết kiệm bao nhiêu VRAM, trả giá bằng gì?

Số đọc từ `results/runs.csv` và `results/autopsy.json`; các phần trăm là **tính từ chính hai
file đó**, ghi luôn công thức để kiểm chứng:

| | `correct` | `qlora` | chênh | tính từ |
|---|---|---|---|---|
| peak VRAM (GB) | 1.97 | **1.19** | **0.78 GB**, tức **−39,6%** | `(1.97 − 1.19) / 1.97`, `runs.csv` |
| target | 0.985 | 0.945 | **−4,0 điểm** | `autopsy.json` |
| format | 1.000 | 1.000 | 0 | `autopsy.json` |
| latency (ms) | 1272.5 | 1537.1 | **+264,6 ms**, tức **+20,8%** | `(1537.1 − 1272.5) / 1272.5`, `autopsy.json` |
| train_seconds | 558.4 | 583.7 | +25,3 s | `runs.csv` |
| trainable params | 10,822,656 | 10,822,656 | 0 | `runs.csv` |
| `max_steps` | 58 | 58 | 0 | `runs.csv` |

Vậy tradeoff: **đổi 0.78 GB VRAM lấy 4,0 điểm target và ~21% latency.** Khuyến nghị của nhà
cung cấp ("đừng dùng QLoRA cho dòng Qwen3.5", deck §13) được số đo **ủng hộ về hướng** — QLoRA
kém hơn — nhưng **biên độ nhỏ** ở quy mô 0.8B này. Trên card 4 GB thì 0.78 GB là khoản tiết
kiệm có ý nghĩa sống còn (đó là khác biệt giữa chạy được và OOM), nên tôi coi đây là tradeoff
*có lý* nếu 4 điểm target đó không quan trọng. Nhưng nó không cứu được gì: cả `correct` lẫn
`qlora` đều vượt (b) ở target, và **cả hai** đều không giải quyết được vấn đề đã giết verdict —
sự quên thảm hoạ (§5).

**Về nguyên nhân của phần latency tăng — tôi không kết luận.** Run QLoRA được đo **chậm hơn
20,8%** ở nhóm latency trong cấu hình này. Một nguyên nhân *khả dĩ* là overhead của việc chạy
base đã lượng tử hoá (dequantize trong lúc forward), nhưng thí nghiệm này **không cô lập** được
nguyên nhân latency: chỉ có một phép đo mỗi arm, batch cố định, và không có profiler. Nên tôi
chỉ báo cáo con số và để nguyên giả thuyết ở dạng giả thuyết. Lưu ý thêm rằng `train_seconds`
của `qlora` **không** thấp hơn — nó cao hơn 25,3 s — nên "QLoRA nhanh hơn" không đúng ở lần
chạy này; khoản tiết kiệm duy nhất đo được là VRAM.

---

## 5. Phán quyết (NB5)

Đọc trực tiếp `results/verdict.json` (không tính lại):

```
passed            = false
target_delta      = +0.495
regression_delta  = -0.611
reasons           = ["general capability regressed by 0.611 (tolerance 0.020)"]
valid_trace_rate  = 0.0
```

**Kết quả cổng hồi quy: FAILED.** `target Δ = +0.495` · `regression Δ = −0.611` ·
`valid_trace_rate = 0.0`.

**Diễn giải.** Bản fine-tune thắng mốc (b) ở nhóm **target** một cách dứt khoát: 0.985 so
với 0.490, tức +0.495 — và đây không phải may mắn của một vài mẫu.
`results/qualitative.json` chấm lại từng mẫu bằng greedy decode và đếm được **48 thắng /
2 hoà / 0 thua** trên 50 ticket, với **0** trường hợp ở mức *field* mà (b) đúng còn (c) sai
field đó. Nhóm **format** cũng đạt tuyệt đối 1.000 và nhanh hơn (b) (1272.5 ms so với
2365.0 ms) — nhất quán với việc model đã học được phản xạ trả JSON ngắn thay vì diễn giải dài,
dù thí nghiệm này không cô lập nguyên nhân của phần latency đó. Nếu cổng chỉ có một vế, đây là
một PASSED rất đẹp.

Nhưng cổng có vế thứ hai, và nó sập: **regression từ 0.6778 xuống 0.0667**, tức mất
**0.611** trong khi ngưỡng cho phép là 0.020. Bản fine-tune chỉ đạt keyword trên **1/15**
câu hỏi phổ thông (5 ca hoà, và 4 trong 5 cái hoà đó là 0.0/0.0 — base cũng sai; 10/15 là
thua thật, delta −1.0 hoặc −0.67). Verdict là FAILED, và tôi **không** sửa ngưỡng, không đổi
tập eval, không làm yếu prompt (b) để biến nó thành PASSED.

Nguyên nhân, theo đúng thứ tự chẩn đoán của NB5: `format` **không** phải thủ phạm (1.000);
`target` **không** phải thủ phạm (+0.495); nên đây là **catastrophic forgetting** — deck
§6.3. Ba nguyên nhân cụ thể, có bằng chứng:

1. **Không có replay data.** 225 mẫu, 2 epoch, và **100%** dữ liệu huấn luyện là một tác vụ
   hẹp (kiểm tra được trên `data/split/train.jsonl`: mọi dòng đều là ticket triage, không có
   dòng kiến thức phổ thông nào). Không trộn 1–5% dữ liệu phổ thông như deck §6.3 khuyến nghị.
   Đây là điều kiện *khả dĩ* giải thích mức sụp 0.611, không phải một phép đo nhân quả: để
   chứng minh, phải chạy lại với replay data (xem §7).
2. **`correct` là cấu hình không-hối-tiếc về *LoRA*, không phải về *dữ liệu*.** Placement
   text-linear + LR 1e-4 + 16-bit làm đúng việc của nó: nó *fit* tác vụ target gần như hoàn
   hảo. Cái sai nằm ở chỗ khác trong pipeline — ở dữ liệu, không ở LoRA.
3. **`regression` của corpus này vốn đã yếu.** (b) chỉ đạt 0.6778, và đọc chuỗi thật thì
   base 0.8B trả lời "thủ đô Việt Nam là **Hàn Quốc**" (`results/qualitative.json`,
   regression `i=0`). Nói cách khác mốc regression rất thấp, vậy mà fine-tune vẫn rơi thêm
   0.611 — mức độ quên còn tệ hơn con số gợi ý.

Ba giải thích trên chỉ áp cho `correct`, vì **`regression` không được đo cho ba contrast**
(`with_regression=False` trong NB5 — xem §4). Nên tôi **không** khẳng định rằng không cấu hình
LoRA nào trong bốn run sửa được sự quên; điều đo được chỉ là `correct` — cấu hình tốt nhất
trong bốn run theo cả target lẫn training loss — vẫn FAILED.

Một điểm tôi phải nói rõ vì nó dễ bị đọc sai: **`valid_trace_rate = 0.0` không phải bằng
chứng của reasoning-trace collapse.** `generate_batch` được gọi với `enable_thinking=False`,
nên *không arm nào* — kể cả base chưa fine-tune — phát ra khối `think`; chỉ số này bằng 0 do
cấu hình decode. Và trên corpus này `masked-think`/`response-only` là no-op, nên thí nghiệm
§17.5 không chạy được với dữ liệu ship sẵn. Tôi để nguyên số 0 và nói rõ nó là gì.

---

## 6. Định tính — có cả ca THUA

Nguồn: `results/qualitative.json`. Cách làm: greedy decode lại **cả hai arm** trên **cả hai
nhóm**, từng mẫu một — (b) = base + `OPTIMIZED_PROMPT`, (c) = base + `adapters/correct` chấm
bằng `NAIVE_PROMPT` đúng như NB5. Script `scripts/qualitative_breakdown.py` tái lập được.
Kiểm chứng tính toàn vẹn: (b) decode lại cho **target 0.49 — khớp chính xác** giá trị đã
đóng băng 0.49, và **regression 0.678** so với 0.6778 đã đóng băng; (c) cho **0.985**,
khớp `verdict.json`. Tập eval không bị đụng tới.

Quy tắc chọn: sắp xếp theo `delta = ft_score − b_score`, thắng giảm dần, thua tăng dần, hoà
theo chỉ số mẫu. **Không** chọn tay.

**Một phát hiện quan trọng phải nói trước bảng:** trên **nhóm target** kết quả là
**48 thắng / 2 hoà / 0 thua**, và **0** trường hợp ở mức *field* mà (b) đúng còn (c) sai.
Nghĩa là **không tồn tại** ca "fine-tune thua" nào trên tác vụ target để mà trích dẫn. Các ca
thua *có thật* nằm ở **nhóm regression: 0 thắng / 5 hoà / 10 thua**. Lấy ba mẫu có `ft_score`
thấp nhất của target rồi dán nhãn "FT thua" sẽ là bịa — nên tôi lấy ca thua từ đúng nhóm mà
chúng đã xảy ra, và đó cũng chính là nhóm làm verdict FAILED.

| # | Nhóm | Ticket / Câu hỏi | Nhãn đúng (hoặc keyword) | (b) base + optimized prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|---|
| 1 | target | `i=5` — "Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. **Thiếu phụ kiện**. Khi nào tiện. Cho tôi hỏi." | `san_pham_loi` / `thap` / `nồi chiên không dầu` / `trung_tinh` | `{"intent": "hoan_tien", "urgency": "cao", "product": "nồi chiên không dầu", "sentiment": "tich_cuc"}` → **0.25** | `{"intent": "san_pham_loi", "urgency": "thap", "product": "nồi chiên không dầu", "sentiment": "trung_tinh"}` → **1.00** | ✅ **FT thắng** delta **+0.75**. (b) đọc "thiếu phụ kiện" thành đòi hoàn tiền + gấp + tích cực; (c) đúng cả 4 field. Đây là loại tín hiệu mà prompt không truyền được nhưng 225 mẫu thì truyền được. |
| 2 | target | `i=6` — "Xin chào, mình đặt balo laptop mã đơn DH863123. **Đổi size**. Hỏi cho biết thôi. Lần cuối mua ở đây." | `doi_tra` / `thap` / `balo laptop` / `tieu_cuc` | `{"intent": "hoan_tien", "urgency": "cao", "product": "balo laptop", "sentiment": "tich_cuc"}` → **0.25** | `{"intent": "doi_tra", "urgency": "thap", "product": "balo laptop", "sentiment": "tieu_cuc"}` → **1.00** | ✅ **FT thắng** delta **+0.75**. (b) bỏ sót tín hiệu "lần cuối mua ở đây" (tiêu cực) và "hỏi cho biết thôi" (urgency thấp); (c) bắt đúng cả hai. |
| 3 | target | `i=12` — "Shop ơi, mình đặt áo khoác gió mã đơn VN613097. **Bị lỗi**. Khi nào tiện. Cảm ơn shop nhiều." | `san_pham_loi` / `thap` / `áo khoác gió` / `tich_cuc` | `{"intent": "van_chuyen", "urgency": "cao", ...}` → **0.50** | `{"intent": "van_chuyen", "urgency": "thap", ...}` → **0.75** | ⚠️ **FT vẫn sai `intent`** (đoán `van_chuyen`, nhãn là `san_pham_loi`) — nhưng vẫn thắng (b) vì sửa được `urgency`. Đây là **1 trong 3** mẫu mà (c) không đạt 1.0. |
| 4 | regression | `i=2` — "**1 km bằng bao nhiêu mét?**" | keyword `1000` | *"1 km (kilometr) là đơn vị đo lường dài rộng, … **kilo = 1000** …"* → **1.00** | `{"intent": "hoi_thong_tin", "urgency": "thap", "product": "tieu_cuc", "sentiment": "trung_tinh"}` → **0.00** | ❌ **FT THUA** delta **−1.00**. Câu hỏi kiến thức bị trả lời bằng JSON triage, và `product` = `"tieu_cuc"` — một giá trị sentiment nằm ở vị trí sản phẩm. Sau fine-tune model **không còn trả lời đúng câu hỏi này**; output cho thấy định tuyến tác vụ lệch sang schema triage. |
| 5 | regression | `i=8` — "**Ai là tác giả của Truyện Kiều?**" | keyword `Nguyễn Du` | *"Truyện Kiều là tác phẩm nổi tiếng nhất của nhà văn **Nguyễn Du**…"* → **1.00** | `{"intent": "hoi_thong_tin", "urgency": "thap", "product": "truyện cổ tích"}` → **0.00** | ❌ **FT THUA** delta **−1.00**. Sau fine-tune model **không còn trả lời đúng câu hỏi này**: nó sinh một `product` lấy từ tên tác phẩm (`"truyện cổ tích"`). Không suy ra được "kiến thức về Nguyễn Du đã bị xoá" từ một mẫu — chỉ suy ra được hành vi sai ở mẫu này. |
| 6 | regression | `i=9` — "**Một năm có bao nhiêu tháng?**" | keyword `12` | *"Một năm có **12 tháng**…"* → **1.00** | `{"intent": "hoi_tieu_cuc", "urgency": "thap", "product": "tieu_cuc"}` → **0.00** | ❌ **FT THUA** delta **−1.00**. `intent` = `hoi_tieu_cuc` **không nằm trong vocab nhãn hợp lệ** (`doi_tra`/`van_chuyen`/`hoan_tien`/`san_pham_loi`/`hoi_thong_tin`) — kiểm chứng được từ `results/qualitative.json`. |
| 7 | regression | `i=6` — "**2 mũ 10 bằng bao nhiêu?**" | keyword `1024` | *"…$$2^{10} = 2 \times 2 \times…$$"* → **1.00** | `{"intent": "hoi_thong_tin", "urgency": "thap", "urgency_score": 0}` → **0.00** | ❌ **FT THUA** delta **−1.00**. Sinh ra khoá **`urgency_score`** không thuộc schema 4 khoá — một dạng sai schema, gợi ý rằng ranh giới giữa các field bị trộn trong lúc fit. |
| 8 | regression | `i=14` — "**Giải thích ngắn gọn quang hợp là gì.**" | keyword `ánh sáng`, `cây` | *"Quang hợp là quá trình sinh học mà các thực vật … sử dụng năng lượng **ánh sáng** mặt trời…"* → **1.00** | `{"intent": "hoi_tieu", "urgency": "thap", "product": "hoi_tieu"}` → **0.00** | ❌ **FT THUA** delta **−1.00**. Lại là nhãn ngoài vocab `hoi_tieu`. Câu trả lời sai cả nội dung lẫn schema. |
| 9 | regression | `i=12` — "**Tóm tắt ý nghĩa của câu 'Có công mài sắt có ngày nên kim'.**" | keyword `kiên trì`, `cố gắng`, `bền` | *"…câu tục ngữ rất nổi tiếng…"* → **0.67** | `{"intent": "hoi_thong_tin", "urgency": "thap", "product": "sắt", "sentiment": "trung_tinh"}` → **0.00** | ❌ **FT THUA** delta **−0.67**. Nghĩa bóng của thành ngữ bị hiểu thành sản phẩm `"sắt"`. |
| 10 | regression | `i=0` — "**Thủ đô của Việt Nam là thành phố nào?**" | keyword `Hà Nội` | *"Thành phố thủ đô của Việt Nam là **Hàn Quốc**…"* → **0.00** | `{"intent": "hoi_thong_tin", …}` → **0.00** | ➖ **Hoà 0.00/0.00.** Base tự nó đã trả lời sai nặng. Giữ ở đây vì nó cho thấy **cả hai arm đều yếu** — mốc (b) không phải một mốc cao. |
| 11 | regression | `i=4` — "**Dịch sang tiếng Anh: 'Tôi thích đọc sách'.**" | keyword `read`, `book` | `"I like reading books."` → **1.00** | `"I like reading books."` → **1.00** | ➖ **Hoà 1.00/1.00.** Ca **duy nhất** fine-tune còn giữ được hành vi ngoài-miền — vì câu lệnh *dịch* nằm xa phân bố triage hơn cả. 14/15 ca còn lại thì không. |

**Mẫu chung của 10 ca FT thua (nhóm regression).** Các đếm dưới đây được **tính lại từ
`results/qualitative.json`** (không đọc bằng mắt):

1. **Câu hỏi phổ thông bị trả lời bằng schema triage.** **9/10** ca loss trả về một object có
   `intent` và `urgency`; ca còn lại là một chuỗi JSON bị lặp/hỏng nên không parse được. Hành vi
   sau fine-tune cho thấy **task routing bị lệch mạnh sang schema triage** — "output phải là
   JSON 4 khoá" trở thành phản xạ gần như không phụ thuộc vào việc người dùng hỏi gì.
2. **Nhãn/khoá ngoài vocab xuất hiện ở 5/10 ca.** Cụ thể: **4** ca có *giá trị* nhãn ngoài vocab
   (`intent` = `hoi_tieu_cuc` ở `i=9` và `i=10`, `san_pham_tieu` ở `i=7`, `hoi_tieu` ở `i=14`)
   và **1** ca có *khoá* không thuộc schema (`urgency_score` ở `i=6`). Đây là dấu hiệu rõ nhất
   của việc fit quá sát trên tập nhỏ: các nhãn sinh ra nằm ngoài cả vocab nhãn hợp lệ
   (`hoi_thong_tin`, `doi_tra`, `van_chuyen`, `hoan_tien`, `san_pham_loi`).
3. **`product` trở thành "chỗ chứa" của nhiều danh từ.** `"truyện cổ tích"` (từ "Truyện Kiều"),
   `"sắt"` (từ "mài sắt"), `"nước sôi"`, `"tieu_cuc"`. Field `product` được định nghĩa là "tên
   sản phẩm xuất hiện nguyên văn trong ticket" — các output này cho thấy model vẫn sinh đúng
   *vị trí* của field nhưng không phân biệt được *khi nào không có sản phẩm*. **2/10** ca loss
   bỏ hẳn khoá `product`.
4. **Không ca nào quên theo kiểu "mất tiếng Việt"**: câu trả lời vẫn trôi chảy. Đây là quên
   **có cấu trúc** ở mức *định tuyến tác vụ*, không phải mất năng lực ngôn ngữ. Vì thế `format`
   vẫn 1.000 trong khi `regression` sụp còn 0.0667: đúng cái bẫy mà deck §6.3 mô tả, và là lý
   do phải đo **bốn nhóm** chứ không đo một nhóm.

Các đếm ở mục 1–3 là **quan sát trên 10 mẫu**, không phải một phép đo trên phân bố rộng: chúng
đủ để mô tả *kiểu* sai, không đủ để ước lượng *tần suất* sai trong thực tế.

---

## 7. Kết luận & điều tôi học được

### Kết luận

**Tôi không nên deploy bản fine-tune này.** Nó thắng ở đúng thứ nó được train để thắng —
target 0.985 so với mốc (b) 0.490, format 1.000, và latency thấp hơn (b)
(`(2365.0 − 1272.5) / 2365.0` = **−46,2%**, tính từ `results/verdict.json`) — và thua ở đúng
thứ khiến nó không được phép deploy: regression rơi từ 0.6778 xuống 0.0667, mất 0.611 trong
khi ngưỡng là 0.020. Cổng hồi quy trả về FAILED, và tôi giữ nguyên FAILED đó. Một adapter trả
JSON triage cho *mọi* câu hỏi là một adapter đã hỏng: trong 15 câu regression nó chỉ đạt
keyword ở **1** câu, và ở mẫu `i=0` thì ngay cả base cũng sai ("thủ đô Việt Nam là Hàn Quốc").
Trong sản phẩm thật, đây là lỗi nghiêm trọng hơn "chưa fine-tune", vì nó phá cả những thứ đang
chạy tốt.

Đòn bẩy thật sự, xếp theo mức độ ảnh hưởng mà **số đo** của tôi chứng minh:

1. **Dữ liệu** — ứng viên lớn nhất, và là thứ duy nhất có thể *giết* kết quả. Placement/LR/rank
   chỉ di chuyển điểm trong khoảng 0.330–0.985 trên **cùng một tác vụ**; còn việc model có làm
   được việc khác hay không thì **không** cấu hình LoRA nào trong bốn run được đo
   (`regression` chỉ đo cho `correct`). Nói cách khác: 225 mẫu, 2 epoch, 0% replay là
   **giả thuyết** giải thích mức mất 0.611 — mạnh, nhưng chưa được kiểm chứng trong lab này.
2. **Learning rate** — đòn bẩy lớn thứ hai, và là đòn bẩy *rẻ nhất*. Chỉ đổi 1e-4 → 1e-5,
   target sụp 0.985 → 0.330 (**−65,5 điểm**), tệ hơn cả prompt tay (b). Một con số.
3. **Vị trí gắn adapter** — `attn_only` với ngân sách tham số **giống hệt** vẫn thua 0.910 so
   với 0.985; muốn bù bằng rank thì phải nâng r=16 → **r=271**, và vẫn thua. **Trong phép so
   sánh này**, placement là đòn bẩy mạnh hơn rank.
4. **Rank** — bằng chứng mạnh nhất rằng rank *không phải* đòn bẩy ở cấu hình đã đo: ×17 rank
   (16 → 271) với cùng ngân sách tham số không lấy lại được 7,5 điểm. Phạm vi: **một** mức
   ngân sách, **một** base model — không phải một quy luật cho mọi rank.
5. **Mask** — điều kiện *tiên quyết*, không phải đòn bẩy. Mask đúng (0.3936, cả hai assert
   xanh) không làm model giỏi lên; mask sai thì mọi số sau đó vô nghĩa. Thực tế
   `check_mask_agreement.py` cho thấy nếu tôi tin vào `assistant_only_loss=True` của TRL thì
   pipeline này đã train trên **0 token** — và training loss vẫn sẽ in ra rất đẹp.

Nói ngắn: **bài toán này *có* đáng fine-tune** — (c) vượt (b) 0.495 điểm target là một
khoảng cách lớn và thật — nhưng **cách tôi làm thì không đáng deploy**, vì tôi tối ưu một
nhóm điểm duy nhất và trả giá bằng nhóm còn lại. Bước sửa đúng không phải tăng rank, không
phải đổi sang QLoRA (cả hai đều không chạm tới nguyên nhân), mà là **trộn 1–5% dữ liệu phổ
thông vào training set** (deck §6.3) rồi chạy lại cổng bốn nhóm.

### Ba điều tôi học được

1. **Tôi từng nghĩ "LoRA yếu thì tăng rank".** Thí nghiệm matched-parameter cho thấy điều
   ngược lại một cách dứt khoát: `attn_only` được đẩy lên **r=271** — gấp 17 lần `correct` —
   với **đúng cùng** 10,822,656 tham số huấn luyện, và vẫn thua **7.5 điểm** target (0.910 vs
   0.985). Trước lab này tôi sẽ đọc "q,v @ r=16 kém" thành "rank thấp"; giờ tôi biết nó là
   "gắn sai chỗ", và cách phân biệt hai giả thuyết đó là **khớp ngân sách tham số trước khi
   so**, không phải tăng rank rồi so.

2. **Tôi từng nghĩ loss thấp là bằng chứng model tốt.** `correct` có run-level `training_loss`
   **thấp nhất** (0.393) *và* target **cao nhất** (0.985) — rồi vẫn **FAILED**, vì regression
   sụp còn 0.0667. Loss không nhìn thấy được sự quên: nó chỉ đo trên phân bố mà tôi đã dạy, còn
   cái hỏng nằm ở phân bố tôi không dạy. Tôi cũng học được rằng thứ hạng loss và thứ hạng
   target **có thể trùng nhau** (lần này trùng) mà kết luận vẫn sai — nên "loss xếp đúng thứ
   tự" không phải giấy thông hành để dùng loss thay cho đánh giá tác vụ.

3. **Tôi từng nghĩ `assistant_only_loss=True` là một flag an toàn.** Trên Qwen3.5 nó không
   phải: `scripts/check_mask_agreement.py` cho thấy đường tokenizer trả mask **0/31 token**
   (train trên không gì cả, chỉ một warning), còn đường trainer **vá template** và phủ thêm
   khối `think` rỗng → 13/31 token, khác NB1 (9/31). Nếu tôi tin vào flag đó, pipeline sẽ
   chạy "thành công" và mọi con số đều vô nghĩa. Bài học cụ thể: **mask phải được decode
   ngược ra text và đọc**, không phải được tin qua một tham số thư viện.

### Nếu có thêm 2 giờ nữa, tôi sẽ thử

Trộn **1–5% replay data phổ thông** vào 225 mẫu (deck §6.3) và chạy lại NB3 + NB5, để trả lời
câu hỏi mà thí nghiệm này để mở: **regression phải tăng bao nhiêu để cổng chuyển thành
PASSED**, và liệu `target` có tụt theo không? Đây là can thiệp duy nhất trong số các can thiệp
đã đo (LR, placement, rank, quantization) mà bằng chứng chỉ thẳng vào nguyên nhân — bốn cái
kia chỉ di chuyển điểm trong cùng một tác vụ, không cái nào chạm tới sự quên.

---

## Phụ lục A — Kiểm chứng từng con số

**A.1 — Metric chấm điểm (nguồn: `results/`)**

| Con số | File nguồn |
|---|---|
| template giữ `think` | `results/template_check.json` (`ok`, `verdict`) |
| `supervised_fraction` 0.3936, hai assert | `results/mask_proof.json` |
| p50 = 93, p95 = 98, p99 = 100, max = 101, suggested 256 | `results/token_stats.json` |
| (a) và (b): target, regression, format, latency | `results/baselines_frozen.json` |
| `n_target=50`, `n_regression=15`, `eval_limit=null`, `smoke_mode=false`, `optimized_prompt_sha` | `results/baselines_frozen.json` |
| `max_steps`, `r`, `lora_alpha`, `trainable_params`, `learning_rate`, `final_loss`, `peak_vram_gb`, `train_seconds`, `n_target_modules`, `tier`, `model`, `precision`, `load_in_4bit` | `results/runs.csv` |
| target + format + latency của 4 run | `results/autopsy.json` |
| FAILED, `target_delta` +0.495, `regression_delta` −0.611, `comparison` | `results/verdict.json` |
| 48/2/0 (target) và 0/5/10 (regression), 15+50 hàng có text, `reproduced_b_target`, `reproduced_ft_target` | `results/qualitative.json` |
| merge: 0.985 → 0.985, `delta=0.0`, `tolerance=0.01`, `n=50` | `results/merge_check.json` |
| hot-swap: 3 adapter, `same_base_loaded=true`, `passed=true` | `results/hotswap_check.json` |

Các phần trăm trong report đều **tính từ chính các file trên** và ghi kèm công thức tại chỗ
(§4.3, §7): VRAM `(1.97−1.19)/1.97 = 39,6%`; latency QLoRA `(1537.1−1272.5)/1272.5 = 20,8%`;
latency (c) so (b) `(2365.0−1272.5)/2365.0 = 46,2%`; loss ratio `1.5424/0.393 = 3,9×`.

**A.2 — Thông tin KHÔNG nằm trong `results/` (ghi rõ, không dùng để chấm)**

| Thông tin | Nguồn | Vì sao không phải artifact |
|---|---|---|
| Tên GPU, 4.0 GB VRAM, `sm_86`, precision `bf16` | `labkit.device.banner()` (log) | thuộc tính môi trường; `runs.csv` chỉ ghi `precision` |
| HF tokenizer: `9/31`, `0/31`, `13/31` token | stdout của `scripts/check_mask_agreement.py` | script tái lập được, nhưng không ghi vào `results/` |
| Kích thước trọng số 9,32 / 4,55 / 1,65 GB | HF Hub API metadata | metadata bên ngoài, dùng để giải thích chọn model |
| `everything` = 94/94 token | log NB1 (`runs/nb1.log`) | NB1 chỉ ghi `assistant-only` vào `mask_proof.json` |
| `q,v @ r=16` = 638.976 tham số | log NB4 (`runs/nb4.log`), bảng `describe_placement()` | chỉ 10,822,656 có trong `runs.csv` |
| loss theo step, `mean_token_accuracy` | log NB3/NB4 | mô tả *cơ chế*, không dùng để chấm (§4.2) |
| `adapters/merged/` = 1,435 MB | kích thước file trên đĩa | không phải artifact; adapter không được commit |
| 225 train / 25 val | `data/split/{train,val}.jsonl` | dữ liệu, tái sinh bằng NB1 (seed 42); không nằm trong `results/` |

Không có con số nào ở A.2 được dùng làm bằng chứng cho phán quyết hay cho điểm target/regression.

**A.3 — Tái lập bảng định tính.** `scripts/qualitative_breakdown.py` decode lại cả hai arm và
**assert** rằng nó khớp giá trị đã đóng băng — (b) target 0.490 khớp 0.490 đã đóng băng, (b)
regression 0.678 so với 0.6778 đã đóng băng. Đó là cách tôi kiểm tra rằng bảng định tính đang
đo đúng thứ đã đóng băng, chứ không phải một lần chạy khác.

**A.4 — Cấu hình tái lập.** `submission/RUN_CONFIG.env` là bản ghi máy đọc được của cấu hình
đã tạo ra `results/` (`COMPUTE_TIER=LAPTOP`, `BASE_MODEL=Qwen/Qwen3.5-0.8B`,
`MASK_MODE=assistant-only`, `EPOCHS=2`, `EVAL_LIMIT` không đặt). Ô "3b. Tái lập committed
evidence" trong `colab/Lab21_RUN_ALL.ipynb` chạy đúng cấu hình đó.

## Phụ lục B — Thưởng đã làm

- [x] **B1** — NB6 merge + hot-swap ≥2 adapter. **Có hai artifact máy đọc được:**
      `results/merge_check.json` (merge không làm tụt điểm) **và** `results/hotswap_check.json`
      (3 adapter cùng nạp trên một base, `passed=true`). Chi tiết: Phụ lục C.
- [ ] **B2** — dataset miền riêng ≥200 mẫu + `data/CUSTOM_DATASET.md`: **không làm.** Tôi giữ
      corpus ship sẵn để `data/checksums.json` không đổi và phép so sánh giữ nguyên.
- [ ] **B3** — reasoning-trace collapse (`assistant-only` vs `response-only`): **không làm.**
      Corpus 250 câu trả lời JSON trần không có `think` nào, và `labkit.data` cảnh báo hai
      mode này là no-op trên corpus đó — muốn chạy §17.5 phải có dataset mới (tức phải làm
      B2 trước). Tôi không nhận điểm cho một thí nghiệm không chạy.
- [ ] **B4** — quét rank có kiểm soát r ∈ {8, 16, 64}: **không làm** trong lượt này.
- [ ] **B5** — push adapter lên HuggingFace Hub công khai: **không làm.** Không có token HF
      được cấu hình cho phiên chạy này, và tôi không dùng credential sẵn có trên máy để
      publish công khai khi chưa được cho phép. Không có link HF nên không nhận +2.

## Phụ lục C — NB6 (bonus B1): merge + hot-swap

Đã chạy `python notebooks/06_merge_and_serve.py`. Đọc từ `results/merge_check.json`:

| | |
|---|---|
| Điểm trước merge | **0.985** (50 mẫu target, `NAIVE_PROMPT`) |
| Điểm sau merge (`merge_and_unload()`) | **0.985** |
| `delta` | **0.0** (ngưỡng `tolerance = 0.01`) → **assert PASS** |
| `n` | 50 |

**Trong lần chạy này không quan sát thấy tụt điểm sau merge**: 0.985 → 0.985 trên 50 mẫu
target. Về mặt đại số, merge cộng trực tiếp LoRA delta vào base weights
(`W = W₀ + (α/r)·BA`), nên không có xấp xỉ *cấu trúc* nào ở bước này; nhưng tôi **không** kết
luận rằng merge không bao giờ gây sai số. Adapter được train ở bf16 và merge trên cùng base
bf16, nên ở đây dtype không gây chênh lệch đo được — nhưng nếu dtype lúc merge khác đi (ví dụ
merge sang fp16 trong khi train bằng bf16, hoặc DoRA cần PEFT ≥ 0.10 để gộp đúng vector
magnitude), kết quả có thể khác. Đó chính là lý do NB6 giữ `TOL = 0.01` và **assert** thay vì
tin vào lý thuyết. Kết luận có phạm vi: **trong cấu hình đã đo**, deploy dạng merge cho điểm
task bằng deploy dạng adapter rời (0.985, khớp cả `verdict.json` lẫn `qualitative.json`), và
không tốn overhead adapter lúc suy luận.

### Hot-swap ≥2 adapter trên cùng một base — artifact `results/hotswap_check.json`

Trước đây phần hot-swap chỉ tồn tại dưới dạng output in ra, nên grader chỉ đọc lại được
assert của merge. NB6 giờ ghi thêm một artifact máy đọc được:

```json
{
  "base_model": "Qwen/Qwen3.5-0.8B",
  "tier": "LAPTOP",
  "available_adapters": ["correct", "attn_only", "qlora"],
  "n_adapters": 3,
  "n_requested": 3,
  "same_base_loaded": true,
  "ticket": "Cho mình hỏi, mình đặt chuột không dây mã đơn VN232232. Cho tôi trả lại. Gấp. Shop hỗ trợ tốt.",
  "outputs": { "...": "mỗi adapter một câu trả lời cho CÙNG ticket" },
  "passed": true
}
```

`passed = (n_adapters >= 2)` — tức đúng ngưỡng của rubric B1. Ba adapter trả lời cùng một
ticket:

```
[correct]   -> {"intent": "doi_tra",   "urgency": "cao",        "product": "chuột không dây", "sentiment": "tich_cuc"}
[attn_only] -> {"intent": "doi_tra",   "urgency": "trung_binh", "product": "chuột không dây", "sentiment": "tich_cuc"}
[qlora]     -> {"intent": "hoan_tien", "urgency": "cao",        "product": "chuột không dây", "sentiment": "tich_cuc"}
```

Hai điều đáng chú ý. Thứ nhất, **cùng một base trong VRAM, ba "khách hàng" khác nhau, ba câu
trả lời khác nhau** — đây chính là lập luận kinh tế của LoRA: base nạp một lần, mỗi tác vụ là
một adapter vài chục MB. Thứ hai, cả ba **bất đồng** ở `intent`/`urgency`. Ticket này có nhãn
trong tập eval (`results/qualitative.json`, target `i=0`) là `doi_tra`/`cao`/`chuột không dây`/
`tich_cuc`, nên ở **một mẫu đơn lẻ** này `qlora` sai `intent` và `attn_only` sai `urgency`,
còn `correct` đúng cả 4 field. Tôi chỉ dùng **một mẫu** ở đây để minh hoạ *cơ chế* hot-swap —
không dùng nó để kết luận về chất lượng; kết luận về chất lượng đến từ `results/autopsy.json`
(50 mẫu).

Ghi chú về kích thước: `adapters/merged/` là **1.435 MB** `model.safetensors` trên đĩa (thông
tin môi trường, không phải artifact) và **không** được commit — nó sinh lại được bằng NB6.
`adapters/*/` nằm trong `.gitignore` đúng như thiết kế, và bản nộp này theo rubric **Option C
(code-only)**: `results/` đầy đủ + report + `requirements.txt`.
