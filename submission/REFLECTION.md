# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**

Hai điều, và cả hai đều ngược với dự đoán của tôi trước khi chạy.

Thứ nhất là **quy mô của đòn bẩy learning rate**. Tôi nghĩ LR là một "núm tinh chỉnh" —
sai 10 lần thì chậm hơn, thêm step là bù được. Số đo cho thấy nó là khác biệt giữa
**có và không**: chỉ đổi `1e-4 → 1e-5`, mọi thứ khác giống hệt kể cả `max_steps=58`, target rơi
**0.985 → 0.330** (−65,5 điểm) và rơi **xuống dưới cả baseline prompt tay (0.490)**. Run-level
`training_loss` là **0.393 vs 1.5424** — gấp **3,9 lần**. Một con số, một dòng `SFTConfig`.

Thứ hai là **cái chết của kết quả nằm ở nhóm điểm tôi không hề tối ưu**. Tôi dành gần hết
thời gian để làm `target` đúng, và nó đúng thật: 0.985 so với mốc 0.490, format 1.000,
**48 thắng / 2 hoà / 0 thua** trên 50 ticket, không một field nào mà (b) thắng (c). Rồi cổng
trả về **FAILED**, vì `regression` sụp **0.6778 → 0.0667**. Điều làm tôi ngạc nhiên không
phải là "quên thảm hoạ có thật" — tôi biết nó có thật — mà là **nó quên theo kiểu có cấu
trúc**: model vẫn viết tiếng Việt trôi chảy, vẫn ra JSON hợp lệ, `format` vẫn 1.000, nhưng
nó trả lời "Ai là tác giả Truyện Kiều?" bằng
`{"intent": "hoi_thong_tin", "urgency": "thap", "product": "truyện cổ tích"}`. Nó không hỏng
cú pháp và không hỏng tiếng Việt; hành vi cho thấy **task routing bị lệch mạnh sang schema
triage**. Nếu chỉ nhìn `format` và `target` tôi sẽ tưởng mọi thứ ổn.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

Không. Tôi dự đoán thời gian sẽ chảy vào **huấn luyện** (NB3 + NB4 = 4 run × 58 step) —
thực tế tổng `train_seconds` của bốn run trong `results/runs.csv` là **2.072 giây wall-clock,
tức ~35 phút**, đúng như dự đoán. Chỗ ngốn thời gian thật là **sinh văn bản để đánh giá**.
Con số lấy được từ artifact: latency trung bình mỗi mẫu của (a) là **7666,8 ms** so với
**2365,0 ms** của (b) — **gấp 3,2 lần** trên đúng cùng 50 mẫu (`results/baselines_frozen.json`);
cộng lại thì NB2 mất khoảng 11 phút và NB5 khoảng 5 phút theo log local.

Lý do rất cụ thể và tôi chỉ hiểu sau khi đọc log: prompt naive khiến model **không biết dừng**.
Nó không ra JSON nên cứ diễn giải cho tới hết `max_new_tokens=160`; prompt tối ưu ép được
JSON ngắn nên dừng sớm hơn nhiều. Vì vậy **baseline yếu nhất lại là baseline đắt nhất để đo**.
Đây là bài học về ngân sách tôi không lường trước: chi phí đánh giá tỉ lệ với **độ dài output**,
không phải số mẫu.

Hai chỗ khác cũng ăn thời gian mà tôi không dự đoán: (a) `Qwen3.5-0.8B` dùng **linear
attention** và máy này không có `causal_conv1d`/`flash-linear-attention`, nên mọi forward
rơi vào reference PyTorch — log cảnh báo "much slower" ở **mỗi** lần nạp model; (b) trên
Windows, HF cache mặc định ở `C:\Users\...\.cache\huggingface` **không ghi được** trong phiên
này, và console mặc định là **cp1252** nên `print()` tiếng Việt ném `UnicodeEncodeError` ngay
dòng đầu của NB1. Cả hai đều là vấn đề môi trường, không phải bug của repo — nhưng phải sửa
(`HF_HOME`, `PYTHONUTF8=1`) trước khi pipeline chạy được.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Tôi từng tin **"tăng rank là cách chính để tăng năng lực của LoRA"**. Kết quả
matched-parameter phá niềm tin đó một cách dứt khoát, và phá đúng bằng cách tôi không ngờ:
tôi không so `q,v @ r=16` với `all-linear @ r=16` — phép so đó vô nghĩa vì `q,v` chỉ có
638.976 tham số, bằng **1/17** ngân sách. Tôi dùng `matched_rank()`, nó giải ra **r=271**,
và `count_lora_params` xác nhận **10.822.656 = 10.822.656** — sai lệch **0.00%**. Rồi
`attn_only` vẫn thua **7.5 điểm** target (0.910 vs 0.985). Tôi đã nhân rank lên **17 lần**
và không mua được gì. Kết luận tôi rút ra không phải "rank không quan trọng" mà là **rank
chỉ có nghĩa sau khi ngân sách tham số đã được khớp** — nếu chưa khớp, mọi so sánh rank chỉ
đang đo ngân sách.

Niềm tin thứ hai tôi bỏ: **"training loss thấp là bằng chứng model tốt"**. `correct` có
run-level `training_loss` thấp nhất (0.393) *và* target cao nhất (0.985) — rồi vẫn FAILED.
Loss chỉ đo trên phân bố tôi đã dạy; cái hỏng nằm ở phân bố tôi không dạy, và loss không thể
thấy nó.

Niềm tin thứ ba, nhỏ hơn nhưng thực dụng: tôi từng tin **"cứ dùng `assistant_only_loss=True`
là mask đúng"**. `scripts/check_mask_agreement.py` cho thấy nó trả mask **0/31 token** — tức
train trên không gì cả, kèm một warning — và đường trainer còn **vá template** để phủ thêm
khối `think` rỗng (13/31), khác mask mà NB1 đã chứng minh (9/31).

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

**Việc tôi dùng nó:** đọc và tóm tắt `README.md`/`rubric.md` để lập danh sách artifact bắt
buộc; đọc `scripts/verify.py` để biết chính xác cổng kiểm tra gì (ví dụ: nó chỉ FAIL khi
`supervised_fraction ≥ 0.95`, và `smoke_mode=true` trong `baselines_frozen.json`); dò
`src/labkit/*` để hiểu `matched_rank()`, `resolve_target_modules()`, `training_epochs()`;
phân tích các file trong `results/` để dựng bảng; và dựng khung `REPORT.md`.
Nó cũng **phát hiện và sửa** hai thứ ở repo: `colab/Lab21_RUN_ALL.ipynb` đang clone repo
upstream thay vì repo của tôi, và `EVAL_LIMIT` mặc định là `"8"` — tức một **Colab RUN ALL
mặc định sẽ sinh ra kết quả không đủ điều kiện nộp**. Cả hai đã sửa, và tôi sửa luôn
`scripts/build_colab.py` (nguồn sinh ra 6 notebook kia) để `make colab` không tái tạo lại
lỗi đó.

**Chỗ nó sai — cụ thể.** Khi viết `scripts/qualitative_breakdown.py` để chấm từng field, nó
viết:

```python
fields_b = {k: ev.triage_field_accuracy(pb, {k: r["label"][k]}) for k in ev.TRIAGE_KEYS}
```

Tôi đọc `src/labkit/evaluate.py` mới thấy hàm này nhận `keys=None → keys = TRIAGE_KEYS`, rồi
với ba key không có trong dict nhãn rút gọn thì `continue` — nên một field **đúng** được
tính **0.25**, không phải 1.0. Bảng field-level sẽ sai một cách **âm thầm**: nó vẫn in ra
số, vẫn trông hợp lý, chỉ là sai hết. Sửa thành `keys=[k]` là xong, nhưng nếu tôi không đọc
source của scorer thì tôi đã tin nó.

Sai thứ hai, và nghiêm trọng hơn về mặt phương pháp: nó đề xuất lấy **3 mẫu có `ft_score`
thấp nhất** của nhóm target rồi dán nhãn "fine-tune thua". Số đo thật nói ngược lại —
decode lại cả arm (b) và (c) cho thấy **48 thắng / 2 hoà / 0 thua** và **0** ca thua ở mức
field. Gọi `ft_score = 0.75` là "thua" trong khi (b) chỉ được 0.50 ở đúng mẫu đó là **bịa
kết quả**. Ca thua có thật nằm ở nhóm **regression** (0/5/10), nên tôi chuyển sang lấy ca
thua từ đúng nhóm đó. Nghĩa là: AI hữu ích cho việc *đọc* và *dựng khung*, nhưng nó **áp
một khuôn mẫu quen thuộc lên số liệu mà chưa kiểm tra số liệu có khớp khuôn đó không** — và
đó đúng là loại lỗi mà lab này tồn tại để bắt.

Sai thứ ba, do vòng audit lại tìm ra: nó đã viết trong report một **bảng loss theo từng step**
(step 5/10/15/…/58) và gọi hai giá trị cuối là "loss tại step 58" / "final step loss". Cả hai
đều sai. `result.training_loss` mà NB3/NB4 ghi vào `runs.csv` là **run-level training loss tổng
hợp do `Trainer` trả về**, không được chứng minh là loss riêng của optimizer step cuối; và
bảng theo step **chỉ tồn tại trong log**, không có artifact nào trong `results/`. Nó cũng ghi
"tiết kiệm 0,79 GB VRAM" trong khi `runs.csv` ghi `correct=1.97`, `qlora=1.19` → chênh **0,78
GB**. Hai lỗi cùng một dạng: **trình bày một con số chặt chẽ hơn mức artifact cho phép**. Cả hai
đã sửa: bảng theo step bị bỏ khỏi phần chấm điểm và chỉ còn mô tả định tính có ghi rõ nguồn log,
còn mọi phần trăm giờ ghi kèm công thức tính từ `runs.csv`/`autopsy.json`.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

Tôi sẽ **không** mở notebook train. Tôi sẽ dựng **cổng hồi quy trước**, cùng với một tập
target nhỏ nhưng đo được — vì đó là thứ quyết định kết quả, không phải cấu hình LoRA. Cụ
thể, theo đúng thứ tự này:

1. **Hỏi khách "model này còn phải làm được gì nữa?"** và viết ra thành một tập eval thứ hai
   (như `eval_regression.jsonl`, 15 câu) **trước** khi có model. Trong lab này tôi chỉ mất
   0.611 ở nhóm đó mà vẫn phải kết luận FAILED — nếu tôi dựng cổng đó muộn hơn, tôi đã ship
   một adapter trả JSON triage cho mọi câu hỏi.
2. **Chốt mốc bằng prompt tốt nhất có thể, đo trước khi train, rồi đóng băng** (SHA prompt +
   checksum tập eval). Trong lab này (b) = 0.490 so với (a) = 0.000 — mốc "prompt tử tế"
   cách mốc "prompt ngây thơ" một trời một vực, nên nếu tôi chỉ so với (a) thì mọi thứ đều
   "thắng".
3. **Đo `p95` token trước khi chọn `max_length`** — ở đây p95 = 98 nhưng tier đặt 1024, tức
   gấp 10 lần mức cần; biết con số thật để quyết định chứ không đoán.
4. **Decode mask ngược ra text và đọc bằng mắt** (`decode_supervised`) trước khi train. Trên
   Qwen3.5, tin vào `assistant_only_loss=True` nghĩa là train trên **0 token**.
5. Và cuối cùng mới tới LR: **1e-4, không phải 1e-5** — vì tôi vừa đo được rằng con số đó
   một mình nó quyết định 65.5 điểm target.
