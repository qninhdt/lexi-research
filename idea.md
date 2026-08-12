- train non-thinking LLM -> SFT
- train thinking LLM -> GRPO + SFT; GRPO cho reasoning token và SFT cho answer 

Hai paper cùng giải một bài toán:

> Có cặp **input (x)** và **reference answer (y^*)**, nhưng không có reasoning label. Ta muốn model tự học một reasoning trace (z) giúp nó tạo ra (y^*).

* **JEPO**: *Beyond Verifiable Rewards: Scaling Reinforcement Learning for Language Models to Unverifiable Data* — arXiv:2503.19618.
* **NRT**: *Native Reasoning Models: Training Language Models to Reason on Unverifiable Data* — ICLR 2026, arXiv:2602.11549. ([arXiv][1])

Điểm quan trọng nhất:

[
\boxed{
\text{Reasoning tự sinh} \rightarrow \text{policy gradient}
}
]

[
\boxed{
\text{Reference answer} \rightarrow \text{teacher-forced token gradient}
}
]

Cả hai **không áp dụng RL loss lên answer được sinh như vanilla GRPO**.

---

# 1. Mô hình hóa chung

Gọi:

* (x): prompt/input.
* (z): reasoning do model tự sinh.
* (y^*): reference answer có trong dataset.

Model được xem như hai quá trình liên tiếp:

[
z\sim\pi_\theta(z\mid x)
]

[
y\sim\pi_\theta(y\mid x,z)
]

Với dữ liệu của bạn:

```text
x:
target = speak
definition = to talk to someone about something
pos = verb
text = She speak very well in the meeting yesterday.
```

```json
y*:
{
  "correction": "She [speak>spoke:tense] very well in the meeting yesterday.",
  "meaning": 4,
  "feedback": "Great use of the word, but the past tense is needed here."
}
```

Reasoning (z) không có trong dataset. Model phải tự khám phá, chẳng hạn:

```text
The target sense is used correctly.
“Yesterday” indicates past time, so “speak” should be “spoke”.
The meaning score should be 4.
```

Câu hỏi là:

> Làm thế nào biết reasoning này tốt, khi không có gold reasoning để so sánh?

Cả JEPO và NRT dùng cùng một nguyên tắc:

> Reasoning tốt là reasoning làm model dự đoán reference answer dễ hơn.

---

# 2. JEPO

## 2.1 Mục tiêu thật sự mà JEPO muốn tối ưu

Ta muốn tăng xác suất model tạo ra reference answer:

[
\max_\theta \log\pi_\theta(y^*\mid x)
]

Nhưng model có thể đi qua nhiều reasoning khác nhau:

[
\pi_\theta(y^*\mid x)
=====================

\mathbb E_{z\sim\pi_\theta(z\mid x)}
\left[
\pi_\theta(y^*\mid x,z)
\right]
]

Do đó:

[
\log\pi_\theta(y^*\mid x)
=========================

\log
\mathbb E_z
\left[
\pi_\theta(y^*\mid x,z)
\right]
]

Việc tính chính xác kỳ vọng trên mọi reasoning (z) là bất khả thi vì số reasoning có thể sinh ra là cực lớn.

JEPO dùng bất đẳng thức Jensen:

[
\log\mathbb E_z[p_z]
\ge
\mathbb E_z[\log p_z]
]

để tạo lower bound:

[
\mathcal L_{\text{JEPO}}
========================

\mathbb E_{z\sim\pi_\theta(z\mid x)}
\left[
\log\pi_\theta(y^*\mid x,z)
\right]
]

Nói đơn giản:

1. Model sample reasoning.
2. Đặt gold answer sau reasoning.
3. Đo log-likelihood của gold answer.
4. Tăng xác suất những reasoning làm gold answer có likelihood cao.

Đây là lý do tên của nó có “Jensen’s Evidence Lower Bound”. ([arXiv][1])

---

## 2.2 Một training step của JEPO

Giả sử model sample hai reasoning.

### Reasoning A

```text
“Yesterday” indicates past tense.
The intended meaning of “speak” is correct.
```

### Reasoning B

```text
“Speak” must always be followed by “to”.
The meaning is only partially correct.
```

Sau đó cùng một gold answer được teacher-force sau mỗi reasoning:

```text
x + reasoning A + gold answer
x + reasoning B + gold answer
```

Giả sử model tính được:

[
\log\pi_\theta(y^*\mid x,z_A)=-10
]

[
\log\pi_\theta(y^*\mid x,z_B)=-30
]

Giá trị âm ít hơn là tốt hơn:

* Reasoning A làm gold answer dễ dự đoán.
* Reasoning B làm gold answer khó dự đoán.

JEPO dùng các giá trị này làm reward cho reasoning.

Các con số trên chỉ là ví dụ minh họa.

---

## 2.3 Hai gradient trong JEPO

Gradient single-sample của JEPO có dạng:

[
\nabla J
========

\underbrace{
\left(
\log\pi_\theta(y^*\mid x,z)-b
\right)
\nabla_\theta\log\pi_\theta(z\mid x)
}*{\text{policy gradient cho reasoning}}
+
\underbrace{
\nabla*\theta\log\pi_\theta(y^*\mid x,z)
}_{\text{supervised gradient cho answer}}
]

### Thành phần 1: reasoning policy gradient

[
\left(
\log\pi_\theta(y^*\mid x,z)-b
\right)
\nabla_\theta\log\pi_\theta(z\mid x)
]

Nó có nghĩa:

* Reasoning làm reference answer có likelihood cao hơn baseline → tăng xác suất reasoning.
* Reasoning làm reference answer có likelihood thấp hơn baseline → giảm xác suất reasoning.

Đây là phần RL.

RL gradient chỉ chạy qua các token trong reasoning (z).

### Thành phần 2: answer likelihood gradient

[
\nabla_\theta\log\pi_\theta(y^*\mid x,z)
]

Đây chính là teacher-forced language-model training:

```text
context: x + sampled reasoning + gold-answer prefix
target:  gold-answer token tiếp theo
```

Nó dạy model sinh từng token của gold answer.

JEPO mô tả rõ đây là một objective kết hợp policy-gradient cho chain-of-thought và supervised likelihood cho reference answer. ([arXiv][1])

---

## 2.4 Mask loss của JEPO

Về mặt ý niệm:

```text
Sequence                     RL mask    Supervised mask

prompt x                        0               0
sampled reasoning z             1               0
reference answer y*             0               1
```

Generated answer không nhận policy-gradient trong core JEPO objective.

Paper có thể generate cả completion rồi trích phần chain-of-thought, nhưng sau đó nó thực hiện forward pass riêng để tính:

[
\pi_\theta(y^*\mid x,z)
]

và backpropagate riêng qua:

* `log π(z | x)`
* `log π(y* | x,z)`

Điều này được thể hiện trực tiếp trong Algorithm 1 của paper. ([arXiv][1])

---

## 2.5 Multi-sample JEPO

Single-sample JEPO dùng một reasoning:

[
\mathbb E_z[\log p(y^*\mid x,z)]
]

Paper còn đưa ra multi-sample lower bound. Sample (K) reasoning:

[
z_1,z_2,\ldots,z_K
]

Rồi tối ưu:

[
\log
\left[
\frac{1}{K}
\sum_{k=1}^{K}
\pi_\theta(y^*\mid x,z_k)
\right]
]

Ý tưởng:

> Không yêu cầu mọi reasoning đều tốt. Chỉ cần trong nhóm có những reasoning làm reference answer rất có khả năng xảy ra.

Khi (K) tăng, lower bound này tiến gần hơn tới marginal likelihood thật:

[
\log\pi_\theta(y^*\mid x)
]

Paper báo cáo multi-sample JEPO thường tốt hơn single-sample trong các thí nghiệm của họ. ([arXiv][1])

---

## 2.6 Điểm mạnh của JEPO

JEPO không cần:

* Exact-match verifier.
* Unit test.
* Reward model.
* LLM judge.
* Gold reasoning.
* Teacher sinh reasoning.

Nó chỉ cần:

```text
input x
reference answer y*
```

Paper thử trên dữ liệu có short-form answer, semi-verifiable answer và long-form mathematical proof; tác giả báo cáo JEPO cạnh tranh với RLVR trên dữ liệu có verifier và vượt một số baseline SFT/RL trên dữ liệu khó xác minh. ([arXiv][1])

---

## 2.7 Hạn chế của JEPO

### Toàn bộ answer bị nén thành một sequence log-likelihood

JEPO dùng:

[
\log\pi(y^*\mid x,z)
====================

\sum_{i=1}^{T}
\log\pi(y_i^*\mid x,z,y^*_{<i})
]

Tất cả token đều đóng góp.

Trong dữ liệu của bạn, điều đó bao gồm:

```text
Correction decision:
spoke
tense

Meaning decision:
4

Feedback wording:
Great use of the word, but...

Formatting:
quotes, braces, commas...
```

Vì vậy reasoning có thể được reward không chỉ vì quyết định `correction` và `meaning`, mà còn vì nó giúp model đoán giọng văn hoặc boilerplate của `feedback`.

Đây chính là điểm NRT cố gắng tổng quát hóa.

### Độ dài answer ảnh hưởng reward

Tổng log-probability thường âm hơn khi answer dài hơn. JEPO chủ ý sử dụng sequence-level log-probability không normalize theo độ dài, vì objective của họ là tăng probability của toàn sequence. ([arXiv][1])

Trong một dataset có answer length dao động mạnh, điều này có thể khiến scale reward giữa các sample khác nhau đáng kể.

---

# 3. NRT

NRT giữ nguyên latent-reasoning setup:

[
z\sim\pi_\theta(z\mid x)
]

Nhưng thay vì bắt buộc dùng log-likelihood của toàn answer, NRT định nghĩa reward tổng quát từ xác suất của từng gold token.

## 3.1 Xác suất từng reference token

Reference answer:

[
y^*=(y_1^*,y_2^*,\ldots,y_T^*)
]

Sau reasoning (z), tính:

[
c_i(z,\theta)
=============

\pi_\theta
\left(
y_i^*
\mid
x,z,y^*_{<i}
\right)
]

Ví dụ:

```text
Token gold                         Probability sau reasoning A

"spoke"                            0.91
"tense"                            0.87
meaning = 4                        0.95
"Great"                            0.58
"use"                              0.75
...
```

Sau đó NRT dùng một hàm aggregation (f):

[
R(z,\theta)=f(c_1,c_2,\ldots,c_T)
]

Objective:

[
J(\theta)
=========

\mathbb E_{z\sim\pi_\theta(z\mid x)}
[R(z,\theta)]
]

Như vậy NRT hỏi:

> Ta nên kết hợp probability của các gold token thế nào để đánh giá reasoning?

Đây là phần tổng quát hơn JEPO. ([arXiv][2])

---

## 3.2 Gradient NRT cũng có hai phần

Gradient tổng quát của NRT:

[
\nabla J
\approx
\underbrace{
R(z,\theta)
\nabla\log\pi_\theta(z\mid x)
}*{\text{RL cho reasoning}}
+
\underbrace{
\sum_i
\alpha_i c_i
\nabla\log
\pi*\theta(y_i^*\mid x,z,y^**{<i})
}*{\text{weighted token prediction}}
]

Trong đó:

[
\alpha_i=\frac{\partial f}{\partial c_i}
]

Tương tự JEPO:

* Term đầu cập nhật reasoning.
* Term sau cập nhật việc dự đoán reference-answer tokens.

Nhưng NRT cho phép mỗi gold token nhận trọng số khác nhau. ([arXiv][2])

---

# 4. Các biến thể reward của NRT

## 4.1 Sequence Log-Probability

Chọn:

[
f(c_1,\ldots,c_T)
=================

\sum_i\log c_i
]

Khi đó:

[
R(z,\theta)
===========

\log\pi_\theta(y^*\mid x,z)
]

Và token reward signal bằng 1 cho mọi gold token.

Do đó:

[
\boxed{
\text{NRT Sequence LogP}
\equiv
\text{single-sample JEPO/JLB về core objective}
}
]

Đây là nguồn gốc của việc hai paper nhìn gần như giống hệt nhau.

NRT gọi implementation của prior work này là **JLB** trong bảng so sánh. 

---

## 4.2 Sequence Probability

Chọn:

[
f(c)=\prod_i c_i
================

\pi_\theta(y^*\mid x,z)
]

Với answer dài, tích của hàng trăm probability cực kỳ nhỏ:

```text
0.5 × 0.7 × 0.8 × ... ≈ gần 0
```

Reward và gradient dễ bị vanishing.

Đó là lý do sequence probability không lý tưởng cho long-form output.

---

## 4.3 Arithmetic Mean

[
f(c)=\frac{1}{T}\sum_i c_i
]

Ví dụ:

```text
100 token dễ: probability ≈ 0.99
3 token quyết định: probability ≈ 0.30
```

Arithmetic mean vẫn rất cao vì các token dễ chiếm số lượng lớn.

Model có thể nhận reward tốt dù reasoning không giúp giải quyết những token khó. NRT cho rằng kiểu reward này có thể bị các token dễ chi phối và góp phần gây policy collapse hoặc null reasoning. 

---

## 4.4 Geometric Mean

[
f(c)
====

\left(
\prod_i c_i
\right)^{1/T}
]

Tương đương:

[
\exp
\left(
\frac{1}{T}
\sum_i\log c_i
\right)
]

Nó normalize theo length và nhạy với token có probability thấp.

Ví dụ một token quan trọng có xác suất gần 0 sẽ kéo reward xuống rõ rệt. Vì vậy model phải cải thiện cả các token khó thay vì chỉ dựa vào nhiều token dễ.

---

## 4.5 Weighted Sum

[
f(c)=\sum_i w_i c_i
]

NRT đề xuất đặt trọng số cao cho token mà base model đang không chắc chắn.

Hai cách paper thử:

### Inverse probability

[
w_i\propto\frac{1}{c_{i,\text{base}}}
]

Token có baseline probability thấp nhận weight cao.

### Negative log-probability

[
w_i\propto-\log c_{i,\text{base}}
]

Token càng khó với base model thì weight càng lớn.

Ý tưởng:

> Reasoning nên tập trung giải quyết những token mà model không thể dự đoán tốt nếu không reasoning.

NRT báo cáo weighted-sum, đặc biệt biến thể (-\log p), là biến thể mạnh nhất trung bình trong thí nghiệm của họ. ([arXiv][2])

---

# 5. NRT ổn định RL thế nào?

## 5.1 Empty-reasoning baseline

NRT tính reward khi không có reasoning:

[
R_{\text{base}}
===============

R(z=\emptyset)
]

Sau đó chỉ giữ phần reasoning giúp tốt hơn baseline:

[
R'_k
====

\max
\left(
0,
R(z_k)-R_{\text{base}}
\right)
]

Ví dụ:

```text
No thinking reward:  0.72

Reasoning A reward:  0.84 → improvement 0.12
Reasoning B reward:  0.68 → improvement 0
Reasoning C reward:  0.76 → improvement 0.04
```

Mục đích:

> Không reward một reasoning chỉ vì model vốn đã biết answer; reasoning phải tạo thêm giá trị so với direct answer.

---

## 5.2 Group-relative normalization

Với (K) reasoning cho cùng prompt, NRT normalize reward trong group:

[
A_k
===

\frac{R'_k-\operatorname{mean}(R')}
{\operatorname{std}(R')}
]

Reasoning tốt hơn các reasoning khác nhận positive advantage; reasoning kém hơn nhận negative advantage.

Đây là machinery tương tự GRPO, nhưng reward không đến từ verifier. Reward đến từ reference-token probabilities. ([arXiv][2])

---

## 5.3 Format supervision

NRT thêm CE nhỏ chỉ cho token đánh dấu:

```text
<think>
</think>
```

Mục đích là phân biệt reasoning và final answer, không phải dùng format làm semantic reward. Paper dùng format-supervision loss riêng thay vì coi format là core reasoning reward. ([arXiv][2])

---

# 6. JEPO và NRT giống nhau ở đâu?

| Thành phần               | JEPO                       | NRT                                      |
| ------------------------ | -------------------------- | ---------------------------------------- |
| Dữ liệu cần              | (x,y^*)                    | (x,y^*)                                  |
| Gold reasoning           | Không                      | Không                                    |
| External verifier        | Không                      | Không                                    |
| Model tự sinh reasoning  | Có                         | Có                                       |
| RL áp dụng lên           | Reasoning                  | Reasoning                                |
| Answer được học bằng     | Teacher-forced likelihood  | Weighted teacher-forced token prediction |
| Generated answer nhận RL | Không trong core objective | Không trong core objective               |
| Reasoning reward         | Gold-answer log-likelihood | Hàm (f) trên gold-token probabilities    |

---

# 7. JEPO và NRT khác nhau ở đâu?

## Khác biệt 1: reward definition

JEPO mặc định:

[
R(z)
====

\sum_i\log c_i
]

NRT:

[
R(z)
====

f(c_1,\ldots,c_T)
]

NRT cho phép:

* Sequence log-probability.
* Geometric mean.
* Arithmetic mean.
* Weighted sum.
* Các weighting scheme theo độ khó token.

## Khác biệt 2: answer-token gradient

JEPO single-sample:

[
\sum_i
\nabla\log c_i
]

Mọi reference token nhận standard CE weight bằng nhau.

NRT tổng quát:

[
\sum_i
s_i
\nabla\log c_i
]

với (s_i) phụ thuộc vào aggregation function.

Ví dụ:

* Sequence LogP: (s_i=1), giống SFT.
* Weighted Sum: (s_i=w_ic_i), không còn là standard full-answer SFT.

## Khác biệt 3: stabilization

NRT nhấn mạnh:

* Baseline reasoning rỗng.
* Clip phần improvement.
* Group-normalized advantage.
* Explicit format supervision.
* Off-policy importance ratio.

JEPO tập trung hơn vào:

* Jensen lower bound.
* Multi-sample tighter bound.
* Kết nối với latent-variable inference.
* KL/reference policy và practical RL integration. ([arXiv][1])

## Khác biệt 4: JEPO multi-sample không hoàn toàn là NRT-logP

Single-sample JEPO và NRT Sequence LogP có cùng core objective:

[
\mathbb E_z[\log p(y^*\mid x,z)]
]

Nhưng multi-sample JEPO tối ưu:

[
\log
\left(
\frac1K
\sum_k p(y^*\mid x,z_k)
\right)
]

Đó là một tightened bound khác, không nên đồng nhất toàn bộ JEPO với mọi biến thể NRT.

---

# 8. Áp dụng vào dataset của bạn

Bạn có ba loại output:

```text
correction
meaning
feedback
```

## Phương án đúng nguyên bản JEPO

Reward reasoning:

[
R_{\text{JEPO}}(z)
==================

\log p(
\text{correction}+
\text{meaning}+
\text{feedback}
\mid x,z)
]

Answer loss:

[
L_{\text{answer}}
=================

-\log p(
\text{correction}+
\text{meaning}+
\text{feedback}
\mid x,z)
]

Ưu điểm:

* Đúng nguyên paper.
* Dễ định nghĩa.
* Toàn bộ gold answer nhận standard CE.

Nhược điểm:

* Feedback wording ảnh hưởng reasoning reward.
* Boilerplate và format cũng ảnh hưởng reward.
* Answer dài có thể làm reward scale khó xử lý.

## Phương án đúng nguyên bản NRT-WS

Đặt weight lớn cho token khó hoặc token thuộc `correction` và `meaning`:

[
R(z)
====

\sum_i w_i c_i
]

Nhưng cần nhớ:

> Trong NRT nguyên bản, cùng hàm (f) cũng quyết định token-prediction gradient.

Nếu đặt:

```text
feedback weight = 0
```

thì feedback cũng gần như không nhận token-prediction gradient từ NRT objective.

Điều này không đúng với mục tiêu của bạn vì bạn vẫn muốn SFT feedback đầy đủ.

## Phương án phù hợp nhất với yêu cầu của bạn

Tách hai objective:

[
L_{\text{total}}
================

L_{\text{SFT-full-answer}}
+
\lambda L_{\text{RL-reasoning}}
]

Trong đó:

[
L_{\text{SFT-full-answer}}
==========================

CE(\text{correction}+\text{meaning}+\text{feedback})
]

Nhưng reasoning reward chỉ dùng:

[
R(z)
====

R_{\text{correction likelihood}}
+
R_{\text{meaning likelihood}}
]

Mask:

```text
Part                         RL                SFT

sampled reasoning            yes               no
gold correction              no                yes
gold meaning                 no                yes
gold feedback                no                yes
```

Đây là:

> **NRT/JEPO-style latent reasoning RL với task-specific reward mask và auxiliary full-answer SFT.**

Nó không phải cấu hình nguyên xi của một trong hai paper, nhưng là adaptation trực tiếp và hợp lý cho schema của bạn.

---

# 9. Cách hiểu trực quan nhất

Hãy tưởng tượng model có hai “vai”, dù thực tế dùng chung weights.

### Vai 1: người phân tích

```text
input → reasoning
```

Người phân tích không có đáp án reasoning mẫu. Nó thử nhiều suy luận.

### Vai 2: người viết answer

```text
input + reasoning → gold answer
```

Ta kiểm tra:

> Với reasoning nào, người viết dự đoán gold correction và meaning dễ nhất?

Reasoning có ích được tăng xác suất bằng RL.

Đồng thời người viết vẫn được dạy từng gold token bằng SFT.

```text
Reasoning tốt hơn
       ↓
Gold answer dễ dự đoán hơn
       ↓
Reward reasoning cao hơn
       ↓
Reasoning đó xuất hiện nhiều hơn
```

Đó là toàn bộ trực giác của JEPO và NRT.

**JEPO chọn một cách đo cố định:** log-likelihood của toàn answer.

**NRT hỏi thêm:** nên tổng hợp xác suất từng token thế nào để reasoning tập trung đúng chỗ?

[1]: https://arxiv.org/html/2503.19618v2 "Beyond Verifiable Rewards: Scaling Reinforcement Learning for Language Models to Unverifiable Data"
[2]: https://arxiv.org/html/2602.11549v2 "Native Reasoning Models: Training Language Models to Reason on Unverifiable Data Code is available at https://github.com/sharkwyf/native-reasoning-models"
