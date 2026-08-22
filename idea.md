Được. Tôi sẽ chốt thành **một spec duy nhất** để có thể giao thẳng cho AI engineer implement. Tôi cũng điều chỉnh một điểm quan trọng sau khi đọc docs hiện tại: **không nên concat nguyên full reasoning history để SFT Qwen3.5**. Qwen khuyến nghị trong multi-turn history chỉ giữ final output/action của các turn trước, không giữ thinking cũ. ([Hugging Face][1])

# Project: Specialized Business Assistant via SFT → Agentic RL

## 0. Mục tiêu cuối cùng

Xây một business/customer-service agent từ general reasoning LLM, chứng minh:

[
\boxed{\text{Base} < \text{SFT} < \text{SFT + Agentic RL}}
]

trên **held-out τ³-bench Retail test set**.

Pipeline:

```text
Qwen3.5-2B
     │
     ├───────────────→ Base τ³ Retail test
     │
     ▼
tau-bench-synthetic
successful reasoning trajectories
     │
     ▼
LoRA SFT
     │
     ├───────────────→ SFT τ³ Retail test
     │
     ▼
τ³ official Retail train split
     │
     ▼
online multi-turn rollout
agent ↔ user simulator ↔ tools ↔ DB
     │
     ▼
τ official verifier/reward
     │
     ▼
TRL GRPOTrainer
     │
     ▼
LoRA Agentic RL
     │
     └───────────────→ SFT+RL τ³ Retail test
```

**Không làm** voice, banking/RAG, browser, Python sandbox, SWE-bench, code execution.

τ³ current repo vẫn nằm ở `sierra-research/tau2-bench`; release hiện tại là **v1.0.1**. Core text domains vẫn gồm Retail/Airline/Telecom, còn voice/knowledge là extensions riêng. ([GitHub][2])

---

# 1. Hardware

Target duy nhất:

```text
Google Colab Pro
1 × NVIDIA L4 24GB
```

Model chính:

```text
Qwen/Qwen3.5-2B
```

Không cần thuê H100/A100 để hoàn thành project.

Không full fine-tune.

Dùng:

```text
BF16 base
+
LoRA
+
gradient checkpointing
+
batch vật lý = 1
+
vLLM colocate cho RL
```

Qwen định vị chính Qwen3.5-2B cho prototyping/task-specific fine-tuning. Public model card hiện báo TAU2-Bench **48.8** cho 2B và **11.6** cho 0.8B ở thinking mode, nên 2B là sweet spot tốt hơn để RL có reward variance. ([Hugging Face][1])

---

# 2. Repositories / dependencies

## Repo cần clone

### A. Benchmark/environment

```bash
git clone --branch v1.0.1 \
  https://github.com/sierra-research/tau2-bench.git \
  third_party/tau2-bench
```

Không clone latest `main` mà không pin version.

τ³ yêu cầu Python `>=3.12,<3.14` và hiện dùng `uv`; text core install rất gọn, Gym là optional extra. ([GitHub][2])

Setup:

```bash
cd third_party/tau2-bench
uv sync --extra gym
```

Không install:

```text
voice
knowledge
all-extras
```

τ³ đã cung cấp `AgentGymEnv` chính thức cho RL và train/test splits cho các domain. ([GitHub][3])

---

## Hugging Face resources

Model:

```text
Qwen/Qwen3.5-2B
```

SFT dataset:

```text
fuvty/tau-bench-synthetic
```

Dataset community này được tạo riêng để train small LLM mà **không dùng original τ evaluation set**. Nó có:

```text
280 synthetic tasks
1,464 full trajectories
~1,200 successful trajectories
4,270 derived SFT turns
Retail + Airline
```

Full trajectories có reasoning; bản 4,270 SFT-ready rows đã strip reasoning. ([Hugging Face][4])

**Ta dùng full trajectories, không dùng trực tiếp SFT-ready subset.**

---

## Training dependencies

Tạo một root `pyproject.toml`/lockfile. Minimum:

```text
torch
transformers >= 5.2
trl[vllm]
peft
datasets
accelerate
wandb
bitsandbytes
rich
pytest
```

`bitsandbytes` chỉ để fallback QLoRA, **không mặc định dùng NF4**.

TRL hiện hỗ trợ PEFT SFT, assistant-only loss và vLLM colocate. `environment_factory` cần Transformers >=5.2; custom `rollout_func` cũng được hỗ trợ. ([Hugging Face][5])

Sau khi một Colab environment chạy thành công:

> **freeze toàn bộ package versions thành lockfile.**

Không để engineer cứ `pip install -U` mỗi run.

---

# 3. Repository structure nên implement

```text
tau-agent-posttraining/
│
├── configs/
│   ├── sft.yaml
│   ├── grpo.yaml
│   ├── eval.yaml
│   └── smoke.yaml
│
├── src/
│   ├── data/
│   │   ├── prepare_sft.py
│   │   ├── profile_lengths.py
│   │   ├── build_splits.py
│   │   └── validate_dataset.py
│   │
│   ├── training/
│   │   ├── train_sft.py
│   │   ├── train_grpo.py
│   │   └── merge_adapter.py
│   │
│   ├── tau/
│   │   ├── rollout.py
│   │   ├── action_parser.py
│   │   ├── reward.py
│   │   └── user_simulator.py
│   │
│   ├── evaluation/
│   │   ├── evaluate_tau.py
│   │   ├── metrics.py
│   │   └── error_analysis.py
│   │
│   └── logging/
│       └── wandb_callbacks.py
│
├── tests/
│   ├── test_chat_template.py
│   ├── test_loss_mask.py
│   ├── test_tau_rollout.py
│   ├── test_reward.py
│   ├── test_no_test_leakage.py
│   └── test_tool_parser.py
│
├── third_party/
│   └── tau2-bench/        # pinned v1.0.1
│
├── artifacts/
│   ├── splits/
│   └── evaluation/
│
├── scripts/
│   ├── setup_colab.sh
│   ├── smoke_test.sh
│   ├── run_sft.sh
│   ├── run_grpo.sh
│   └── run_final_eval.sh
│
└── README.md
```

---

# 4. Domain: chỉ Retail ở version 1

**Không làm Airline ngay.**

Scope v1:

```text
domain = retail
modality = text
communication = half-duplex
```

Retail gần nhất với traditional website assistant:

```text
cancel order
return product
exchange item
change delivery address
change payment
look up order
policy/refusal
customer information
```

Nếu Retail thành công mới thêm Airline như extension.

---

# 5. Dataset cho SFT

Use:

```python
load_dataset("fuvty/tau-bench-synthetic", ...)
```

Filter:

```text
domain == retail
reward == 1.0
normal termination
```

Dataset tạo task theo GT-first: lấy DB state → programmatically tạo expected actions → GLM-5 tạo user scenario → GLM-5 reasoning agent chạy trong environment; chỉ successful trajectories được dùng cho SFT-ready data. ([Hugging Face][4])

---

# 6. Cách preprocess SFT — phần rất quan trọng

**Không train một full trajectory thành một target 10k token.**

Giả sử:

```text
system
user
assistant think1 + tool_call1
tool result1
assistant think2 + tool_call2
tool result2
assistant think3 + final
```

Tạo thành **3 training examples**.

### Example 1

```text
PROMPT:
system
user

TARGET:
think1
tool_call1
```

### Example 2

```text
PROMPT:
system
user
assistant tool_call1       ← previous thinking removed
tool result1

TARGET:
think2
tool_call2
```

### Example 3

```text
PROMPT:
system
user
assistant tool_call1
tool result1
assistant tool_call2
tool result2

TARGET:
think3
final answer
```

Đây đặc biệt phù hợp Qwen3.5 vì Qwen nói **thinking content của previous turns không nên nằm trong multi-turn history**; chat template chính thức cũng thực hiện behavior đó. ([Hugging Face][1])

Nó đồng thời:

* giảm context;
* tránh train lại thinking của turn cũ;
* tăng số supervised decision points;
* phù hợp inference behavior của Qwen3.5.

---

# 7. Không tự handcraft ChatML

Luôn dùng tokenizer/chat template chính thức:

```python
tokenizer.apply_chat_template(...)
```

Thinking:

```text
enable_thinking=True
```

Qwen3.5 mặc định non-thinking; thinking phải được enable bằng chat-template/API parameter. ([Hugging Face][1])

Tool schemas cũng phải đi qua model-native tool format.

**Không manually tạo**:

```text
<|im_start|>
<think>
...
```

trừ khi tokenizer API thực sự trả format đó.

---

# 8. SFT loss mask

Mục tiêu:

```text
system                 MASK
user                   MASK
previous assistant     MASK
tool results            MASK

CURRENT assistant:
reasoning              LOSS
tool call / answer      LOSS
```

TRL hỗ trợ `assistant_only_loss=True`; Qwen family được hỗ trợ template masking. ([Hugging Face][5])

Nhưng vì ta split theo turn, tôi thích dùng **conversational prompt-completion format**:

```python
{
    "prompt": [...history...],
    "completion": [current_assistant_message],
}
```

rồi:

```text
completion_only_loss = True
```

và verify bằng unit test rằng:

```text
labels == -100
```

cho:

* system;
* user;
* previous assistant;
* tool results.

---

# 9. SFT train/val split

**Split theo `task_id`, không split random theo individual turns.**

Sai:

```text
task X round 1 → train
task X round 2 → val
```

Đó là leakage.

Đúng:

```text
90% synthetic task IDs → SFT train
10% synthetic task IDs → SFT val
```

Mọi trajectory/round của cùng task nằm cùng một side.

Seed cố định:

```text
seed = 42
```

Save:

```text
artifacts/splits/sft_train_task_ids.json
artifacts/splits/sft_val_task_ids.json
```

---

# 10. Context-length policy

Trước train chạy:

```text
profile_lengths.py
```

Log histogram:

```text
prompt tokens
target tokens
total tokens
P50
P90
P95
P99
max
```

Target ban đầu:

```text
max_seq_length = 4096
```

Nếu:

```text
> 5% samples exceed 4096
```

thì thử:

```text
6144
```

rồi tối đa ban đầu:

```text
8192
```

Không tự động truncate tool schema/policy hay final action nếu có thể tránh.

**Goal của project là 4k-ish, không phải long-context training.**

---

# 11. SFT config ban đầu

```yaml
model: Qwen/Qwen3.5-2B

precision:
  bf16: true

max_length: 4096

lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: all-linear

training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16

  learning_rate: 1.0e-4
  lr_scheduler: cosine

  warmup_ratio: 0.05

  num_train_epochs: 1

  gradient_checkpointing: true
  max_grad_norm: 1.0

  packing: false

  seed: 42
```

TRL docs cũng gợi ý khoảng `1e-4` cho adapter training. ([Hugging Face][5])

### Packing

Ban đầu:

```text
packing = false
```

vì dataset multi-turn + tool formatting và muốn debug mask dễ.

Sau khi xác nhận 100% đúng:

```text
packing = true
```

nếu muốn throughput.

---

# 12. SFT checkpoints

Save:

```text
checkpoint-step-N
best-val-loss
final-sft-adapter
```

Sau SFT:

```text
base Qwen
+
SFT LoRA
→ merge
```

tạo:

```text
qwen3.5-2b-tau-retail-sft-merged
```

Lý do merge trước RL:

```text
SFT behavior trở thành RL starting policy
+
RL dùng một LoRA adapter mới
```

Vậy ta có clean decomposition:

```text
Base
   ↓
SFT merged base
   ↓
RL LoRA
```

và dễ rollback/ablation.

---

# 13. Baseline evaluation

Phải evaluate **trước khi SFT**.

Model:

```text
Qwen/Qwen3.5-2B
thinking = ON
```

Dataset:

```text
τ³ v1.0.1
domain = retail
split = test
```

Không dùng:

```text
base split
```

làm primary final evaluation, vì `base` chứa toàn bộ original tasks, trong khi official train/test split được thêm để RL training/evaluation tách biệt. ([GitHub][3])

---

# 14. User simulator

τ conversational mode có:

```text
User simulator
       ↕
your agent
       ↕
tools/database
```

`AgentGymEnv` hỗ trợ normal conversational mode bằng `solo_mode=False` và configurable `user_llm`. ([GitHub][6])

**Final project nên dùng normal mode.**

Không dùng `solo_mode=True` làm main result vì khi đó:

```text
không còn customer conversation
```

và câu chuyện “business assistant” yếu đi rõ rệt.

Chọn **một user simulator duy nhất và freeze nó cho toàn bộ experiment**:

```text
Base eval
SFT eval
RL rollouts
SFT+RL eval
```

Official τ leaderboard hiện khuyến nghị `gpt-5.2` cho user simulator. ([GitHub][7])

Nếu API budget là constraint, có thể thay bằng một fixed cheaper simulator, nhưng lúc đó README phải ghi rõ protocol là custom.

**Không thay user simulator giữa các checkpoints.**

---

# 15. Evaluation decoding

Giữ **cùng config cho Base / SFT / SFT+RL**.

Ví dụ:

```yaml
thinking: true

temperature: 0.6
top_p: 0.95
top_k: 20

max_generated_tokens_per_turn: 1024

max_agent_turns: 8
```

Qwen3.5-2B có warning chính thức rằng thinking mode dễ rơi vào thinking loops, nên cần hard cap và log truncation. ([Hugging Face][1])

Không dùng 32k/80k reasoning chỉ vì model hỗ trợ.

---

# 16. Primary evaluation metric

## Primary

```text
τ Retail held-out test Pass^1 / task success rate
```

Final report:

```text
Base
SFT
SFT + RL
```

với **cùng task IDs, user simulator, decoding config và benchmark commit**.

Final evaluation nên chạy:

```text
4 trials / task
```

Official leaderboard cũng strongly prefers >=4 trials vì tính stochastic của user simulator. ([GitHub][7])

Development evaluations có thể:

```text
1 trial/task
```

để tiết kiệm.

---

# 17. Secondary evaluation metrics

Mỗi checkpoint phải report:

```text
overall reward
DB success
COMMUNICATE success
```

Retail default reward được thiết kế outcome-based:

[
R = R_{DB}\times R_{COMMUNICATE}
]

Reference actions chỉ được replay để tạo target final DB state; agent **không bắt buộc imitate đúng tool trajectory**. ([GitHub][8])

Ngoài ra log:

```text
partial_action_reward        # diagnostic only

invalid_tool_call_rate
tool_argument_error_rate

avg_tool_calls
avg_agent_turns

avg_reasoning_tokens
P95 reasoning tokens

episode_truncation_rate
premature_termination_rate

success_by_task_type
```

`partial_action_reward` **không được dùng làm final success metric**, vì một alternative valid trajectory vẫn có thể đúng dù khác reference. ([GitHub][8])

---

# 18. Current τ reward caveat

Có một open issue hiện tại chỉ ra một số Retail/Airline tasks vẫn có reward gaps/no-op false positives. ([GitHub][9])

Do đó engineer phải:

1. pin v1.0.1;
2. không tự sửa official evaluator giữa experiment;
3. log DB, COMMUNICATE và partial action separately;
4. tạo secondary report `retail_clean_diagnostic` loại đúng những task IDs được public issue liệt kê;
5. **primary metric vẫn là full official test split**.

Với RL train set, nếu những known-broken task IDs xuất hiện:

> exclude chúng khỏi RL training trước khi bắt đầu experiment.

Không train policy để exploit một evaluator bug đã biết.

---

# 19. Agentic RL dataset

Đây là điểm khác SFT.

Không dùng synthetic successful trajectories nữa.

Use:

```text
official τ³ Retail train split
```

TRL sẽ online-generate trajectories từ chính SFT checkpoint.

Flow:

```text
official train task
       ↓
SFT policy
       ↓
user simulator
       ↓
assistant response/tool call
       ↓
Tau Gym
       ↓
tool result / DB mutation
       ↓
user simulator
       ↓
...
       ↓
episode finished
       ↓
official Tau reward
       ↓
GRPO update
```

Đây mới thật sự là:

[
\boxed{\text{Agentic RL}}
]

chứ không phải offline training trên teacher trajectory.

---

# 20. TRL integration: dùng `rollout_func`, không ép `environment_factory`

Đây là quyết định implementation quan trọng.

Tau conversational Gym có cả:

```text
assistant → plain user message
assistant → tool call
```

và sau plain assistant message, user simulator có thể reply rồi episode tiếp tục.

TRL `environment_factory` được thiết kế tốt nhất cho model→tool→observation loops; trainer tự điều khiển tool-call loop. TRL cung cấp `rollout_func` chính thức khi cần tự quản lý interaction loop. ([Hugging Face][10])

Vì vậy:

```text
Tau AgentGymEnv
       ↓
custom Tau rollout_func
       ↓
TRL GRPOTrainer
```

**Không viết RL optimizer.**

Chỉ viết rollout adapter.

`rollout_func` phải trả:

```text
prompt_ids
completion_ids
logprobs
```

và có thể trả thêm metadata để reward/logger dùng. API này hiện được TRL hỗ trợ nhưng đánh dấu experimental, nên cần pin version sau khi smoke-test. ([Hugging Face][11])

---

# 21. Rollout behavior

Pseudo-flow:

```python
env = AgentGymEnv(
    domain="retail",
    task_id=task_id,
    solo_mode=False,
    user_llm=FIXED_USER_SIMULATOR,
)

obs, info = env.reset()

history = build_initial_history(
    policy=info["policy"],
    tools=info["tools"],
    observation=obs,
)

while not done:
    assistant_output = policy.generate(
        history,
        thinking=True,
    )

    action = parse_assistant_action(assistant_output)

    obs, reward, terminated, truncated, info = env.step(action)

    # Qwen3.5 best practice:
    # DO NOT persist previous thinking content
    history.append(strip_thinking_keep_final_action(assistant_output))
    history.append(obs)

final_reward = reward
```

Important:

> completion tokens used for policy loss include model-generated reasoning/actions.

But:

```text
user messages
tool observations
system prompt
```

remain context only.

---

# 22. RL reward v1

**Do not invent dense reward initially.**

Use:

```text
reward = official_tau_reward
```

i.e. binary task outcome.

Why?

Because it gives the cleanest experiment:

```text
SFT optimizes teacher imitation
vs
RL optimizes actual task success
```

Recent work specifically on Tau-Bench found that naïve per-turn dense rewards can actually degrade performance because of bad credit assignment. ([arXiv][12])

So v1:

[
R\in{0,1}
]

No:

```text
+0.1 for read tool
+0.2 for valid JSON
+0.5 for every expected action
```

unless sparse reward fails experimentally.

---

# 23. Difficulty filtering trước RL

Đây là optimization đáng làm trên L4.

Trước GRPO, chạy SFT checkpoint:

```text
4 rollouts / official training task
```

Tính empirical success.

Bucket:

```text
easy:
4/4 success

learnable:
1/4, 2/4, 3/4 success

hard:
0/4 success
```

Main RL sampling:

```text
~70% learnable
~15% easy
~15% hard
```

Lý do:

```text
0,0,0,0
→ reward_std = 0
→ GRPO gần như không có signal

1,1,1,1
→ reward_std = 0
→ cũng không có signal
```

Ta muốn:

```text
0,1,0,1
```

TRL thậm chí log sẵn `frac_reward_zero_std` để phát hiện chính vấn đề này. ([Hugging Face][11])

---

# 24. GRPO smoke config

Đừng chạy full experiment ngay.

```yaml
model:
  qwen3.5-2b-sft-merged

lora:
  r: 16
  alpha: 32
  dropout: 0.0

training:
  bf16: true

  per_device_train_batch_size: 1
  gradient_accumulation_steps: 2

  learning_rate: 1.0e-5

  num_generations: 2

  max_completion_length: 1024

  gradient_checkpointing: true

  use_vllm: true
  vllm_mode: colocate

  vllm_gpu_memory_utilization: 0.20

  vllm_enable_sleep_mode: true

  beta: 0.0

  max_steps: 20
```

TRL colocate chạy inference engine chung GPU với trainer; nếu memory căng, sleep mode có thể offload rollout resources trong optimization phase. ([Hugging Face][11])

`beta=0` còn giúp không cần load reference model, giảm memory. ([Hugging Face][11])

Smoke chỉ pass nếu:

```text
no OOM
no deadlock
tool execution works
user simulation works
reward works
backward works
checkpoint works
```

---

# 25. GRPO actual config

Nếu smoke pass:

```yaml
num_generations: 4

max_completion_length: 1536-2048

generation_batch_size: 4

learning_rate: 5e-6 to 1e-5

max_agent_turns: 8

gradient_checkpointing: true

use_vllm: true
vllm_mode: colocate

vllm_gpu_memory_utilization: 0.20-0.30

vllm_enable_sleep_mode: true

beta: 0.0
```

Không dùng:

```text
G=8
16k trajectory
10 RL epochs
```

trên L4.

---

# 26. GRPO loss formulation

TRL hiện mặc định `loss_type="dapo"` trong `GRPOTrainer`, vì vanilla sequence-normalized GRPO có response-length bias. ([Hugging Face][11])

Tôi khuyên project chính:

```yaml
loss_type: dapo
```

nhưng README ghi rõ:

> Online RL implemented with Hugging Face `GRPOTrainer`, using the DAPO token-normalized GRPO loss variant.

Nếu muốn đúng textbook GRPO tuyệt đối, thêm **small ablation** sau:

```text
loss_type=grpo
vs
loss_type=dapo
```

Không cần cho MVP.

---

# 27. Training sampling parameters

RL cần exploration:

```yaml
temperature: 0.8-1.0
top_p: 0.95
top_k: 20
```

Evaluation dùng lower stochasticity:

```text
temperature = 0.6
```

**Training decoding ≠ evaluation decoding là được**, miễn:

```text
Base eval
SFT eval
RL eval
```

dùng cùng evaluation config.

---

# 28. W&B — SFT phải log gì

Project:

```text
wandb project:
tau-agent-posttraining
```

Runs:

```text
qwen35-2b-base-eval
qwen35-2b-sft-retail
qwen35-2b-sft-eval
qwen35-2b-grpo-retail
qwen35-2b-grpo-eval
```

## SFT built-in

Log:

```text
train/loss
eval/loss

learning_rate
grad_norm

epoch
global_step

num_tokens
samples/sec
tokens/sec
```

TRL SFTTrainer exposes standard loss/token metrics. ([Hugging Face][5])

## Custom hardware

Every N steps:

```text
gpu/memory_allocated_gb
gpu/memory_reserved_gb
gpu/max_memory_allocated_gb

system/step_time
```

---

# 29. W&B — GRPO phải log gì

TRL already logs many useful GRPO metrics including:

```text
reward
reward_std
frac_reward_zero_std

entropy

completions/mean_length
completions/min_length
completions/max_length

step_time

clip_ratio/*
```

and when using vLLM:

```text
sampling/sampling_logp_difference/*
sampling/importance_sampling_ratio/*
```

([Hugging Face][11])

---

## Custom τ metrics

Log every rollout batch:

```text
tau/success_rate

tau/db_reward
tau/communicate_reward

tau/partial_action_reward

tau/invalid_tool_call_rate
tau/tool_argument_error_rate

tau/avg_tool_calls
tau/avg_agent_turns

tau/avg_reasoning_tokens
tau/p95_reasoning_tokens

tau/truncation_rate
tau/premature_termination_rate

tau/success_easy
tau/success_learnable
tau/success_hard
```

---

# 30. Log actual trajectories

Extremely important.

TRL can log completions to W&B via:

```text
log_completions=True
```

([Hugging Face][11])

Ngoài ra custom W&B Table:

| task_id | step | reward | reasoning | action | tool result | final | success |
| ------- | ---: | -----: | --------- | ------ | ----------- | ----- | ------- |

Sample:

```text
5–10 rollout groups mỗi 20–50 optimizer steps
```

Không upload toàn bộ thousands of trajectories nếu log quá lớn.

---

# 31. Evaluation trajectory table

Mỗi final eval tạo artifact:

```text
eval_results.jsonl
```

Mỗi row:

```json
{
  "task_id": "...",
  "checkpoint": "sft_rl",
  "trial": 0,
  "reward": 1,
  "db_reward": 1,
  "communicate_reward": 1,
  "partial_action_reward": 0.67,
  "num_tool_calls": 3,
  "num_agent_turns": 4,
  "reasoning_tokens": 621,
  "termination_reason": "agent_stop",
  "success": true
}
```

Không chỉ save aggregate metric.

---

# 32. Error taxonomy

Implement automatic error classification:

```text
A. invalid tool syntax
B. nonexistent tool
C. wrong tool
D. wrong argument
E. policy violation
F. missing required communication
G. incorrect DB mutation
H. unnecessary repeated read calls
I. premature final answer
J. thinking loop / truncation
K. user misunderstanding
```

README/report phải có:

```text
Base failure distribution
SFT failure distribution
RL failure distribution
```

Đây là phần giúp project trông như engineering/research thực sự.

---

# 33. Final evaluation protocol

Run exactly:

```text
Checkpoint A: Base
Checkpoint B: SFT
Checkpoint C: SFT + RL
```

on:

```text
τ³ v1.0.1
Retail
official test split
same user simulator
same decoding
same task IDs
4 trials/task
```

Result table:

| Model           | Pass¹ |   DB | Comm. | Invalid tools | Avg turns |
| --------------- | ----: | ---: | ----: | ------------: | --------: |
| Qwen3.5-2B Base |     X |    X |     X |             X |         X |
| + SFT           |     Y |    Y |     Y |             Y |         Y |
| + Agentic RL    |     Z |    Z |     Z |             Z |         Z |

Desired:

[
\boxed{X<Y<Z}
]

---

# 34. Statistical reporting

Đừng chỉ ghi:

```text
48 → 53 → 55
```

Compute:

```text
95% bootstrap CI
```

over task results.

Và paired delta:

```text
ΔSFT = SFT - Base
ΔRL  = RL - SFT
```

Ví dụ report:

```text
Base        47.2 ± 2.8
SFT         56.8 ± 2.7
SFT + RL    60.1 ± 2.6

Δ SFT       +9.6 pp
Δ RL        +3.3 pp
```

Các số trên **chỉ minh họa**.

---

# 35. Evaluation cadence

Không chạy full 4-trial benchmark liên tục.

### SFT

```text
Base full-ish dev eval

SFT:
epoch 0.5 → small dev
epoch 1   → full dev/test once
```

### RL

Every:

```text
20–50 optimizer steps
```

run:

```text
10–20 fixed dev tasks
1 trial
```

Final checkpoint mới:

```text
full held-out test × 4 trials
```

---

# 36. Hugging Face artifacts

Upload:

### Adapter 1

```text
<username>/qwen3.5-2b-tau-retail-sft
```

### Adapter 2

```text
<username>/qwen3.5-2b-tau-retail-grpo
```

Optional:

```text
<username>/qwen3.5-2b-tau-retail-sft-merged
```

Model card phải ghi:

```text
Base model
Data provenance
τ version + commit
SFT hyperparameters
RL hyperparameters
User simulator
Evaluation split
Known benchmark limitations
Results
```

**Không gọi model là official τ leaderboard model.**

Domain-specific fine-tuned models được τ leaderboard xếp vào **Custom submission**, không phải standard zero-shot submission. ([GitHub][7])

---

# 37. External baselines

Không tốn GPU chạy GPT/Claude.

README có section:

```text
External reference results
```

Dẫn:

* official τ leaderboard snapshot;
* Qwen official model card.

Ví dụ Qwen currently reports:

```text
Qwen3.5-2B TAU2 = 48.8
Qwen3.5-0.8B TAU2 = 11.6
```

nhưng phải annotate:

> Not directly comparable to our held-out τ³ Retail protocol because Qwen's published result uses their TAU2 evaluation setup and an airline modification.

([Hugging Face][1])

**Scientific comparison chính vẫn là Base/SFT/RL do chính bạn chạy trong cùng harness.**

---

# 38. Tests bắt buộc trước khi train

Engineer không được start full training nếu chưa pass:

### Data

```text
✓ no official test task in SFT data
✓ no official test task in RL train
✓ train/val split by task_id
```

### Mask

```text
✓ system labels = -100
✓ user labels = -100
✓ tool result labels = -100
✓ target assistant reasoning = train
✓ target tool call = train
```

### Qwen history

```text
✓ old thinking removed
✓ current thinking retained
```

### Environment

```text
✓ reset produces clean DB
✓ tool mutates isolated state
✓ reward matches official evaluator
```

### Rollout

```text
✓ plain assistant message accepted
✓ tool call accepted
✓ user simulator replies
✓ multiple turns work
✓ termination detected
```

### RL

```text
✓ G=2 produces two independent environments
✓ rewards attached to correct trajectories
✓ generated token masks only cover policy tokens
✓ one backward step succeeds
```

---

# 39. Implementation stages

## Stage A — infrastructure

Goal:

```text
Tau installs
Qwen loads
one Retail task works end-to-end
```

Acceptance:

```text
agent → user → tool → DB → reward
```

---

## Stage B — baseline

Evaluate:

```text
Qwen3.5-2B base
20 Retail test tasks
1 trial
```

Then small full-test baseline if stable.

---

## Stage C — SFT smoke

```text
100 training examples
20–50 steps
```

Verify:

```text
loss decreases
model generates valid tool calls
no mask bug
```

---

## Stage D — SFT full

```text
all eligible Retail successful synthetic trajectories
1 epoch
```

Evaluate.

Expected hypothesis:

[
SFT > Base
]

If SFT does **not** beat base:

> stop RL and debug SFT first.

---

## Stage E — RL rollout smoke

```text
10 official train tasks
G=2
20 steps
```

Check W&B.

---

## Stage F — difficulty profiling

Run:

```text
all official train tasks
4 rollouts/task
```

Create:

```text
easy
learnable
hard
```

---

## Stage G — RL actual

Start:

```text
G=4
learnable-heavy sampling
```

Train modest number of updates.

Do **not** precommit to 10 epochs.

Stop based on:

```text
training reward plateau
dev success plateau
entropy collapse
test-independent dev degradation
```

---

## Stage H — final evaluation

Run 3 checkpoints:

```text
Base
SFT
SFT+RL
```

4 trials/task.

Generate final report.

---

# 40. W&B dashboard cuối cùng nên có 6 panels

### Panel 1 — Main result

```text
Pass¹:
Base → SFT → RL
```

### Panel 2 — SFT

```text
train loss
val loss
grad norm
LR
```

### Panel 3 — RL learning

```text
reward
reward_std
frac_reward_zero_std
```

### Panel 4 — Policy health

```text
entropy
clip ratio
sampling logp difference
```

### Panel 5 — Agent behavior

```text
tool calls
turns
reasoning length
invalid call rate
```

### Panel 6 — Evaluation breakdown

```text
DB
COMMUNICATE
task category
failure types
```

---

# 41. Stop/rollback criteria

### SFT

Rollback if:

```text
val loss sharply increases
tool validity decreases
thinking collapses
held-out τ score decreases materially
```

### RL

Stop if:

```text
frac_reward_zero_std > ~80% persistently
```

→ change task sampling.

Stop/rollback if:

```text
entropy collapses
thinking length explodes
invalid tool calls increase strongly
dev task success decreases for multiple evals
```

Do not blindly train more.

---

# 42. What counts as project success?

Ideal:

```text
Base          45
SFT           55
SFT + RL      59
```

But even:

```text
Base          45
SFT           53
SFT + RL      55
```

is successful.

What matters:

[
\Delta_{\mathrm{SFT}}>0
]

and:

[
\Delta_{\mathrm{RL}}>0
]

with reproducible methodology.

Recent Tau-specific RL work provides precedent that RL can improve tool-calling agents; one 2026 experiment reports Qwen3.5-4B going from 63.8% to 66.7% on Tau-Bench Airline, while also showing that naïve reward shaping can hurt badly. ([arXiv][12])

---

# 43. CV story

Project title:

**Post-Training a 2B Reasoning LLM for Stateful Customer-Service Agents**

A strong final bullet would be structurally:

> Specialized Qwen3.5-2B into a stateful customer-service agent using reasoning SFT and online multi-turn RL with TRL, training against executable τ³-bench Retail environments and verifiable database-state rewards on a single NVIDIA L4; improved held-out task success from **X → Y → Z** across Base, SFT, and SFT+RL checkpoints.

Second bullet:

> Built a reproducible agentic post-training pipeline with LoRA, vLLM colocated rollouts, tool/API execution, user simulation, outcome verification, W&B trajectory analytics, reward-variance filtering, and paired held-out evaluation.

Đây **đủ mạnh cho JD yêu cầu “fine-tune LLM phục vụ assistant/agent”**. Không cần H100 để câu chuyện CV mạnh hơn; thứ quan trọng là bạn có clean experiment, reproducibility, online environment interaction, RL signal và error analysis.

[1]: https://huggingface.co/Qwen/Qwen3.5-2B "Qwen/Qwen3.5-2B · Hugging Face"
[2]: https://github.com/sierra-research/tau2-bench "GitHub - sierra-research/tau2-bench: τ-Bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains · GitHub"
[3]: https://github.com/sierra-research/tau2-bench/blob/main/RELEASE_NOTES.md "tau2-bench/RELEASE_NOTES.md at main · sierra-research/tau2-bench · GitHub"
[4]: https://huggingface.co/datasets/fuvty/tau-bench-synthetic "fuvty/tau-bench-synthetic · Datasets at Hugging Face"
[5]: https://huggingface.co/docs/trl/sft_trainer?utm_source=chatgpt.com "SFT Trainer"
[6]: https://github.com/sierra-research/tau2-bench/blob/main/src/tau2/gym/README.md "tau2-bench/src/tau2/gym/README.md at main · sierra-research/tau2-bench · GitHub"
[7]: https://github.com/sierra-research/tau2-bench/blob/main/docs/leaderboard-submission.md?utm_source=chatgpt.com "tau2-bench/docs/leaderboard-submission.md at main"
[8]: https://github.com/sierra-research/tau2-bench/blob/main/docs/evaluation.md?utm_source=chatgpt.com "tau2-bench/docs/evaluation.md at main"
[9]: https://github.com/sierra-research/tau2-bench/issues/384?utm_source=chatgpt.com "No-op and missing reward checks in airline/retail tasks allow ..."
[10]: https://huggingface.co/docs/trl/openenv "OpenEnv Integration for Training LLMs with Environments · Hugging Face"
[11]: https://huggingface.co/docs/trl/grpo_trainer "GRPO Trainer · Hugging Face"
[12]: https://arxiv.org/html/2604.02869v1?utm_source=chatgpt.com "Multi-Turn Reinforcement Learning for Tool-Calling Agents ..."
