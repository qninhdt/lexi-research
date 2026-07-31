# %% [markdown]
# # Lexi Lab — Colab launcher
#
# This notebook is a launcher, not a program. Every cell is a shell command or
# three lines of environment plumbing; none defines a function or a class, and a
# test enforces that. An experiment that needs a code change gets a commit, not a
# cell — otherwise the run that produced a number cannot be reconstructed from
# the repository.
#
# Generated from `notebooks/lexi_colab.py`, which is the source of truth. Edit
# that file and run `make -f ops/Makefile notebook`.

# %% [markdown]
# ## 1. Clone and install
#
# The GPU stack comes from `requirements-colab.txt`; the package is installed
# editable so `lexi` lands on PATH and edits to the checkout take effect.

# %%
!git clone -q https://github.com/qninhdt/lexi-research.git /content/lexi-research
%cd /content/lexi-research
!pip install -q -r requirements-colab.txt
!pip install -q -e .

# %% [markdown]
# ## 2. Secrets
#
# From Colab's secret store into the environment. Nothing is printed, and
# nothing is written to a file the notebook could commit.

# %%
import os
from google.colab import userdata

for name in ("WANDB_API_KEY", "LEXI_TEACHER_BASE_URL", "LEXI_TEACHER_API_KEY", "LEXI_TEACHER_MODEL"):
    try:
        os.environ[name] = userdata.get(name)
    except Exception:
        print(f"{name} is not set — any stage needing it will fail loudly")

# %% [markdown]
# ## 3. Pull the dataset
#
# The parquet is Cambridge-derived and lives in remote storage, never in Git.

# %%
!dvc pull data/clean/train.parquet data/clean/val.parquet data/clean/band_config.json

# %% [markdown]
# ## 4. Prove the environment before spending on it
#
# One epoch over 50 synthetic rows against the real checkpoint. It prints how
# much of the model the adapter actually reached — the number that says whether a
# long run would have been training almost nothing.

# %%
!lexi smoke --gpu

# %% [markdown]
# ## 5. Train
#
# Arms change through `--override`, never through an edit. Add as many as the
# sweep needs; each one becomes part of the run config W&B records.

# %%
!lexi train sft \
    --train data/clean/train.parquet \
    --band-config data/clean/band_config.json \
    --output runs/sft/adapter \
    --override train.target_modules=all-linear

# %% [markdown]
# ## 6. Confirm the artifact
#
# The adapter and its band config went up as one artifact. A checkpoint without
# the config that derives its bands produces meaningless bands, so a run that
# shipped only half of the pair is a run to discard.

# %%
!ls -la runs/sft/adapter
print("check the W&B run page for the artifact version this run published")
