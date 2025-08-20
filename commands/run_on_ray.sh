RAY_ADDRESS=http://localhost:8268 ray job submit \
  --address="http://<head-node>:8265" \
  --runtime-env-json='{"working_dir": "./", "env_vars": {"TPU_MIN_LOG_LEVEL": "3", "XLA_FLAGS": "--xla_hlo_profile"}, "pip": ["einops", "pytest"]}' \
  -- pytest tests/partitioning/flax/test_attention.py