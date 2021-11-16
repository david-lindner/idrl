black --check --verbose active_reward_learning
isort --check-only active_reward_learning
jsonlint-php experiment_configs/*/*.json
mypy --ignore-missing-imports active_reward_learning
