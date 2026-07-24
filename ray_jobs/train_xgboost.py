"""Ray Train — distributed XGBoost on the Playground RayCluster.

Modern Ray 2.52 API (train_loop_per_worker style). Trains a binary classifier on
the sklearn breast-cancer dataset, data-parallel across `num_workers` Ray workers.
Ray sets up XGBoost's distributed collective; each worker trains on its data shard.

Run on the cluster:
    ray job submit --working-dir . -- python ray_jobs/train_xgboost.py
or directly on the head pod:
    python train_xgboost.py
"""

import ray
import ray.train
from ray.train import ScalingConfig, RunConfig
from ray.train.xgboost import XGBoostTrainer
import xgboost


def train_fn_per_worker(config: dict):
    label = config["label_column"]

    # Each worker gets its shard of the dataset.
    train_df = ray.train.get_dataset_shard("train").materialize().to_pandas()
    eval_df = ray.train.get_dataset_shard("validation").materialize().to_pandas()

    dtrain = xgboost.DMatrix(train_df.drop(columns=[label]), label=train_df[label])
    deval = xgboost.DMatrix(eval_df.drop(columns=[label]), label=eval_df[label])

    evals_result: dict = {}
    xgboost.train(
        params={
            "objective": "binary:logistic",
            "eval_metric": ["logloss", "error"],
            "tree_method": "hist",
        },
        dtrain=dtrain,
        num_boost_round=config.get("num_boost_round", 20),
        evals=[(dtrain, "train"), (deval, "validation")],
        evals_result=evals_result,
        verbose_eval=False,
    )

    # Report final metrics (no checkpoint -> no shared storage needed for the demo).
    val_err = evals_result["validation"]["error"][-1]
    metrics = {
        "train_logloss": evals_result["train"]["logloss"][-1],
        "validation_logloss": evals_result["validation"]["logloss"][-1],
        "validation_error": val_err,
        "validation_accuracy": 1.0 - val_err,
    }
    ray.train.report(metrics)

    # Print from rank 0 so the final numbers land in the job output.
    # (Ray Train v2 only propagates metrics to the driver Result via a persisted
    # checkpoint, which needs shared storage — see playbook "Next steps".)
    if ray.train.get_context().get_world_rank() == 0:
        print("\n===TRAIN_METRICS===")
        for k, v in metrics.items():
            print(f"  {k:22s} {v:.4f}")


def main():
    ray.init(address="auto")

    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    frame = load_breast_cancer(as_frame=True).frame  # features + "target"
    train_df, eval_df = train_test_split(frame, test_size=0.2, random_state=42)

    trainer = XGBoostTrainer(
        train_fn_per_worker,
        train_loop_config={"label_column": "target", "num_boost_round": 20},
        datasets={
            "train": ray.data.from_pandas(train_df),
            "validation": ray.data.from_pandas(eval_df),
        },
        scaling_config=ScalingConfig(num_workers=2, resources_per_worker={"CPU": 1}),
        run_config=RunConfig(name="xgb-breast-cancer", storage_path="/tmp/ray_results"),
    )

    result = trainer.fit()

    # Ray Train v2: `result.metrics` is None when reporting without a checkpoint,
    # so fall back to the reported-metrics history.
    metrics = result.metrics or {}
    if not metrics and result.metrics_dataframe is not None and len(result.metrics_dataframe):
        metrics = result.metrics_dataframe.iloc[-1].to_dict()

    print("\n===TRAIN_DONE===")
    print(f"  workers          : 2 (data-parallel)")
    print(f"  result status    : {'error' if result.error else 'success'}")
    if metrics:
        for k, v in metrics.items():
            print(f"  {k:22s} {v}")
    else:
        print("  metrics          : reported per-worker (see ===TRAIN_METRICS=== above)")


if __name__ == "__main__":
    main()
