import argparse
import os
import time
import boto3
import sagemaker
from sagemaker.huggingface import HuggingFace


def _resolve_role(role_arn: str | None, region: str) -> str:
    if role_arn:
        return role_arn
    try:
        from sagemaker import get_execution_role
        return get_execution_role()
    except Exception:
        iam = boto3.client("iam", region_name=region)
        role_name = os.environ.get("SAGEMAKER_EXEC_ROLE_NAME", "SageMakerExecutionRole")
        return iam.get_role(RoleName=role_name)["Role"]["Arn"]


def _bool(s: str) -> bool:
    return str(s).lower() in {"1", "true", "t", "yes", "y"}


def main():
    parser = argparse.ArgumentParser()
    # Infra
    parser.add_argument("--role-arn", type=str, default=None)
    parser.add_argument("--bucket", type=str, required=True, help="S3 bucket for outputs/checkpoints")
    parser.add_argument("--region", type=str, default=boto3.Session().region_name or "us-east-1")
    parser.add_argument("--instance-type", type=str, default="ml.g5.2xlarge")
    parser.add_argument("--instance-count", type=int, default=1)
    parser.add_argument("--job-name", type=str, default=f"s2t-intensity-train-{int(time.time())}")
    parser.add_argument("--use-spot", type=_bool, default=True)
    parser.add_argument("--max-run", type=int, default=24 * 3600)
    parser.add_argument("--max-wait", type=int, default=26 * 3600)
    parser.add_argument("--image-uri", type=str, default=None,
                        help="Optional: override HF DLC image URI for training")
    parser.add_argument("--transformers-version", type=str, default="4.49.0")
    parser.add_argument("--pytorch-version", type=str, default="2.5.1")
    parser.add_argument("--py-version", type=str, default="py311")

    parser.add_argument("--model-id", type=str, default="openai/whisper-small")
    parser.add_argument("--dataset", type=str, default="librispeech", choices=["librispeech", "common_voice"])
    parser.add_argument("--librispeech-config", type=str, default="clean")
    parser.add_argument("--train-split", type=str, default="train.100")
    parser.add_argument("--eval-split", type=str, default="validation")
    parser.add_argument("--test-split", type=str, default="test")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--intensity-method", type=str, default="rms", choices=["rms", "lufs"])
    parser.add_argument("--lambda-intensity", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--fp16", type=_bool, default=True)

    args = parser.parse_args()
    sess = sagemaker.Session(boto3.Session(region_name=args.region))
    role = _resolve_role(args.role_arn, args.region)

    hyperparameters = {
        "model_id": args.model_id,
        "dataset": args.dataset,
        "librispeech_config": args.librispeech_config,
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "test_split": args.test_split,
        "language": args.language,
        "intensity_method": args.intensity_method,
        "lambda_intensity": args.lambda_intensity,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "fp16": args.fp16,
        "output_dir": "/opt/ml/model",
    }

    # Where model.tar.gz (and optional checkpoints) go in S3
    output_path = f"s3://{args.bucket}/sm-training-output/"
    checkpoint_s3_uri = f"s3://{args.bucket}/sm-training-checkpoints/"

    estimator_kwargs = dict(
        entry_point="train_entry.py",
        source_dir=".",
        role=role,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        hyperparameters=hyperparameters,
        output_path=output_path,
        checkpoint_s3_uri=checkpoint_s3_uri,
        sagemaker_session=sess,
        dependencies=["sagemaker/train/requirements-train.txt"],
        use_spot_instances=args.use_spot,
        max_run=args.max_run,
        max_wait=args.max_wait if args.use_spot else None,
    )

    if args.image_uri:
        estimator = HuggingFace(image_uri=args.image_uri, **estimator_kwargs)
    else:
        estimator = HuggingFace(
            transformers_version=args.transformers_version,
            pytorch_version=args.pytorch_version,
            py_version=args.py_version,
            **estimator_kwargs
        )

    print(f"Starting SageMaker TrainingJob: {args.job_name}")
    estimator.fit(job_name=args.job_name, inputs=None)
    print("Done. Model artifact (model.tar.gz):", estimator.model_data)


if __name__ == "__main__":
    main()
