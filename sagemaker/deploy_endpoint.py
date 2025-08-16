# sagemaker/deploy_endpoint.py
import argparse, os, tarfile, tempfile, boto3
from sagemaker import Session
from sagemaker.pytorch import PyTorchModel
from sagemaker.serverless import ServerlessInferenceConfig

def _tar_dir(src_dir: str, out_tar: str):
    with tarfile.open(out_tar, "w:gz") as tar:
        for root, _, files in os.walk(src_dir):
            for f in files:
                full = os.path.join(root, f)
                rel  = os.path.relpath(full, src_dir)
                tar.add(full, arcname=rel)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True, help="S3 bucket for model artifacts (must exist)")
    ap.add_argument("--ckpt_dir", required=True, help="Local directory saved by training (e.g., ./checkpoints/mtl_whisper_small)")
    ap.add_argument("--region", default=os.getenv("AWS_REGION", "us-east-1"))
    ap.add_argument("--role_arn", required=True, help="IAM Role ARN for SageMaker")
    ap.add_argument("--endpoint_name", default="s2t-intensity-whisper")
    ap.add_argument("--instance_type", default="ml.g5.xlarge")
    ap.add_argument("--framework_version", default="2.4")  # choose one your account supports
    ap.add_argument("--py_version", default="py311")
    ap.add_argument("--serverless", action="store_true", help="Deploy as Serverless Inference")
    args = ap.parse_args()

    boto_sess = boto3.Session(region_name=args.region)
    sm_sess = Session(boto_session=boto_sess)

    # 1) Package model artifacts
    with tempfile.TemporaryDirectory() as tmp:
        tar_path = os.path.join(tmp, "model.tar.gz")
        _tar_dir(args.ckpt_dir, tar_path)
        s3_uri = sm_sess.upload_data(path=tar_path, bucket=args.bucket, key_prefix="s2t-intensity/model")
        print(f"Uploaded model to: {s3_uri}")

    # 2) Define PyTorchModel with our serving code in sagemaker/ (source_dir)
    #    Also ship your 'src/' so the custom head can be imported at runtime.
    pytorch_model = PyTorchModel(
        model_data=s3_uri,
        role=args.role_arn,
        entry_point="inference.py",
        source_dir="sagemaker",            # <-- contains inference.py + requirements.txt
        dependencies=["src"],              # <-- make src/ available on PYTHONPATH
        framework_version=args.framework_version,
        py_version=args.py_version,
        env={
            "LANGUAGE": "en",
            "TASK": "transcribe",
            "BASE_MODEL_ID": "openai/whisper-small"
        }
    )

    # 3) Deploy
    if args.serverless:
        serverless_cfg = ServerlessInferenceConfig(memory_size_in_mb=4096, max_concurrency=2)
        predictor = pytorch_model.deploy(
            endpoint_name=args.endpoint_name,
            serverless_inference_config=serverless_cfg
        )
    else:
        predictor = pytorch_model.deploy(
            endpoint_name=args.endpoint_name,
            initial_instance_count=1,
            instance_type=args.instance_type,
            model_server_workers=1
        )

    print(f"Endpoint is up: {predictor.endpoint_name}")

if __name__ == "__main__":
    main()
