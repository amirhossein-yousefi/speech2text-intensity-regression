import os
import shlex
import subprocess
import sys


def to_cli_flag(k, v):
    # Turn (key, value) into CLI flag(s). Booleans become --flag true/false.
    if isinstance(v, bool):
        v = "true" if v else "false"
    return f"--{k.replace('_', '-')} {shlex.quote(str(v))}"


def main():
    incoming = sys.argv[1:]

    sm_model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

    if not any(arg.startswith("--output_dir") for arg in incoming):
        incoming += ["--output_dir", sm_model_dir]

    cmd = [sys.executable, "src/train_multitask_whisper.py"] + incoming
    print("Launching:", " ".join(shlex.quote(p) for p in cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
