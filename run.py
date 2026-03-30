import subprocess
import sys

def run_command(cmd):
    print(f"\nRunning: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Command failed: {cmd}")
        sys.exit(result.returncode)

commands = [
    "python model_diffusion_latent_v2.py train_ae --tran-weight 10.0 --ae-epochs 100 --device cuda",
    "python model_diffusion_latent_v2.py train_diffusion --diff-epochs 100 --device cuda",
    "python model_diffusion_latent_v2.py test --device cuda",
    
    "python model_diffusion_latent.py train --ae-epochs 100 --diff-epochs 100 --device cuda",
    "python model_diffusion_latent.py mpjpe",
]

for cmd in commands:
    run_command(cmd)

print("\nAll commands completed successfully!")