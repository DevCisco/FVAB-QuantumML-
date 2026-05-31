import subprocess
import sys
import time

def run_all_experiments():
    regimes = [32, 16, 8, 4]
    backbones = ['resnet']

    print("=== AVVIO SUITE DI PRODUZIONE ===")
    total_start = time.time()

    for b in backbones:
        for d in regimes:
            print(f"\n[SUITE] Lancio esperimento: Backbone={b}, d={d}")
            
            cmd = [
                sys.executable,   # stesso interprete Python in uso (non hardcoded "python")
                "-c",
                (
                    f"from train_vqc_production import train_production; "
                    f"train_production(d={int(d)}, backbone='{b}', epochs=5)"
                )
            ]
            subprocess.run(cmd, shell=False, check=False)

    total_duration = (time.time() - total_start) / 60
    print(f"\n=== SUITE COMPLETATA IN {total_duration:.2f} MINUTI ===")

if __name__ == "__main__":
    run_all_experiments()