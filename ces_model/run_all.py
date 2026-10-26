import subprocess, sys
for s in ["sim01_validation", "sim02_positions", "sim03_regimes", "sim04_orderings", "sim05_montecarlo", "sim06_power", "sim07_estimation", "sim08_rebate_match", "sim09_rebate_match_power"]:
    subprocess.run([sys.executable, s + ".py"], check=True)
