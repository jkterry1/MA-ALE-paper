import os
for name in os.listdir("/home/ben/job_results/"):
    if "." not in name:
        print(f"(cd ~/job_results/{name} && mkdir checkpoint && mv  runs/*/checkpoint*.pth checkpoint/)")
