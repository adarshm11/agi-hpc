"""Patch setup.py files to add sm_60 (P100) CUDA architecture."""

def patch_file(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        new_lines.append(line)
        if 'arch=compute_53,code=sm_53' in line:
            indent = line[:len(line) - len(line.lstrip())]
            new_lines.append(indent + 'cc_flag.append("-gencode")\n')
            new_lines.append(indent + 'cc_flag.append("arch=compute_60,code=sm_60")\n')

    with open(filepath, "w") as f:
        f.writelines(new_lines)
    print(f"Patched {filepath}")

patch_file("/home/006992466/nemotron/src_packages/causal-conv1d/setup.py")
patch_file("/home/006992466/nemotron/src_packages/mamba/setup.py")
