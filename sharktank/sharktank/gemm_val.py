import numpy as np
import torch
import subprocess
import os

# Set seed for reproducibility
np.random.seed(42)

def generate_gemm_mlir(mlir_path, M =1024, N=1024, K=1024, device_count=2, ):
    # check if our parent directory is shark-ai/sharktank/sharktank
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if not current_dir.endswith("sharktank/sharktank"):
        print("current_dir:", current_dir)
        raise ValueError("This script must be run from the shark-ai/sharktank/sharktank directory.")

    # if the directory does not exist for mlir_path, create it
    mlir_dir = os.path.dirname(mlir_path)
    if not os.path.exists(mlir_dir):
        os.makedirs(mlir_dir)
        print(f"Created directory for MLIR file: {mlir_dir}")
    
    # Generate the MLIR file for GEMM
    mlir_cmd = [
        "python",
        "-m sharktank.examples.sharding.export_gemm",
        "--mlir={mlir_path}",
        "--device_count={device_count}",
        "--m={M}",
        "--n={N}",
        "--k={K}",
    ]

    # Run the command to generate the MLIR file
    mlir_cmd_str = " ".join(mlir_cmd).format(
        mlir_path=mlir_path,
        device_count=device_count,
        M=M,
        N=N,
        K=K
    )

    print("Generating MLIR file with command: ", mlir_cmd_str) 
    subprocess.run(mlir_cmd_str, check=True, shell=True)

    # Check if the MLIR file was created
    if not os.path.exists(mlir_path):
        raise FileNotFoundError(f"MLIR file {mlir_path} was not created successfully.")
    print(f"MLIR file generated at {mlir_path}")

    # Update the mlir_path with a hotfix
    # iree-opt --torch-recompose-complex-ops gemm.mlir -o=updated_gemm.mlir
    updated_mlir_path = mlir_path.replace(".mlir", "_updated.mlir")
    hotfix_cmd = [
        "iree-opt",
        "--torch-recompose-complex-ops",
        mlir_path,
        "-o",
        updated_mlir_path
    ]
    print("Applying torch-recompose-complex-ops with command:", " ".join(hotfix_cmd))
    subprocess.run(hotfix_cmd, check=True)

    if not os.path.exists(updated_mlir_path):
        raise FileNotFoundError(f"Updated MLIR file {updated_mlir_path} was not created successfully.")
    print(f"Updated MLIR file generated at {updated_mlir_path}")

    return updated_mlir_path

def generate_gemm_vmfb(mlir_path, vmfb_path, device_count=2):
    # Compile the MLIR file to VMFB
    # cmd format: iree-compile updated_gemm.mlir --iree-hip-target=gfx942 -o=updated_gemm.vmfb
    #               --iree-hal-target-device=hip[0] --iree-hal-target-device=hip[1] --iree-opt-level=O3

    # if the directory does not exist for vmfb_path, create it
    vmfb_dir = os.path.dirname(vmfb_path)
    if not os.path.exists(vmfb_dir):
        os.makedirs(vmfb_dir)
        print(f"Created directory for VMFB file: {vmfb_dir}")

    # based on the device_count, we will generate the target devices
    target_devices = [f"--iree-hal-target-device=hip[{i}]" for i in range(device_count)]
    target_devices_str = " ".join(target_devices)

    compile_cmd = [
        "iree-compile",
        mlir_path,
        "--iree-hip-target=gfx942",
        target_devices_str,
        "--iree-opt-level=O3",
        "-o={vmfb_path}",
    ]

    compile_cmd_str = " ".join(compile_cmd).format(vmfb_path=vmfb_path)
    print("Compiling MLIR to VMFB with command:", compile_cmd_str)
    subprocess.run(compile_cmd_str, check=True, shell=True)

    if not os.path.exists(vmfb_path):
        raise FileNotFoundError(f"VMFB file {vmfb_path} was not created successfully.")
    print(f"VMFB file generated at {vmfb_path}")

    return vmfb_path

def execute_gemm_kernel(vmfb_path, A, B, Bias, build_dir, device_count=2):

    # Check if the build directory exists, if not create it
    if not os.path.exists(build_dir):
        os.makedirs(build_dir)
        print(f"Created build directory: {build_dir}")

    np.save(f"{build_dir}/a.npy", A)
    np.save(f"{build_dir}/b.npy", B)
    np.save(f"{build_dir}/bias.npy", Bias)
    a_path = os.path.join(build_dir, "a.npy")
    b_path = os.path.join(build_dir, "b.npy")
    bias_path = os.path.join(build_dir, "bias.npy")
    print(f"Input matrices saved to {a_path}, {b_path}, {bias_path}")

    # number of outputs depends on device_count
    output_files = ["--output=@{}/c{}.npy".format(build_dir, i) for i in range(device_count)]
    output_files_str = " ".join(output_files)

    # hip devices
    hip_devices = ["--device=hip://{}".format(i) for i in range(device_count)]
    hip_devices_str = " ".join(hip_devices) 
    # Run the kernel using iree-run-module
    kernel_cmd = [
        "iree-run-module",
        "--hip_use_streams=true",
        "--module={vmfb_path}",
        hip_devices_str,
        "--input=@{a_path}",
        "--input=@{b_path}",
        "--input=@{bias_path}",
        output_files_str
    ]

    kernel_cmd_str = " ".join(kernel_cmd).format(
        vmfb_path=vmfb_path,
        a_path=a_path,
        b_path=b_path,
        bias_path=bias_path,
    )
    print("Running kernel command:", kernel_cmd_str)
    result = subprocess.run(kernel_cmd_str, check=True, env=os.environ, shell=True)
    
    if result.returncode != 0:
        raise RuntimeError("Kernel execution failed with return code: {}".format(result.returncode))
    
    # check if output files were created
    output_files_created = all(os.path.exists(f"{build_dir}/c{i}.npy") for i in range(device_count))
    if not output_files_created:
        raise FileNotFoundError("Output files were not created successfully.")
    print("Output files created successfully.")

    # concatenate the output files to numpy array
    C = np.concatenate([
        np.load(f"{build_dir}/c{i}.npy") for i in range(device_count)
    ], axis=0)

    print("C.shape after kernel:", C.shape)
    return C

def validate_gemm_result(C, golden_C):
    # Validate the result against the golden matrix
    if np.allclose(C, golden_C):
        print("Validation successful: C matches golden_C")
    else:
        print("Validation failed: C does not match golden_C")
        print("Difference:", np.abs(C - golden_C).max())
        raise ValueError("Matrix multiplication result does not match the expected output.")

    # Print random 10 indices and their values from C
    random_indices = np.random.choice(C.size, size=10, replace=False)
    print("Random indices and their values from C:")
    for idx in random_indices:
        row, col = divmod(idx, C.shape[1])
        print(f"C[{row}, {col}] = {C[row, col]} == golden_C[{row}, {col}] = {golden_C[row, col]}")

def main():
    device_count = 4  # Number of devices to use
    M, N, K = 2048, 2048, 2048 
    # Generate GEMM MLIR
    mlir_path = generate_gemm_mlir(device_count=device_count, M=M, N=N, K=K, mlir_path="gemm_example/export_gemm.mlir")

    # Generate VMFB from MLIR
    vmfb_path = generate_gemm_vmfb(mlir_path=mlir_path, device_count=device_count, vmfb_path="gemm_example/gemm.vmfb")

    # Generate input matrices
    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    Bias = np.zeros((M, N), dtype=np.float32)  # Bias is not used in this example

    # Execute GEMM kernel
    C = execute_gemm_kernel(vmfb_path, A, B, Bias, device_count=device_count, build_dir="./gemm_example")

    # Validate the result
    golden_C = np.matmul(A, B)
    validate_gemm_result(C, golden_C)

if __name__ == "__main__":
    main()

""" 
# Generate input matrices
A = np.random.rand(1024, 1024).astype(np.float32)
B = np.random.rand(1024, 1024).astype(np.float32)
Bias = np.zeros((1024, 1024), dtype=np.float32)  # Bias is not used in this example

# Split A along dim=0 into two halves and save as a0.npy and a1.npy
np.save("a.npy", A)
np.save("b.npy", B)
np.save("bias.npy", Bias)

# Validate matrix multiplication
golden_C = np.matmul(A, B)

# Optional: validate dimensions and sanity check
print("A.shape:", A.shape)
print("B.shape:", B.shape)
print("golden_C.shape:", golden_C.shape)

# Run the kernel using iree-run-module
kernel_cmd = [
    "iree-run-module",
    "--hip_use_streams=true",
    "--module=gemm.vmfb",
    "--device=hip://0",
    "--device=hip://1",
    "--input=@a.npy",
    "--input=@b.npy",
    "--input=@bias.npy",
    "--output=@c0.npy",
    "--output=@c1.npy"
]
print("Running kernel command:", " ".join(kernel_cmd))
subprocess.run(kernel_cmd, check=True)

# Read c0.npy and c1.npy and concatenate to form C
if os.path.exists("c0.npy") and os.path.exists("c1.npy"):
    c0 = np.load("c0.npy")
    c1 = np.load("c1.npy")
    C = np.concatenate([c0, c1], axis=0)
    print("C.shape after kernel:", C.shape)

# Validate the result against the golden matrix
if np.allclose(C, golden_C):
    print("Validation successful: C matches golden_C")
else:
    print("Validation failed: C does not match golden_C")
    print("Difference:", np.abs(C - golden_C).max())
    raise ValueError("Matrix multiplication result does not match the expected output.")


# Print random 10 indices and their values from C
random_indices = np.random.choice(C.size, size=10, replace=False)
print("Random indices and their values from C:")
for idx in random_indices:
    row, col = divmod(idx, C.shape[1])
    print(f"C[{row}, {col}] = {C[row, col]} == golden_C[{row}, {col}] = {golden_C[row, col]}")
"""