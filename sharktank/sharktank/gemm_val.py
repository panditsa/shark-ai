import numpy as np
import torch
import subprocess
import os
import time
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

def generate_gemm_vmfb(build_dir, mlir_path, vmfb_path, device_count=2):
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
        "--mlir-print-ir-after-all"
    ]
    # "--mlir-print-ir-after-all"

    compile_cmd_str = " ".join(compile_cmd).format(vmfb_path=vmfb_path)
    print("Compiling MLIR to VMFB with command:", compile_cmd_str)
    # store the stdout and stderr in a log file
    with open(build_dir + "/compile_log.txt", "w") as log_file:
        result = subprocess.run(compile_cmd_str, check=True, shell=True, stdout=log_file, stderr=subprocess.STDOUT)
    #subprocess.run(compile_cmd_str, check=True, shell=True)

    if not os.path.exists(vmfb_path):
        raise FileNotFoundError(f"VMFB file {vmfb_path} was not created successfully.")
    print(f"VMFB file generated at {vmfb_path}")

    return vmfb_path

def execute_gemm_kernel(vmfb_path, A, B, Bias, build_dir, device_count=2, unified_output=False, function_name="isolated_benchmark"):

    # Check if the build directory exists, if not create it
    if not os.path.exists(build_dir):
        os.makedirs(build_dir)
        print(f"Created build directory: {build_dir}")

    # Save the tensor as npy files
    print("Saving input matrices to numpy files...")
    if not isinstance(A, torch.Tensor):
        raise TypeError("Input A must be a torch.Tensor")
    if not isinstance(B, torch.Tensor):
        raise TypeError("Input B must be a torch.Tensor")
    if not isinstance(Bias, torch.Tensor):
        raise TypeError("Input Bias must be a torch.Tensor")
    
    A = A.cpu().numpy()  # Convert to numpy array
    B = B.cpu().numpy()  # Convert to numpy array
    Bias = Bias.cpu().numpy()  # Convert to numpy array
    if A.ndim != 2 or B.ndim != 2 or Bias.ndim != 2:
        raise ValueError("Input matrices A, B, and Bias must be 2D tensors.")
    
    np.save(f"{build_dir}/a.npy", A)
    np.save(f"{build_dir}/b.npy", B)
    np.save(f"{build_dir}/bias.npy", Bias)
    a_path = os.path.join(build_dir, "a.npy")
    b_path = os.path.join(build_dir, "b.npy")
    bias_path = os.path.join(build_dir, "bias.npy")
    print(f"Input matrices saved to {a_path}, {b_path}, {bias_path}")

    # number of outputs depends on device_count
    if unified_output:
        output_files = ["--output=@{}/c.npy".format(build_dir, i) for i in range(0, 1)]
        output_files_str = " ".join(output_files)
    else:
        output_files = ["--output=@{}/c{}.npy".format(build_dir, i) for i in range(device_count)]
        output_files_str = " ".join(output_files)
    # hip devices
    hip_devices = ["--device=hip://{}".format(i) for i in range(device_count)]
    hip_devices_str = " ".join(hip_devices) 
    # Run the kernel using iree-run-module
    kernel_cmd = [
        "iree-run-module",
        #"--hip_use_streams=true",
        "--module={vmfb_path}",
        "--function={function_name}",
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
        function_name=function_name
    )
    print("Running kernel command:", kernel_cmd_str)

    start_time = time.perf_counter()
    result = subprocess.run(kernel_cmd_str, check=True, env=os.environ, shell=True)
    end_time = time.perf_counter()

    if result.returncode != 0:
        raise RuntimeError("Kernel execution failed with return code: {}".format(result.returncode))
    
    execution_time = end_time - start_time
    print(f"Experiment ran for {execution_time:.4f} seconds.")
    
    # check if output files were created

    if unified_output:
        output_files_created = os.path.exists(f"{build_dir}/c.npy")
        if not output_files_created:
            raise FileNotFoundError("Output file c.npy was not created successfully.")
        C = torch.from_numpy(np.load(f"{build_dir}/c.npy")).to(dtype=torch.float32)
    else:
        # Check if all output files were created
        output_files_created = all(os.path.exists(f"{build_dir}/c{i}.npy") for i in range(device_count))
        if not output_files_created:
            raise FileNotFoundError("Output files were not created successfully.")
        print("Output files created successfully.")

        # Directly read the output npy files into torch tensors and concatenate
        C_parts = [
            torch.from_numpy(np.load(f"{build_dir}/c{i}.npy")).to(dtype=torch.float32)
            for i in range(device_count)
        ]

        # C_m = [torch.cat([C_parts[0], C_parts[1]], dim=0)]
        # C_m.append(torch.cat([C_parts[2], C_parts[3]], dim=0))

        C = torch.cat(C_parts, dim=1)
        #C = torch.cat(C_parts, dim=1)

    nBias = torch.from_numpy(np.load(f"{build_dir}/bias.npy")).to(dtype=torch.float32)
    # print first 10 elements of nBias
    print("Bias first 10 elements:", nBias.flatten()[:10])

    print("C.shape after kernel:", C.shape)
    return C

def validate_gemm_result(A, B, C, atol=1, transpose_B=False):
    if transpose_B:
        expected = (A.float() @ B.t().float()).to(dtype=torch.float32)
    else:
        expected = (A.float() @ B.float()).to(dtype=torch.float32)
    diff_mask = ~torch.isclose(C, expected, atol=atol)
    breaking_indices = torch.nonzero(diff_mask, as_tuple=False)

    # print all of B and C
    print(A)

    # store B in a readable format in a file
    with open("B.txt", "w") as f:
        for row in B:
            f.write(" ".join(f"{val:.4f}" for val in row) + "\n")
    print("B matrix saved to B.txt")

    # store C in a readable format in a file
    with open("C.txt", "w") as f:
        for row in C:
            f.write(" ".join(f"{val:.4f}" for val in row) + "\n")
    print("C matrix saved to C.txt")

    with open("expected.txt", "w") as f:
        for row in expected:
            f.write(" ".join(f"{val:.4f}" for val in row) + "\n")
    print("Expected matrix saved to expected.txt")

    if diff_mask.any():
        max_diff = (C - expected).abs().max().item()
        print(f"Max absolute difference: {max_diff}")
        max_print = 10
        for idx in breaking_indices:
            idx = tuple(idx.tolist())
            computed_val = C[idx]
            expected_val = expected[idx]
            print(f"Mismatch at index {idx}: C={computed_val}, expected={expected_val}")
            max_print -= 1
            if max_print <= 0:
                break
        return False
    
    # print 10 random indices and their values from C and expected
    random_indices = torch.randint(0, C.numel(), (10,))
    for idx in random_indices:
        row, col = divmod(idx.item(), C.shape[1])
        print(f"Random index C[{row}, {col}] = {C[row, col]} == expected[{row}, {col}] = {expected[row, col]}")
    print("All values match within the specified tolerance.")
    return True

def main():

    device_count = 2  # Number of devices to use
    M, N, K = 64, 128, 64
    unified_output = True 
    transpose_B = True
    function_name = "isolated_benchmark"
    #function_name = "\'isolated_benchmark\'" 
    # Generate GEMM MLIR
    # mlir_path = generate_gemm_mlir(device_count=device_count, M=M, N=N, K=K, mlir_path="gemm_example/export_gemm.mlir")

    # mlir_path = 'gemm_2d_n.mlir' 
    mlir_path = 'wave.mlir'
    #mlir_path = "/home/sanketp/work/shark-ai/sharktank/sharktank/gemm_example/2d_old_reg.mlir"
    # Generate VMFB from MLIR
    #vmfb_path = generate_gemm_vmfb(device_count=device_count, build_dir="./gemm_example", mlir_path=mlir_path, vmfb_path="gemm_example/gemm.vmfb")
    #vmfb_path = '/home/sanketp/work/shark-ai/sharktank/sharktank/gemm_example/gemm.vmfb'
    vmfb_path = '/home/sanketp/work/wave/dist_gemm.vmfb'
    # Generate input matrices
    A = torch.randn((M, K), dtype=torch.float16)

    b_shape = (N, K) if transpose_B else (K, N)
    B = torch.randn(b_shape, dtype=torch.float16)

    # if transpose_B:
    #     B = B.t()

    # Bias is not used in this example, but we need to provide it 
    Bias = torch.zeros((M, N), dtype=torch.float32)

    # Execute GEMM kernel
    C = execute_gemm_kernel(vmfb_path, A, B, Bias, device_count=device_count, build_dir="./gemm_example", unified_output=unified_output, function_name=function_name)

    # Validate the result
    if not validate_gemm_result(A, B, C, transpose_B=transpose_B):
       raise ValueError("GEMM validation failed: Computed result does not match expected result.")
    
    print("GEMM validation successful: Computed result matches expected result.")

if __name__ == "__main__":
    main()
