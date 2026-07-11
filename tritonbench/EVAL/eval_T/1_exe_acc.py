import os
import subprocess
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench_config import DATA_T_DIR, PY_INTERPRETER, parse_gpus  # noqa: E402

gold_folder = str(DATA_T_DIR) + "/"
py_interpreter = PY_INTERPRETER


def compare_python_files(file1, file2):
    result1 = subprocess.run([py_interpreter, file1], capture_output=True, text=True)
    output1 = result1.stdout
    result2 = subprocess.run([py_interpreter, file2], capture_output=True, text=True)
    output2 = result2.stdout
    return output1 == output2, file1.split("/")[-1]


def test_close_parallel(llm_folder, gold_folder_path, gpus):
    files = [f for f in os.listdir(llm_folder) if f.endswith(".py")]
    correct_count = 0
    total_count = len(files)

    with ProcessPoolExecutor(max_workers=len(gpus)) as executor:
        futures = []
        for idx, f in enumerate(files):
            file1 = os.path.join(llm_folder, f)
            file2 = os.path.join(gold_folder_path, f)
            gpu_id = gpus[idx % len(gpus)]
            futures.append(executor.submit(run_with_gpu, file1, file2, gpu_id))

        for future in futures:
            is_correct, file_name = future.result()
            if is_correct:
                correct_count += 1
            else:
                file_path = os.path.join(llm_folder, file_name)
                os.remove(file_path)
                print(f"Deleted {file_name}", flush=True)

    correct_rate = (correct_count / total_count) * 100 if total_count else 0
    assert total_count == len(files), "error in files"
    print(f"\nCorrect execution rate: {correct_rate:.2f}% = {correct_count} / {total_count}", flush=True)


def run_with_gpu(file1, file2, gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return compare_python_files(file1, file2)


def execute_4folders(root_folder, gpus):
    for folder in os.listdir(root_folder):
        llm_folder = os.path.join(root_folder, folder)
        if not os.path.isdir(llm_folder):
            continue
        test_close_parallel(llm_folder, gold_folder, gpus)
        print(f"above is the compare execution for {folder}", flush=True)
        print("========" * 30, flush=True)


def execute_4folder(folder, gpus):
    assert os.path.isdir(folder), folder
    test_close_parallel(folder, gold_folder, gpus)
    print(f"above is the compare execution for {folder}", flush=True)
    print("========" * 30, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Execution accuracy for Triton-T.")
    parser.add_argument("--folder", type=str, required=True, help="folder of generated .py files")
    parser.add_argument("--GPUs", type=str, required=True, help="GPU list, e.g. 0 or 0,1")
    args = parser.parse_args()
    gpus = parse_gpus(args.GPUs)
    assert os.path.isdir(args.folder), args.folder
    py_files = [f for f in os.listdir(args.folder) if f.endswith(".py")]
    if len(py_files) == 0:
        execute_4folders(args.folder, gpus)
    else:
        execute_4folder(args.folder, gpus)


if __name__ == "__main__":
    main()
