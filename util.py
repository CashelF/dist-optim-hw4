import torch
import torch.distributed as dist
import os
import torch.multiprocessing as mp

def generate_random_tensor(rank, message_size):
    
    # Set the random seed to the worker's rank. This makes the
    # generated tensor reproducible for a given worker.
    torch.manual_seed(rank)

    # We create a 3x3 tensor of random floating-point numbers.
    # This tensor represents the data `x_n` for worker `n`.
    return torch.rand(message_size)

def test_func_H3T1S2(rank, world_size, message_size, received_tensor):

    # =========================================================
    #   Step 1: Generate a random tensor for this worker.
    # =========================================================
    # Set the random seed to the worker's rank. This makes the
    # generated tensor reproducible for a given worker.
    torch.manual_seed(rank)

    # We create a tensor of random floating-point numbers.
    # This tensor represents the data `x_n` for worker `n`.
    expect_tensor = generate_random_tensor((rank - 1) % world_size, message_size)
    
    print(f"""Worker {rank}: 
        Expected tensor: 
        {expect_tensor}
        Received tensor:
        {received_tensor} \n
        """)
    
    assert torch.equal(expect_tensor, received_tensor), f"Test failed in worker {rank}: Received tensor does not match expected tensor."
    
import torch
import torch.multiprocessing as mp
def test_func_H4T2(reduce_func, world_size, message_size, backend):
    mp.spawn(run_all_reduce, args=(world_size, reduce_func, message_size, backend), nprocs=world_size, join=True)
    print("Test passed!")
    

def run_all_reduce(rank, reduce_func, world_size, message_size, backend):
    
    """A helper function to run the example."""
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    
    # Create a local tensor with unique values for each process
    
    local_data = generate_random_tensor(rank, message_size)
    original_copy = local_data.clone()
    print(f"Process {rank} initial: {local_data}")

    result = reduce_func(local_data)

    expected_result = original_copy.clone()
    dist.all_reduce(expected_result)
    
    print(f"Process {rank} expected: {expected_result}; output: {result}")
    
    torch.testing.assert_close(result, expected_result), f"Process {rank} verification failed! Expected: {expected_result}; output: {result}"
    print(f"Process {rank} verification successful!")
    
    
    
import ast
import os
import textwrap

def combine_cache_functions(cache_dir="cache", out_file="hw4_functions.py"):
    """
    Combine all function, async function, and class definitions from .py files
    in `cache_dir` into a single Python file `out_file`. Also collects unique
    import statements (import / from ... import ...) found at top-level.
    Lines starting with IPython magics (e.g., '%%writefile') are ignored.
    """
    py_files = sorted(
        [p for p in os.listdir(cache_dir) if p.endswith(".py")]
    )

    imports = []
    import_set = set()
    defs = []

    for fname in py_files:
        path = os.path.join(cache_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            raw_src = f.read()

        # Remove IPython magic lines that may remain
        src_lines = [ln for ln in raw_src.splitlines() if not ln.strip().startswith("%%")]
        src = "\n".join(src_lines)

        try:
            tree = ast.parse(src)
        except Exception:
            # If parsing fails, include whole file as a fallback (without magics)
            cleaned = "\n".join(src_lines).strip()
            if cleaned:
                defs.append(cleaned)
            continue

        for node in tree.body:
            # Collect import statements
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                try:
                    code = ast.get_source_segment(src, node)
                except Exception:
                    code = None
                if not code:
                    # Try to unparse as fallback (Py3.9+)
                    try:
                        code = ast.unparse(node)
                    except Exception:
                        code = None
                if code:
                    code_stripped = code.strip()
                    if code_stripped not in import_set:
                        import_set.add(code_stripped)
                        imports.append(code_stripped)
            # Collect functions, async functions, and classes
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                try:
                    seg = ast.get_source_segment(src, node)
                except Exception:
                    seg = None
                if not seg:
                    # Fallback using lineno/end_lineno
                    if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
                        seg = "\n".join(src.splitlines()[node.lineno - 1 : node.end_lineno])
                    else:
                        seg = ""
                if seg:
                    # Ensure consistent indentation
                    defs.append(textwrap.dedent(seg).rstrip())

        # Optionally, include other top-level needed constructs (skipped by design)

    # Write combined file
    header = (
        "# Auto-generated file combining functions/classes from files in 'cache'\n"
        "# Generated by combine_cache_functions\n\n"
    )
    with open(out_file, "w", encoding="utf-8") as outf:
        outf.write(header)
        if imports:
            outf.write("\n".join(imports))
            outf.write("\n\n")
        for d in defs:
            outf.write(d)
            outf.write("\n\n")

    print(f"Created '{out_file}' with {len(imports)} unique imports and {len(defs)} definitions.")
