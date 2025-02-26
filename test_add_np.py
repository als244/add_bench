import numpy as np
import time

num_els = 458000000
dtype_size = 1

a = np.random.randint(0, 256, size=num_els, dtype=np.uint8)
b = np.random.randint(0, 256, size=num_els, dtype=np.uint8)

start_time = time.time_ns()

c = a + b

end_time = time.time_ns()

elapsed_time_ns = end_time - start_time

elapsed_micros = elapsed_time_ns / 1e3

elapsed_time_sec = elapsed_time_ns / 1e9

total_bytes = num_els * 3 * dtype_size


mem_bw_util = (total_bytes / 1e9) / elapsed_time_sec

print(f"Elapsed Time (micros): {elapsed_micros:.3f}\nMem BW Estimate (GB / sec): {mem_bw_util:.3f}")

