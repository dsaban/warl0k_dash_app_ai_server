import gc
import time
# import utime as time


try:
	from ulab import numpy as np
	from ulab import math as np_math
except ImportError:
	print("Using fallback numpy implementation")
	import array as np  # fallback, just to avoid crashes
	# import numpy as np_math  # fallback, just to avoid crashes
	import math as np_math  # fallback, just to avoid crashes


# import math

def print_memory(label):
	dir(gc)
	print(f"[{label}] Free RAM: {gc} bytes")

def tiny_model(seed, idx):
	""" Simple tiny 'model' simulating a transformation """
	factor = (seed * (idx + 1)) % 97
	noise = (seed % (idx + 3)) * 0.01
	result = np_math.sin(seed * (idx + 1) * 0.5) + np_math.cos(factor * 0.2) + noise
	return abs(result)

def generate_secrets(seed):
	secrets = []
	for i in range(3):
		t0 = time.time_ns()
		secret = tiny_model(seed, i)
		t1 = time.time_ns()
		print(f"Secret {i+1}: {secret:.5f} | Time: {t1 - t0} ns")
		secrets.append(secret)
	return secrets

def main():
	gc.collect()
	print_memory("START")

	seed = time.time_ns() % 1000000  # Simple session seed (millisecond timer)
	print(f"Session seed: {seed}")

	start_time = time.time_ns()
	secrets = generate_secrets(seed)
	concatenated_secret = ''.join('{:.5f}'.format(s) for s in secrets)
	end_time = time.time_ns()

	print(f"Concatenated secret: {concatenated_secret}")
	print(f"Total generation time: {(end_time - start_time)} ns")

	print_memory("END")

main()
