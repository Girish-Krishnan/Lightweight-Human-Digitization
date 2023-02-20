import multiprocessing

# Define the function to be executed in parallel
def task(num):
    result = num * num
    print(f"The square of {num} is {result}")

if __name__ == "__main__":
    # Create a pool of processes with 4 workers
    with multiprocessing.Pool(processes=4) as pool:
        # Submit the task to the pool for parallel processing
        results = [pool.apply_async(task, args=(i,)) for i in range(10)]
        # Wait for all processes to complete and print their results
        for result in results:
            result.get()
