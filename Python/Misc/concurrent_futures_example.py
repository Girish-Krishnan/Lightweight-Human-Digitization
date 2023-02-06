import concurrent.futures
import time

def worker(num):
    print(f'Worker {num} started')
    time.sleep(2) # simulating some work
    return f'Worker {num} finished'

if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        results = [executor.submit(worker, i) for i in range(5)]

        for f in concurrent.futures.as_completed(results):
            print(f.result())
