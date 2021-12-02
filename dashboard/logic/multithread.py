from concurrent.futures import ThreadPoolExecutor, as_completed


def parallel_for(data, func):
    processes = []
    with ThreadPoolExecutor() as executor:
        for d in data:
            processes.append(executor.submit(func, d))
    return as_completed(processes)


def parallel_for(data, func, param):
    processes = []
    with ThreadPoolExecutor() as executor:
        for d in data:
            processes.append(executor.submit(func, d, param))
    return as_completed(processes)
