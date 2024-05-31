import multiprocessing
import time

def compute_function(arg1, arg2, arg3):
    """A sample function that takes three arguments and computes some operation."""
    # Just a placeholder operation, replace it with actual computations
    result = arg1 + arg2 + arg3
    return result

def main():
    # Define a list of tuples, each containing the set of arguments needed for `compute_function`
    inputs = [(1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12), (13, 14, 15),
              (16, 17, 18), (19, 20, 21), (22, 23, 24), (25, 26, 27), (28, 29, 30)]

    # Measure the time for multiprocessing
    start_time = time.time()
    with multiprocessing.Pool(processes=8) as pool:
        # Use starmap to apply 'compute_function' to the 'inputs'
        results = pool.starmap(compute_function, inputs)
    end_time = time.time()
    print(f"Multiprocessing time: {end_time - start_time} seconds")

    # # Measure the time for single processing
    # start_time = time.time()
    # results = [compute_function(*args) for args in inputs]
    # end_time = time.time()
    # print(f"Single processing time: {end_time - start_time} seconds")

if __name__ == '__main__':
    main()