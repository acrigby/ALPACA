# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 17:15:15 2023

@author: aidan
"""
import multiprocessing
from time import sleep
import os

semaphore = multiprocessing.Semaphore(2)

def do_job(id):
    with semaphore:
        print(f"Process={os.getpid()}")
        sleep(1)
    print("Finished job")

def main():
    pool = multiprocessing.Pool(6)
    for job_id in range(6):
        print("Starting job")
        pool.apply_async(do_job, [job_id])
    pool.close()
    pool.join()

if __name__ == "__main__":
    main()