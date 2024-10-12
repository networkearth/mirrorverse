import os

import click
import pandas as pd
from multiprocessing import Process, Queue
from tqdm import tqdm

import haven.db as db

def job(table, input_queue, output_queue):
    while True:
        file_path = input_queue.get()
        if file_path == 'STOP':
            break
        try:
            df = pd.read_parquet(file_path)
            db.write_data(df, table, ['h3_resolution', 'region', 'date'])
            output_queue.put(True)
        except Exception as e:
            output_queue.put(e)

@click.command()
@click.option('--input-directory', required=True, type=str)
@click.option('--table', required=True, type=str)
@click.option('--num_workers', required=True, type=int)
def main(input_directory, table, num_workers):
    paths = []
    for path in os.listdir(input_directory):
        if path.endswith('.parquet'):
            paths.append(os.path.join(input_directory, path))
    paths = sorted(paths)

    input_queues = [Queue() for _ in range(num_workers)]
    output_queue = Queue()

    workers = [
        Process(
            target=job,
            args=(table, input_queue, output_queue)
        )
        for input_queue in input_queues
    ]

    for worker in workers:
        worker.start()
    
    for i, path in enumerate(paths):
        input_queues[i % num_workers].put(path)

    for path in tqdm(paths):
        result = output_queue.get()
        if isinstance(result, Exception):
            raise result
        
    for input_queue in input_queues:
        input_queue.put('STOP')

    for worker in workers:
        worker.join()

if __name__ == '__main__':
    main()