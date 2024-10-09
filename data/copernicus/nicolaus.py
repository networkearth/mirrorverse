import os
import shutil
import click

from datetime import datetime, timedelta
from time import time
from multiprocessing import Process, Queue

def compose_command(pattern, dataset_id, doi, output_dir):
    regex = pattern.format(year=f'{doi.year}', month=f'{doi.month:02}', day=f'{doi.day:02}')
    return f'copernicusmarine get -i "{dataset_id}" --regex "{regex}" -o {output_dir} --overwrite --force-download'
    
class NicolausCommandException(Exception):
    pass

def run_command(command):
    code = os.system(command)
    if code != 0:
        raise NicolausCommandException(f"Command {command} failed with code {code}")
    
def command_runner(input_queue, output_queue):
    while True:
        command = input_queue.get()
        if command == 'STOP':
            break
        try:
            run_command(command)
            output_queue.put(True)
        except NicolausCommandException as e:
            output_queue.put(e)

def print_remaining_time(remaining_seconds):
    hours = remaining_seconds // 3600
    minutes = (remaining_seconds % 3600) // 60
    seconds = remaining_seconds % 60
    print(f"Time Remaining: {hours}h {minutes}m {seconds}s")


@click.command()
@click.option('--start-date', type=str, required=True)
@click.option('--end-date', type=str, required=True)
@click.option('--local-output-dir', type=str, required=True)
@click.option('--external-output-dir', type=str, required=True)
@click.option('--dataset-id', type=str, required=True)
@click.option('--num-workers', type=int, default=os.cpu_count() - 1)
@click.option('--pattern', type=str, required=True)
def main(start_date, end_date, local_output_dir, external_output_dir, dataset_id, num_workers, pattern):
    doi = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    pull_commands = []
    while doi <= end_date:
        pull_commands.append(compose_command(pattern, dataset_id, doi, local_output_dir))
        doi += timedelta(days=1)

    input_queues = [Queue() for _ in range(num_workers)]
    output_queue = Queue()

    workers = [
        Process(
            target=command_runner,
            args=(input_queue, output_queue)
        )
        for input_queue in input_queues
    ]
    for worker in workers:
        worker.start()


    try:
        assert list(os.listdir(local_output_dir)) == [], f"Local Output Directory {local_output_dir} is not empty"
    except FileNotFoundError:
        pass

    start = time()
    total_commands = len(pull_commands)
    while pull_commands:
        # Prep the Local Output Directory
        print('Clearing Local Output Directory...')
        if os.path.exists(local_output_dir):
            shutil.rmtree(local_output_dir)
        os.mkdir(local_output_dir)

        # Pull the Data from the Copernicus Marine Server
        print('Running Pulls...')
        count = 0
        for input_queue in input_queues:
            if pull_commands:
                input_queue.put(pull_commands.pop(0))
                count += 1
        for _ in range(count):
            result = output_queue.get()
            if isinstance(result, Exception):
                raise result

        # Move the Data to the External Directory
        print('Running Moves...')
        move_commands = []
        for root, _, files in os.walk(local_output_dir):
            for file in files:
                if file.endswith('.nc'):
                    local_path = os.path.join(root, file)
                    external_path = os.path.join(external_output_dir, os.path.relpath(os.path.join(root, file), local_output_dir))
                    os.makedirs(os.path.dirname(external_path), exist_ok=True)
                    move_commands.append(f'mv {local_path} {external_path}')
    
        for input_queue in input_queues:
            if move_commands:
                input_queue.put(move_commands.pop(0))
        for _ in range(count):
            result = output_queue.get()
            if isinstance(result, Exception):
                raise result
            
        commands_completed = total_commands - len(pull_commands)
        time_elapsed = time() - start
        time_remaining = (time_elapsed / commands_completed) * len(pull_commands)
        print(f"Completed {commands_completed}/{total_commands} Commands")
        print_remaining_time(time_remaining)

    print('Cleaning Up...')
    if os.path.exists(local_output_dir):
        shutil.rmtree(local_output_dir)
    os.mkdir(local_output_dir)

    for input_queue in input_queues:
        input_queue.put('STOP')
    for worker in workers:
        worker.join()

if __name__ == '__main__':
    main()