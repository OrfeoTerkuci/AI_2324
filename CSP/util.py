from tqdm import tqdm

progressBars = dict()


def get_new_run_id(initial=False):
    import json
    calls_info = {}
    new_run_id = 0
    if initial:
        with open('function_calls.json', 'r') as file:
            # Json file has structure {run_id: call_count, ...}
            # We want to update the call count for the current run
            try:
                calls_info = json.load(file)
            except:
                calls_info = {}
            new_run_id = len(calls_info)
            file.close()
        calls_info[f"run_{new_run_id}"] = 0
        json.dump(calls_info, open('function_calls.json', 'w'))
        return

    if not initial:
        calls_info = json.load(open('function_calls.json', 'r'))
        new_run_id = len(calls_info) - 1
    return calls_info, new_run_id


def update_json():
    import json
    calls_info, new_run_id = get_new_run_id()
    calls_info[f"run_{new_run_id}"] += 1
    json.dump(calls_info, open('function_calls.json', 'w'))


def monitor(f):
    """ Decorator to time functions and count the amount of calls. """
    def wrapper(*args, **kwargs):
        if f not in progressBars:
            progressBars[f] = tqdm(desc=f.__name__, unit=" calls")
            get_new_run_id(initial=True)
        progress = progressBars[f]
        progress.update(1)

        # Update the json file
        update_json()

        return f(*args, **kwargs)
    return wrapper
