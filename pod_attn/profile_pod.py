import subprocess
import pandas as pd
import os
import io
import numpy as np
import matplotlib.pyplot as plt
import itertools


RATIO_LIST = [8, 16, 32]
PCL_LIST = [64, 256, 1024, 4096]
DCL_LIST = [64, 256, 1024, 4096]
PREFILL_BATCH_LIST = [1, 2, 3, 4, 5, 6]
# PREFILL_BATCH_LIST = [1]
REPEAT_TIMES = 512


# check if this script is running under root priviledge
def check_root():
    if os.geteuid() != 0:
        print("This script must be run as root!")
        exit(1)

def read_csv(content):
    # check the content of the csv file
    lines = content.split('\n')
    head_idx = 0
    while head_idx < len(lines):
        if lines[head_idx].startswith('"'):
            break
        head_idx += 1
    csv_content = '\n'.join(lines[head_idx:])
    # print(csv_content)
    # read the csv file
    df = pd.read_csv(io.StringIO(csv_content), sep=',')
    # turn the metric values into float
    metric_values = df['Metric Value']
    metric_values = metric_values.apply(lambda x: float(''.join(x.split(','))))
    # check if the number of rows is REPEAT_TIMES
    if len(df) % REPEAT_TIMES == 0:
        row_group_size = len(df) // REPEAT_TIMES
        # merge every row group, adding "Metric Value" up
        metric_values = metric_values.values
        metric_values = metric_values.reshape(-1, row_group_size)
        metric_values = metric_values.sum(axis=1)
        df = df.iloc[::row_group_size]
        df['Metric Value'] = metric_values
    else:
        print(f"Warning: the number of rows is not a multiple of {REPEAT_TIMES}")
    # extract "Metric Value" and calculate the average
    avg = metric_values.mean()
    return avg

def run_profile(
    ratio: int, # ratio of decode requests to prefill requests
    p_cl: int, # prefill context length
    p_cs: int, # prefill chunk size
    d_cl: int, # decode context length
):
    print(f"Running profile for wordload: ratio: {ratio}, p_cl: {p_cl}, p_cs: {p_cs}, d_cl: {d_cl}")
    # check if this script is running under root priviledge
    check_root()
    # run the profile script
    command_template = [
        # ncu options
        "/usr/bin/ncu", 
        # "--target-processes", "all",
        "--metrics", "gpu__time_duration.sum",
        "-k", "\"regex:.*fwd.*\"",
        "--csv",
        # python script
        "/home/duchuheng/miniconda3/envs/pod-attn/bin/python", 
        "tests/banner_fig_profile.py",
        "--p_cl", str(p_cl),
        "--p_cs", str(p_cs),
        "--d_cl", str(d_cl),
        "--repeat", str(REPEAT_TIMES),
    ]
    avg_list = []
    for p_bs in PREFILL_BATCH_LIST:
        decode_batch_size = p_bs * ratio
        command = command_template + ["--p_bs", str(p_bs), "--d_bs", str(decode_batch_size)]
        avg_dict = {}
        print(f"  p_bs: {p_bs}, d_bs: {decode_batch_size}.. ", end='', flush=True)
        # run three stages of the profile
        for stage in ['fused', 'prefill', 'decode']:
            command.append("--stage")
            command.append(stage)
            # print the command
            cmd_str = ' '.join(command)
            # run the command
            result = subprocess.run(cmd_str, capture_output=True, shell=True)
            # print(result.stdout.decode())
            # print(result.stderr.decode())
            out, err = result.stdout, result.stderr
            avg = read_csv(out.decode())
            # clear the command
            command.pop()
            command.pop()
            avg_dict[stage] = avg
            print(f"{stage}: {avg:.2f}ns", end=' ')
        print()
        avg_list.append((avg_dict['fused'], avg_dict['prefill'] + avg_dict['decode']))
    print()
    return avg_list


if __name__ == '__main__':
    # set the columns of the dataframe
    df = None
    for ratio, p_cl, d_cl in itertools.product(RATIO_LIST, PCL_LIST, DCL_LIST):
        for p_bs, avgs in zip(PREFILL_BATCH_LIST, run_profile(ratio, p_cl, p_cl, d_cl)):
            print(f'{ratio} {p_cl} {d_cl} {p_bs}: fused {avgs[0]:.2f}ns, flash {avgs[1]:.2f}ns')
            df_new = pd.DataFrame({
                'ratio': [ratio, ratio],
                'p_cl': [p_cl, p_cl],
                'd_cl': [d_cl, d_cl],
                'p_bs': [p_bs, p_bs],
                'method': ['fused', 'flash'],
                'avg': avgs
            })
            print(df_new)
            if df is None:
                df = df_new
            else:
                df = pd.concat([df, df_new], ignore_index=True)
    # save the dataframe
    df.to_csv('profile_pod.csv', index=False)
