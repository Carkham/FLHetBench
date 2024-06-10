import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple
from collections import defaultdict
import json

with open("data/cached_timers.json", "r") as f:
    traces_old = json.load(f)
traces = {}
for key, value in traces_old.items():
    if value['ready_time'] != []:
        traces[key] = value


def time_sequence_compute(trace, start, end):
    start = int(start)
    end = int(end)
    time_sequence = np.zeros(int(end - start))
    for s, e in trace:
        time_sequence[int(s - start):int(e - start + 1)] = 1
    return time_sequence


time_sequences = {}
for key, value in traces.items():
    start = value['trace_start']
    end = value['trace_end']
    trace = value['ready_time']
    time_sequence = time_sequence_compute(trace, start, end)
    time_sequences[key] = time_sequence

import math

non_zero_time_seqs = {k: np.nonzero(v)[0] for k, v in time_sequences.items()}
time_seq_lens = {k: len(v) for k, v in time_sequences.items()}


def monte_carlo_process(
        device_speeds: List[List[float]],
        state_trace_ids: list,
        ddl: float = 120,
        t_cost: float = None,
        t_break: float = 20,
        selected_num_per_round: int = 20,
        model_size: float = 85800194 * 4 / 1024,
        r: float = 0.2,
        rounds: int = 3000,
        trips: int = 10000,
        training_strategy: str = "deadline-based",
        seed: int = 42):
    """
    Calculate the $\mathcal{S}$ for DevMC-R, StatMC-R and InterMC-R
    :param device_speeds: list, each element is a list of speed tested with MobiPerf
    :param state_trace_ids: list, raw data from cached_timers.json includes 'trace_start', 'trace_end' and 'ready_time'
    :param ddl: falot, report deadline
    :param t_cost: float, $t_{cost}$ when assessing only state heterogeneity. Set None if device heterogeneity enables.
    :param t_break: float, cost for server allocation and aggregation
    :param selected_num_per_round: int, number of selected clients per round
    :param model_size: float, model size
    :param r: float, r used for quantile
    :param rounds: int, number of communication rounds
    :param trips: int, number of total trips
    :param training_strategy: str, training strategy for monte carlo process
    :param seed: int, random seed
    :return:
    """

    def is_avail(clnt_id, t):
        trace_start = int(traces[clnt_id]['trace_start'])
        trace_end = int(traces[clnt_id]['trace_end'])
        time_sequence = time_sequences[clnt_id]
        t_idx = int(t - trace_start) % (int(trace_end - trace_start))
        t_value = time_sequence[t_idx]
        return t_idx, t_value

    def state_t_cost(state_trace_id, index, target_cost):
        time_seq_len = len(time_sequences[state_trace_id])
        non_zero_time_seq = non_zero_time_seqs[state_trace_id]
        target_cost = math.ceil(target_cost)

        if target_cost <= 0:
            return 0
        else:
            cur_pos = np.where(non_zero_time_seq == index)[0]
            rest_time = len(non_zero_time_seq) - cur_pos
            if rest_time >= target_cost:
                return int(non_zero_time_seq[(target_cost + cur_pos - 1) % len(non_zero_time_seq)]) - index + 1
            else:
                target_cost = target_cost - rest_time
                res_step = time_seq_len - index
                res_step += int(non_zero_time_seq[(target_cost - 1) % len(non_zero_time_seq)]) + 1
                res_step += ((target_cost - 1) // len(non_zero_time_seq)) * time_seq_len
            return int(res_step)

    rng = np.random.default_rng(seed=seed)
    success_time = defaultdict(int)  # $S_i$
    t = 0
    if training_strategy != "deadline-based" and training_strategy != "readiness-based":
        raise ValueError("training stratety must be one of ['deadline-based', 'readiness-based']")
    cur_round, cur_trip = 0, 0
    while True:
        t = t + t_break

        if state_trace_ids is not None:
            ready_clnt_ids = []
            for i, state_trace_id in enumerate(state_trace_ids):
                _, t_value = is_avail(state_trace_id, t)
                if t_value == 1:
                    ready_clnt_ids.append(i)
        else:
            ready_clnt_ids = list(range(len(device_speeds)))

        selected_clnt_ids = None
        if len(ready_clnt_ids) > 0:
            if len(ready_clnt_ids) >= selected_num_per_round:
                selected_clnt_ids = rng.choice(ready_clnt_ids, selected_num_per_round, replace=False)
            else:
                selected_clnt_ids = [i for i in ready_clnt_ids]
        else:
            selected_clnt_ids = []

        if len(selected_clnt_ids) > 0:
            end_t_list = []
            for id in selected_clnt_ids:
                if device_speeds is not None:  # t_cost_i based on Dev
                    speed_list = device_speeds[id]
                    down_speed, up_speed = rng.choice(speed_list, 2, replace=True)
                    t_cost_i = model_size / down_speed + model_size / up_speed
                else:  # t_cost_i from prior uniform distribution
                    t_cost_i = t_cost
                if state_trace_ids is not None:
                    state_trace_id = state_trace_ids[id]
                    t_idx, _ = is_avail(state_trace_id, t)
                    t_cost_i = state_t_cost(state_trace_id, t_idx, t_cost_i)

                if training_strategy == "deadline-based":
                    if t_cost_i > ddl:
                        e = t + ddl
                    else:
                        e = t + t_cost_i
                        success_time[id] += 1

                    end_t_list.append(e)
                elif training_strategy == "readiness-based":
                    end_t_list.append(t + t_cost_i)

            if training_strategy == "deadline-based":
                t = max(end_t_list)
            elif training_strategy == "readiness-based":
                t = np.percentile(end_t_list, (1 - r) * 100)

        cur_trip += len(selected_clnt_ids)
        cur_round += 1
        if cur_round >= rounds and training_strategy == "deadline-based":
            return success_time
        if cur_trip >= trips and training_strategy == "readiness-based":
            return t


def DevMC_R(device_speeds: List[List[float]],
            ddl: int = 120,
            rounds: int = 3000,
            selected_num_per_round: int = 20,
            model_size: float = 85800194 * 4 / 1024,
            c: int = 600,
            seed: int = 42):
    """
    Calculate the DevMC_R with Monte Carlo methods
    :param device_speeds: list, each element is a list of speed tested with MobiPerf
    :param ddl: int, report deadline
    :param rounds: int, number of communication rounds
    :param selected_num_per_round: int, number of selected clients per round
    :param model_size: float, model size
    :param c: S_{ideal}
    :return:
    """
    success_times = monte_carlo_process(device_speeds=device_speeds,
                                        state_trace_ids=None,
                                        ddl=ddl, rounds=rounds, seed=seed,
                                        selected_num_per_round=selected_num_per_round,
                                        model_size=model_size, training_strategy="deadline-based")
    score_list = [min(v, c) for v in success_times.values()]
    expected_score_list = [c] * len(device_speeds)
    score = np.log(np.sum(score_list) + 1) / np.log(np.sum(expected_score_list) + 1)

    return score


def DevMC_T(device_speeds: List[List[float]],
            r: float = 0.2,
            trips: int = 10000,
            t_break: int = 20,
            selected_num_per_round: int = 20,
            model_size: float = 85800194 * 4 / 1024,
            seed: int = 42):
    """
    Calculate the DevMC_T with Monte Carlo methods
    :param device_speeds: list, each element is a list of speed tested with MobiPerf
    :param r: float, report deadline
    :param trips: int, number of total trips
    :param t_break: float, cost for server allocation and aggregation
    :param selected_num_per_round: int, number of selected clients per round
    :param model_size: float, model size
    :return:
    """
    t = monte_carlo_process(device_speeds=device_speeds,
                               state_trace_ids=None,
                               selected_num_per_round=selected_num_per_round,
                               trips=trips, r=r, model_size=model_size, seed=seed,
                               training_strategy="readiness-based")

    return t / (trips * t_break / selected_num_per_round)


def StatMC_R(state_trace_ids: list,
             ddl: int = 120,
             rounds: int = 3000,
             selected_num_per_round: int = 20,
             model_size: float = 85800194 * 4 / 1024,
             c: int = 600,
             seed: int = 42):
    """
    Calculate the StatMC_R with Monte Carlo methods
    :param state_trace_ids: list, each element is a list of speed tested with MobiPerf
    :param ddl: int, report deadline
    :param rounds: int, number of communication rounds
    :param selected_num_per_round: int, number of selected clients per round
    :param model_size: float, model size
    :param c: S_{ideal}
    :return:
    """
    metric_list = []
    for t_cost in np.arange(10, ddl + 10, 10):
        success_times = monte_carlo_process(device_speeds=None,
                                            state_trace_ids=state_trace_ids,
                                            ddl=ddl, rounds=rounds, t_cost=t_cost, seed=seed,
                                            selected_num_per_round=selected_num_per_round,
                                            model_size=model_size, training_strategy="deadline-based")
        success_time_list = [min(v, c) for v in success_times.values()]
        expected_time_list = [c] * len(state_trace_ids)
        metric = np.log(np.sum(success_time_list) + 1) / np.log(np.sum(expected_time_list) + 1)
        metric_list.append(metric)

    return np.mean(metric_list)


def StatMC_T(state_trace_ids: list,
             r: float = 0.2,
             trips: int = 10000,
             t_break: int = 20,
             t_cost_max: int = 240,
             selected_num_per_round: int = 20,
             model_size: float = 85800194 * 4 / 1024,
             seed: int = 42):
    """
    Calculate the StatMC_T with Monte Carlo methods
    :param state_trace_ids: list, each element is a list of speed tested with MobiPerf
    :param r: float, report deadline
    :param trips: int, number of total trips
    :param t_break: float, cost for server allocation and aggregation
    :param t_cost_max: upper bound for t_cost
    :param selected_num_per_round: int, number of selected clients per round
    :param model_size: float, model size
    :return:
    """
    metric_list = []
    for t_cost in np.arange(0, t_cost_max + 10, 10):
        t = monte_carlo_process(device_speeds=None,
                                state_trace_ids=state_trace_ids,
                                selected_num_per_round=selected_num_per_round,
                                trips=trips, r=r, t_cost=t_cost, seed=seed,
                                model_size=model_size, training_strategy="readiness-based")
        metric_list.append(t / (trips * t_break / selected_num_per_round))

    return np.mean(metric_list)


def InterMC_R(device_speeds: List[List[float]],
              state_trace_ids: list,
              ddl: int = 120,
              rounds: int = 3000,
              selected_num_per_round: int = 20,
              model_size: float = 85800194 * 4 / 1024,
              c: int = 600,
              seed: int = 42):
    success_times = monte_carlo_process(device_speeds=device_speeds,
                                        state_trace_ids=state_trace_ids,
                                        ddl=ddl, rounds=rounds, seed=seed,
                                        selected_num_per_round=selected_num_per_round,
                                        model_size=model_size, training_strategy="deadline-based")
    score_list = [min(v, c) for v in success_times.values()]
    expected_score_list = [c] * len(device_speeds)
    score = np.log(np.sum(score_list) + 1) / np.log(np.sum(expected_score_list) + 1)

    return score


def InterMC_T(device_speeds: List[List[float]],
              state_trace_ids: list,
              r: float = 0.2,
              trips: int = 10000,
              t_break: int = 20,
              selected_num_per_round: int = 20,
              model_size: float = 85800194 * 4,
              seed: int = 42):
    t = monte_carlo_process(device_speeds=device_speeds,
                               state_trace_ids=state_trace_ids,
                               selected_num_per_round=selected_num_per_round,
                               trips=trips, r=r, model_size=model_size, seed=seed,
                               training_strategy="readiness-based")
    return t / (trips * t_break / selected_num_per_round)


if __name__ == "__main__":
    from tqdm import tqdm
    import json
    import random
    import numpy as np

    with open("data/dev_case1.json") as f:
        dev_case_1 = json.load(f)

    dev_speeds_case_1 = [[speed / 8 for speed in item['tcp_speed_results']] for item in dev_case_1]
    print(dev_speeds_case_1)

    with open("data/dev_case2.json") as f:
        dev_case_2 = json.load(f)

    dev_speeds_case_2 = [[speed / 8 for speed in item['tcp_speed_results']] for item in dev_case_2]
    print(dev_speeds_case_2)

    with open("data/state_case1.json") as f:
        stat_case_1 = json.load(f)

    with open("data/state_case2.json") as f:
        stat_case_2 = json.load(f)

    # StatMC-R
    statmc_r_case_1_list = []
    statmc_r_case_2_list = []

    statmc_r_case_1_list.append(
        StatMC_T(stat_case_1, r=0.2, trips=10000, seed=99999, model_size=85800194 * 4 / 1024))
