import numpy as np
import random
from queue import PriorityQueue
from queue import Queue

def run_sim(f):
    beta = 1 / 1.5 # beta is 1/lambda
#     interarrival_times = np.random.exponential(scale=beta, size=10000)
    interarrival_times = np.random.lognormal(4.5, 4, size=10000)
    arrival_times = np.cumsum(interarrival_times)

    Q = PriorityQueue()
    for t in arrival_times:
        Q.put((t, 'arrival'))

    Q_f = Queue()
    Q_s = Queue()
    total_arrivals = 0
    total_fastserver_arrivals = 0
    times_in_service = []
    while not Q.empty():
        next_item = Q.get()
        if next_item[1] == 'arrival':
            total_arrivals += 1
            # add it to appropriate server queue
            if total_fastserver_arrivals / total_arrivals < f:
                total_fastserver_arrivals += 1
                Q_f.put(next_item[0])
                if Q_f.qsize() == 1:
                    # if server queue is empty then this arrival is serviced immediately
                    # so we add an exit event right now
                    # deterministic
                    R = 1.5
                    # pareto
#                     a = 2.25
#                     k = 5/6
#                     R = (np.random.pareto(a) + 1) * k
                    Q.put((next_item[0]+R, 'exit', 'f', R))
                    times_in_service.append(R)
            else:
                Q_s.put(next_item[0])
                if Q_s.qsize() == 1:
                    # if server queue is empty then this arrival is serviced immediately
                    # so we add an exit event right now
                    # deterministic
                    R = 3
                    # pareto
#                     a = 2.25
#                     k = 5/3
#                     R = (np.random.pareto(a) + 1) * k
                    Q.put((next_item[0]+R, 'exit', 's', R))
                    times_in_service.append(R)
        elif next_item[1] == 'exit':
            # add an exit even from the next arrival in the server queue
            if next_item[2] == 'f':
                # remove from the server queue
                rem_item = Q_f.get()
                if Q_f.qsize() > 0:
                    # use next item in queue to create exit event
                    # deterministic
                    R = 1.5
                    # pareto
#                     a = 2.25
#                     k = 5/6
#                     R = (np.random.pareto(a) + 1) * k
                    new_exit_time = next_item[0] + R
                    time_in_service = new_exit_time - Q_f.queue[0]
                    Q.put((new_exit_time, 'exit', 'f', time_in_service))
                    times_in_service.append(time_in_service)
            else:
                # remove from the server queue
                rem_item = Q_s.get()
                if Q_s.qsize() > 0:
                    # use next item in queue to create exit event
                    # deterministic
                    R = 3
                    # pareto
#                     a = 2.25
#                     k = 5/3
#                     R = (np.random.pareto(a) + 1) * k
                    new_exit_time = next_item[0] + R
                    time_in_service = new_exit_time - Q_s.queue[0]
                    Q.put((new_exit_time, 'exit', 's', time_in_service))
                    times_in_service.append(time_in_service)
    return np.mean(times_in_service)

def run_sim_batch(f):
    beta = 1 / 0.1 # beta is 1/lambda
#     interarrival_times = np.random.exponential(scale=beta, size=10000)
    interarrival_times = np.random.lognormal(4.5, 4, size=10000)
    arrival_times = np.cumsum(interarrival_times)

    Q = PriorityQueue()
    for t in arrival_times:
        # generate batch arrivals
        x = np.random.randint(6)
        delta = 0
        for i in range(x):
            Q.put((t+delta, 'arrival'))
            delta += 0.000001

    Q_f = Queue()
    Q_s = Queue()
    total_arrivals = 0
    total_fastserver_arrivals = 0
    times_in_service = []
    while not Q.empty():
        next_item = Q.get()
        if next_item[1] == 'arrival':
            total_arrivals += 1
            # add it to appropriate server queue
            if total_fastserver_arrivals / total_arrivals < f:
                total_fastserver_arrivals += 1
                Q_f.put(next_item[0])
                if Q_f.qsize() == 1:
                    # if server queue is empty then this arrival is serviced immediately
                    # so we add an exit event right now
                    # deterministic
#                     R = 1.5
                    # pareto
                    a = 2.25
                    k = 5/6
                    R = (np.random.pareto(a) + 1) * k
                    Q.put((next_item[0]+R, 'exit', 'f', R))
                    times_in_service.append(R)
            else:
                Q_s.put(next_item[0])
                if Q_s.qsize() == 1:
                    # if server queue is empty then this arrival is serviced immediately
                    # so we add an exit event right now
                    # deterministic
#                     R = 3.0
                    # pareto
                    a = 2.25
                    k = 5/3
                    R = (np.random.pareto(a) + 1) * k
                    Q.put((next_item[0]+R, 'exit', 's', R))
                    times_in_service.append(R)
        elif next_item[1] == 'exit':
            # add an exit even from the next arrival in the server queue
            if next_item[2] == 'f':
                # remove from the server queue
                rem_item = Q_f.get()
                if Q_f.qsize() > 0:
                    # use next item in queue to create exit event
                    # deterministic
#                     R = 1.5
                    # pareto
                    a = 2.25
                    k = 5/6
                    R = (np.random.pareto(a) + 1) * k
                    new_exit_time = next_item[0] + R
                    time_in_service = new_exit_time - Q_f.queue[0]
                    Q.put((new_exit_time, 'exit', 'f', time_in_service))
                    times_in_service.append(time_in_service)
            else:
                # remove from the server queue
                rem_item = Q_s.get()
                if Q_s.qsize() > 0:
                    # use next item in queue to create exit event
                    # deterministic
#                     R = 3.0
                    # pareto
                    a = 2.25
                    k = 5/3
                    R = (np.random.pareto(a) + 1) * k
                    new_exit_time = next_item[0] + R
                    time_in_service = new_exit_time - Q_s.queue[0]
                    Q.put((new_exit_time, 'exit', 's', time_in_service))
                    times_in_service.append(time_in_service)
    return np.mean(times_in_service)

fs = np.linspace(0.5, 1, 51)
for f in fs:
    total = 0
    for i in range(10):
        total += run_sim(f)
    print(f, total/10)