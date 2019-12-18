import numpy as np
import random
from queue import PriorityQueue
from queue import Queue

class Event:
    def __init__(self, event_type, time,
                 server=None, time_in_service=None, state_at_arrival=None):
        # both arrival and exit events have these properties
        self.type = event_type
        self.time = time # for arrivals: arrival time, for exits: exit time
        # exit events have these additional properties (will be None for arrival events)
        self.server = server
        self.time_in_service = time_in_service
        self.state_at_arrival = state_at_arrival
    
    def __lt__(self, other):
        return self.time < other.time

class Server:
    def __init__(self, service, params):
        self.Q = Queue()
        self.size = 0
        self.arrival_time = None
        self.arrival_state = None
        self.service = service
        self.params = params
    
    def pop(self):
        self.Q.get()
        self.size = self.Q.qsize()
        if self.size > 0:
            self.arrival_time = self.Q.queue[0][0]
            self.arrival_state = self.Q.queue[0][1]
        else:
            self.arrival_time = None
            self.arrival_state = None
    
    def put(self, time, state):
        self.Q.put((time, state))
        self.size = self.Q.qsize()
        self.arrival_time = self.Q.queue[0][0]
        self.arrival_state = self.Q.queue[0][1]
    
    def generate_service_time(self):
        if self.service == 'deterministic':
            return self.deterministic_service_time()
        elif self.service == 'pareto':
            return self.pareto_service_time()
        elif self.service == 'lognormal':
            return self.lognormal_service_time()
    
    def deterministic_service_time(self):
        return self.params[0]
    
    def pareto_service_time(self):
        a = self.params[0]
        k = self.params[1]
        return (np.random.pareto(a) + 1) * k

##### Q Learning #####
import random

max_q_size = 10 + 2 # 0-10 and 10+ is 12 states
state_space_size = max_q_size**2 # 2 queues
action_space_size = 2 # send arrival to the fast server or slow server
q_table = np.zeros([state_space_size, action_space_size])
times_table = np.zeros([state_space_size, action_space_size])

# Hyperparameters
alpha = 0.2
gamma = 0.1
epsilon = 0.8

# For plotting metrics
all_epochs = []
all_penalties = []

all_times = []
all_ratios = []

batch_arrivals = False

for i in range(1, 51):
    # reset the whole simulation
    state = 0
    
    beta = 1 / 0.5 # beta is 1/lambda
    interarrival_times = np.random.exponential(scale=beta, size=10000)
#     interarrival_times = np.random.lognormal(0.5, 4, size=10000)
    arrival_times = np.cumsum(interarrival_times)
    Q = PriorityQueue()
    for t in arrival_times:
        if batch_arrivals:
            # generate batch arrivals
            x = np.random.randint(6)
            delta = 0
            for i in range(x):
                Q.put(Event('arrival', t+delta))
                delta += 0.000001
        else:
            Q.put(Event('arrival', t))
    S_f = Server('deterministic', [1.5])
    S_s = Server('deterministic', [3.0])
#     S_f = Server('pareto', [2.25, 5/6])
#     S_s = Server('pareto', [2.25, 5/3])
    total_arrivals = 0
    total_fastserver_arrivals = 0
    ratio = 0
    times_in_service = []
    
    
    while not Q.empty(): # while not done
        # pop next event in even queue
        # if it is an arrival, produce an action and store this for future reward computation
        # if it is a reward, produce a reward based off the original action
        event = Q.get()
        if event.type == 'arrival':
            total_arrivals += 1
            # generate an action to determine which server to add to
            if random.uniform(0, 1) < epsilon:
                # Explore action space
                action = random.choice(range(action_space_size))
            else:
                # Exploit learned values
                action = np.argmax(q_table[state])
            if action == 0: # send arrival to fast server
                total_fastserver_arrivals += 1
                S_f.put(event.time, state)
                if S_f.size == 1:
                    # if server queue was empty then this arrival is serviced immediately
                    # so we add an exit event right now
                    service_time = S_f.generate_service_time()
                    exit_time = event.time + service_time
                    Q.put(Event('exit', exit_time, 'f', service_time, state))
                    times_in_service.append(service_time)
                    times_table[state, action] = service_time
            else: # send arrival to slow server
                S_s.put(event.time, state)
                if S_s.size == 1:
                    # if server queue was empty then this arrival is serviced immediately
                    # so we add an exit event right now
                    service_time = S_s.generate_service_time()
                    exit_time = event.time + service_time
                    Q.put(Event('exit', exit_time, 's', service_time, state))
                    times_in_service.append(service_time)
                    times_table[state, action] = service_time
            # update state since we just took an action
            state = max_q_size * np.min([S_s.size, max_q_size-1]) + np.min([S_f.size, max_q_size-1])
            
        elif event.type == 'exit':
            
            # compute reward and update q_table
            oldstate = event.state_at_arrival
            server = event.server
            if server == 'f':
                oldaction = 0
            else:
                oldaction = 1

            if times_table[oldstate, oldaction] == 0: # first time this has happened, reward is neutral
                reward = 0
            else:
                if event.time_in_service <= np.min(times_table[oldstate]):
                    reward = 1
                else:
                    reward = -1
            
            old_value = q_table[oldstate, oldaction]
            if oldaction == 0:
                if (oldstate+1) % max_q_size == 0:
                    next_state = oldstate
                else:
                    next_state = oldstate + 1
            else:
                if oldstate >= (max_q_size**2 - max_q_size):
                    next_state = oldstate
                else:
                    next_state = oldstate + max_q_size
            next_max = np.max(q_table[next_state])
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[oldstate, oldaction] = new_value
            
            # add an exit even from the next arrival in the server queue
            if event.server == 'f':
                rem_item = S_f.pop()
                if S_f.size > 0:
                    # use next item in queue to create exit event
                    service_time = S_f.generate_service_time()
                    exit_time = event.time + service_time
                    time_in_service = exit_time - S_f.arrival_time
                    Q.put(Event('exit', exit_time, 'f', time_in_service, S_f.arrival_state))
                    times_in_service.append(time_in_service)
                    times_table[S_f.arrival_state, 0] = time_in_service
            else:
                # remove from the server queue
                rem_item = S_s.pop()
                if S_s.size > 0:
                    # use next item in queue to create exit event
                    service_time = S_s.generate_service_time()
                    exit_time = event.time + service_time
                    time_in_service = exit_time - S_s.arrival_time
                    Q.put(Event('exit', exit_time, 's', time_in_service, S_s.arrival_state))
                    times_in_service.append(time_in_service)
                    times_table[S_s.arrival_state, 1] = time_in_service
            # update state since we just changed a queue
            state = max_q_size * np.min([S_s.size, max_q_size-1]) + np.min([S_f.size, max_q_size-1])
            
        ratio = total_fastserver_arrivals / total_arrivals

        
    if i % 1 == 0:
#         epsilon /= 2
        epsilon *= 2/3
        if epsilon < 0.00001:
            epsilon = 0
        print(ratio, np.mean(times_in_service))
        all_ratios.append(ratio)
        all_times.append(np.mean(times_in_service))
        times_in_service = []

print("Training finished.\n")