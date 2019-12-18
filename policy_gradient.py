import numpy as np
import random
from queue import PriorityQueue
from queue import Queue

### Stochastic Policy Gradients ###

# hyperparameters
H = 32 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-2
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2

# model initialization
D = 5 ###
model = {}
model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
model['W2'] = np.random.randn(H) / np.sqrt(H)

grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # rmsprop memory

def sigmoid(x): 
    return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]


### def prepro(I):
#     """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
#     I = I[35:195] # crop
#     I = I[::2,::2,0] # downsample by factor of 2
#     I[I == 144] = 0 # erase background (background type 1)
#     I[I == 109] = 0 # erase background (background type 2)
#     I[I != 0] = 1 # everything else (paddles, ball) just set to 1
#     return I.astype(np.float).ravel()

def discount_rewards(r):
    # might just hard code this more since last entry is 1 or -1 and rest are 0...
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def policy_forward(x):
    h = np.dot(model['W1'], x)
    h[h<0] = 0 # ReLU nonlinearity
    logp = np.dot(model['W2'], h)
    p = sigmoid(logp)
    return p, h # return probability of taking action 2, and hidden state

def policy_backward(eph, epdlogp):
    """ backward pass. (eph is array of intermediate hidden states) """
    dW2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model['W2'])
    dh[eph <= 0] = 0 # backpro prelu
    dW1 = np.dot(dh.T, epx)
    return {'W1':dW1, 'W2':dW2}

# set up intial state
# env = gym.make("Pong-v0")
# observation = env.reset()
prev_x = None ### dont need but will need something to store prev/rolling stats
xs,hs,dlogps,drs = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0

buffer_size = 5
buffer_times = []
prev_time_average = 0
bprops = [[], [], []]

# set up simulator code
beta = 1 / 0.5 # beta is 1/lambda
interarrival_times = np.random.exponential(scale=beta, size=1000000)
arrival_times = np.cumsum(interarrival_times)
Q = PriorityQueue()
for t in arrival_times:
    Q.put((t, 'arrival'))
Q_f = Queue()
Q_s = Queue()
total_arrivals = 0
total_fastserver_arrivals = 0
ratio = 0
times_in_service = []
S_f = 1.5
S_s = 3.0
avgtimes = 0
prevtime = 0
fatimes = 0
ftimes = []
satimes = 0
stimes = []
arrtimes = []
prevf = 0
prevs = 0
state = [0, 0, 0, 0, 0]

while not Q.empty():
    ### set up input x vector
    next_item = Q.get()
    curr_time = next_item[0]
    event_type = next_item[1]
    if event_type == 'arrival':
        total_arrivals += 1
        arrtimes.append(curr_time - prevtime)
        prevtime = curr_time
        avgtimes = np.mean(arrtimes)
        state[2] = avgtimes
        # preprocess the observation, set input to network to be difference image
        x = np.array(state)
        ### store/update 'prev' variables here (need to add) OR later when we update state?

        # use neural net to determine action
        # forward the policy network and sample an action from the returned probability
        aprob, h = policy_forward(x)
        action = 2 if np.random.uniform() < aprob else 3 # roll the dice!

        # record various intermediates (needed later for backprop)
#         xs.append(x) # observation
#         hs.append(h) # hidden state
#         y = 1 if action == 2 else 0 # a "fake label"
        y = 1
#         dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

        bprop = [x, h, y-aprob]
        
        if action == 2: # send arrival to fast server
            total_fastserver_arrivals += 1
            Q_f.put((curr_time, state)) ### may need to store more info here for future inputs
            # update state
            state[0] += 1
            if Q_f.qsize() == 1:
                # if server queue was empty then this arrival is serviced immediately
                # so we add an exit event right now
                Q.put((curr_time+S_f, 'exit', 'f', S_f, state, bprop))
                times_in_service.append(S_f)
                ftimes.append(S_f)
                fatimes = np.mean(ftimes)
                state[3] = fatimes
        else: # send arrival to slow server
            Q_s.put((curr_time, state))
            # update state
            state[1] += 1
            if Q_s.qsize() == 1:
                # if server queue was empty then this arrival is serviced immediately
                # so we add an exit event right now
                Q.put((curr_time+S_s, 'exit', 's', S_s, state, bprop))
                times_in_service.append(S_s)
                stimes.append(S_s)
                satimes = np.mean(stimes)
                state[4] = satimes
    elif event_type == 'exit':
        # step the environment and get new measurements
        # handle exit event, update state, add new exit event if needed
        server = next_item[2]
        if server == 'f':
            # add time to buffer and update state
            buffer_times.append(next_item[3])
            bprops[0].append(next_item[5][0])
            bprops[1].append(next_item[5][1])
            bprops[2].append(next_item[5][2])
            # update state
            state[0] -= 1
            # remove from the server queue and store its time for use in reward
            rem_item = Q_f.get()
            # determine if we need to add a new exit event for next in server queue
            if Q_f.qsize() > 0:
                # use next item in queue to create exit event
                new_exit_time = curr_time + S_f
                time_in_service = new_exit_time - Q_f.queue[0][0]
                Q.put((new_exit_time, 'exit', 'f', time_in_service, Q_f.queue[0][1], bprop))
                times_in_service.append(time_in_service)
                ftimes.append(time_in_service)
                fatimes = np.mean(ftimes)
                state[3] = fatimes
        else:
            # add time to buffer and update state
            buffer_times.append(next_item[3])
            bprops[0].append(next_item[5][0])
            bprops[1].append(next_item[5][1])
            bprops[2].append(next_item[5][2])
            # update state
            state[1] -= 1
            # remove from the server queue
            rem_item = Q_s.get()
            # determine if we need to add a new exit event for next in server queue
            if Q_s.qsize() > 0:
                # use next item in queue to create exit event
                new_exit_time = curr_time + S_s
                time_in_service = new_exit_time - Q_s.queue[0][0]
                Q.put((new_exit_time, 'exit', 's', time_in_service, Q_s.queue[0][1], bprop))
                times_in_service.append(time_in_service)
                stimes.append(time_in_service)
                satimes = np.mean(stimes)
                state[4] = satimes

    ratio = total_fastserver_arrivals / total_arrivals
        

    if len(buffer_times) == buffer_size: # an episode finished
        episode_number += 1

        # stack together all inputs, hidden states, action gradients, and rewards for this episode
#         epx = np.vstack(xs)
#         eph = np.vstack(hs)
#         epdlogp = np.vstack(dlogps)
        epx = np.vstack(bprops[0])
        eph = np.vstack(bprops[1])
        epdlogp = np.vstack(bprops[2])
        #epr = np.vstack(drs)
        xs,hs,dlogps,drs = [],[],[],[] # reset array memory
        bprops = [[], [], []]

        # compute the discounted reward backwards through time
        ### might have to straight compute reward here (1 or -1)
        if np.mean(buffer_times) < prev_time_average or prev_time_average == 0: # prev == 0 is the first time (no prev exists)
            reward = 1
        else:
            reward = -1
        discounted_epr = np.ones(epdlogp.shape) * reward
        # set next previous time and reset buffer times for next buffer
        prev_time_average = np.mean(buffer_times)
        buffer_times = []
        #discounted_epr = discount_rewards(epr)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        #discounted_epr -= np.mean(discounted_epr)
        #discounted_epr /= np.std(discounted_epr)

        epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
        grad = policy_backward(eph, epdlogp)
        for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch

        # perform rmsprop parameter update every batch_size episodes
        if episode_number % batch_size == 0:
            for k,v in model.items():
                g = grad_buffer[k] # gradient
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

        if episode_number % 100 == 0:
            print(ratio, np.mean(times_in_service))