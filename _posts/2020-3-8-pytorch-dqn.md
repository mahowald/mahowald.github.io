---
layout: post
title: "Playing TicTacToe with a DQN, redux"
tags: [deep-q-networks]
author: Matthew Mahowald
mathjax: true
---

In this post, we return again to the world's most challenging game, the apex of strategy and art, a game which has broken minds and machines and left a trail of debris and gibbering madmen along the highway of history.
I refer, of course, to Tic Tac Toe.

Avid readers of this blog (hi, mom!) might recall that we previously attempted Tic Tac Toe using a DQN and the [Keras-RL](https://github.com/keras-rl/keras-rl) package (built on Keras and TensorFlow).
In this post, I'll do much the same, except this time I'll shameless plagiarize [the official PyTorch documentation's DQN tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html) instead.
Grab the [complete code from github here!](https://github.com/mahowald/tictactoe)

## Setting up the game

Once again, we'll use a deep Q network to learn our policies.
I'm choosing this because I want an _off-policy_ batch reinforcement learning algorithm:
In off-policy approaches, we can incorporate a wide variety of actions and their outcomes into our policy learning, including actions that haven't been selected by the current policy.
This is in contrast to on-policy approaches, in which we can only train on actions that have been selected by the current policy.
In Tic Tac Toe, the space of possible board configurations is discrete and relatively small (specifically, there are $19683 = 3^{9}$ possible board states), so we can explore this space relatively easily.

Let's start by definig our environment.
We'll re-use the `TicTacToe` class we defined previously.
Here's a skeleton of this class:

```python
class TicTacToe(gym.Env):
    reward_range = (-np.inf, np.inf)
    observation_space = spaces.MultiDiscrete([2 for _ in range(0, 3*3*3)])
    action_space = spaces.Discrete(9) 

    def seed(self, seed=None):
        pass
    
    def reset(self):
        # [...]
        return obs
    
    def step(self, action):
        # [...]
        return obs, reward, done, {}
    
    def render(self, mode="human"):
        pass
```

As before, the board is represented to the agent as a flattened $3 \times 3 \times 3$ tensor of binary indicators.
The first two dimensions of the unflattened tensor correspond to the board position, and the final dimension indicates whether a space is unoccupied (`0`), occupied by player 1 (`1`), or occupied by player 2 (`2`).
The agent's action is an integer between 0 and 9, indicating which board position the agent wishes to place its tile at.
These correspond to the board positions in the expected way:

```python
[ 0, 1, 2,
  3, 4, 5,
  6, 7, 8 ]
```

The most relevant function defined in `TicTacToe` is the `step` function, which defines the reward structure, as well as how the environment is updated with each move.
Take a look:

```python
def step(self, actions):
        exp = {"state": "in progress"}
        
        # get the current player's action
        # action = actions[self.current_player]
        action = actions
        
        reward = 0
        done = False
        # illegal move
        if self.board[action] != 0:
            reward = -10 # illegal moves are really bad
            done = True
            self.summary["total games"] += 1
            self.summary["illegal moves"] += 1
            return self._one_hot_board(), reward, done, None
        
        self.board[action] = self.current_player + 1
        
        # check if the other player can win on the next turn:
        for streak in self.winning_streaks:
            if ((self.board[streak] == 2 - self.current_player).sum() >= 2) and (self.board[streak] == 0).any():
                reward = -2
                
        # check if we won
        for streak in self.winning_streaks:
            if (self.board[streak] == self.current_player + 1).all():
                reward = 1 # player wins!
                self.summary["total games"] += 1
                self.summary["player {} wins".format(self.current_player)] += 1
                done = True
        # check if we tied, which ends the game
        if (self.board != 0).all():
            reward = 0
            done = True
            self.summary["total games"] += 1
            self.summary["ties"] += 1
        
        # move to the next player
        self.current_player = 1 - self.current_player
        
        return self._one_hot_board(), reward, done, None
```

The basic reward structure is:

* +1 points for winning the game
* -2 points for a move that lets the other player win on the next turn
* -10 points for making an illegal move

## Creating the model

To create the model, we can hew very closely to [the official PyTorch documentation's DQN tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html).
In particular, we can re-use verbatim their `ReplayMemory` class and training loop.
For the policy itself, I opted to mimic the architecture used in the previous blog post: a feedforward neural net with three hidden layers consisting of 128, 256, and 128 neurons, respectively.
I also added an `act` method to produce an output for a specific input.

Here's the class:

```python
class Policy(nn.Module):

    def __init__(self, n_inputs=3*9, n_outputs=9):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(n_inputs, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, n_outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x
    
    def act(self, state):
        with torch.no_grad():
            return self.forward(state).max(1)[1].view(1, 1)
```

## Training the model

To train the model, we'll continue closely following PyTorch's tutorials.
First, let's initialize the policy net and target nets:

```python
policy = Policy(n_inputs=3 * 9, n_outputs=9).to(device)
target = Policy(n_inputs=3 * 9, n_outputs=9).to(device)
target.load_state_dict(policy.state_dict())
target.eval()
```

Recall that during training, we are trying to minimize the loss function

$$
(R_{t+1} + \gamma_{t+1} \max_{\alpha '} q_{\bar{\theta}} (S_{t+1}, a ') - q_{\theta} (S_{t}, A_{t}))^{2}
$$

Here, $q_{\bar{\theta}}$ is the target network, and $q_{\theta}$ is the policy network; we are updating only the $\theta$ parameters with each optimization step.

For completeness, I'll include the update step here, but note that this is again from the excellent PyTorch DQN tutorial:

```python
def optimize_model(
    device: torch.device,
    optimizer: optim.Optimizer,
    policy: Policy,
    target: Policy,
    memory: ReplayMemory,
    batch_size: int,
    gamma: float,
):
    """Model optimization step, copied verbatim from the Torch DQN tutorial.
    
    Arguments:
        device {torch.device} -- Device
        optimizer {torch.optim.Optimizer} -- Optimizer
        policy {Policy} -- Policy net
        target {Policy} -- Target net
        memory {ReplayMemory} -- Replay memory
        batch_size {int} -- Number of observations to use per batch step
        gamma {float} -- Reward discount factor
    """
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy
    state_action_values = policy(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(batch_size, device=device)
    next_state_values[non_final_mask] = target(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(
        state_action_values, expected_state_action_values.unsqueeze(1)
    )

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
```

Where we diverge from the tutorial is in our training loop:
I decided to use a linearly annealed epsilon greedy policy (in which, during training, the model chooses a random action with probability `eps`, and this parameter is linearly interpolated toward a minimum value).
In addition, we need a mechanism to handle the actions of player 2.
In the `keras-rl` implementation, I handled this through self-play: player 2 was a copy of the model, operating under the same conditions as player 1.
In practice, self-play of this form can lead to non-optimal strategy learning as the model learns how to beat itself, rather than the optimal player.
To account for this, this time around I've set the agent up to play against a random player.
The random player picks a random legal move each turn:

```python
def select_dummy_action(state: np.array) -> int:
    # Select a random (valid) move, given a board state.
    state = state.reshape(3, 3, 3)
    open_spots = state[:, :, 0].reshape(-1)
    p = open_spots / open_spots.sum()
    return np.random.choice(np.arange(9), p=p)
```

Here's also the function we'll use to select the model action:

```python
def select_model_action(
    device: torch.device, model: Policy, state: torch.tensor, eps: float
) -> Tuple[torch.tensor, bool]:

    sample = random.random()
    if sample > eps:
        return model.act(state), False
    else:
        return (
            torch.tensor(
                [[random.randrange(0, 9)]],
                device=device,
                dtype=torch.long,
            ),
            True,
        )
```
Note that I'm allowing the model to pick any random move, not just random valid moves---it'd be interesting to see if learning improves or worsens by not allowing illegal moves during training.
My intuition is that by not allowing illegal moves to be selected randomly, the model will see fewer illegal transitions (and corresponding negative rewards) in the replay buffer, resulting in a model that plays worse.


Finally, here's the full fit loop:

```python
n_steps = 500_000,
batch_size = 128,
gamma = 0.99,
eps_start = 1.0,
eps_end = 0.1,
eps_steps = 200_000

optimizer = optim.Adam(policy.parameters(), lr=1e-3)
memory = ReplayMemory(50_000)

env = TicTacToe()
state = torch.tensor([env.reset()], dtype=torch.float).to(device)

for step in range(n_steps):
    t = np.clip(step / eps_steps, 0, 1)
    eps = (1 - t) * eps_start + t * eps_end

    action, was_random = select_model_action(device, policy, state, eps)
    if was_random:
        _randoms += 1
    next_state, reward, done, _ = env.step(action.item())

    # player 2 goes
    if not done:
        next_state, _, done, _ = env.step(select_dummy_action(next_state))
        next_state = torch.tensor([next_state], dtype=torch.float).to(device)
    if done:
        next_state = None

    memory.push(state, action, next_state, torch.tensor([reward], device=device))

    state = next_state
    optimize_model(
        device=device,
        optimizer=optimizer,
        policy=policy,
        target=target,
        memory=memory,
        batch_size=batch_size,
        gamma=gamma,
    )
    if done:
        state = torch.tensor([env.reset()], dtype=torch.float).to(device)
    if step % target_update == 0:
        target.load_state_dict(policy.state_dict())
```

## Testing it out

Recall that the board positions are indexed as follows:
```
0|1|2
-----
3|4|5
-----
6|7|8
```

In my initial play against the agent, it opens with position 6:

```
0|0|0
-----
0|0|0
-----
1|0|0
```

Being a veteran tic-tac-toe player myself, I then take position 4.
The bot responds by forcing me to take position 8:

```
0|0|0
-----
0|2|0
-----
1|1|0
```

I oblige:

```
1|0|0
-----
0|2|0
-----
1|1|2
```

Again, my move is forced---this time, to position 3:

```
1|1|0
-----
2|2|0
-----
1|1|2
```

Here, the bot makes its first mistake: it takes position 1 instead of blocking me at position 5.
I eke out the win:

```
1|1|0
-----
2|2|2
-----
1|1|2
```

And that's about it!
Next time, we'll take a look at pitting the PyTorch DQN agent against the Keras-RL agent, and see who is the victor.
In the meantime, a full containerized implementation of this agent is available [on my GitHub.](https://github.com/mahowald/tictactoe)