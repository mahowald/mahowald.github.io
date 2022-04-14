---
layout: post
title: "Playing TicTacToe with a DQN"
tagline: "A curious game"
tags: [deep-q-networks]
author: Matthew Mahowald
mathjax: true
---


With the possible exception of computer vision, reinforcement
learning has probably captured more of the public imagination
than any other area of data science, artificial intelligence,
or machine learning. It perhaps most closely mirrors what we
think of as intelligence: an environment is observed, the
machine takes action, and learns from the consequences of those
actions. In recent years, huge advances in reinforcement learning
(such as [Deep Mind's AlphaZero](https://deepmind.com/blog/article/alphazero-shedding-new-light-grand-games-chess-shogi-and-go))
have allowed machines to match or exceed the greatest human players
in challenging, difficult games like Chess, Go, or DotA. With this
in mind, I turned my attention toward the most challenging game of
all: tic tac toe. In this post, I'll train a deep Q network (DQN)
to play tic tac toe. It will learn to play entirely by playing against
itself.

## A little background

Reinforcement learning has a somewhat-deserved reputation for having
a rather high barrier to entry (even relative to other areas of
machine learning). A large part of this stems from the fact that the
training process follows a different pattern than other areas of
machine learning. In this post, I'll focus on [deep Q networks](https://www.nature.com/articles/nature14236),
but I'll try to keep the content broadly applicable to other
reinforcement learning models (e.g. DDPGs) as well. 

The models we'll build are called _agents_, and they define functions
that map _states_ to _actions_. This mapping (the strategy used to
determine the next action based on the current state) is called a
_policy_. Often, the policies are learned through _Q-learning_, which
is a process for learning what the optimal reward resulting from a given
action in a given state will be. The "Q" in Q-learning refers to
the Q-value, or discounted future reward, for an action given the current
state of the agent and environment. This is defined recursively
through the Bellman Equation:

$$
Q(s, a) := r(s,a) + \gamma \max_{a} Q(s', a)
$$

In this equation, $s$ and $a$ are the current state and action, respectively,
$s'$ is the state resulting from the action $a$, and $\gamma$ is a "discount factor"
(a real number typically between 0 and 1) that is used to decrease the weight
of future rewards. This equation is greedy and recursive, so if a reward $r$ could be
obtained in three actions (and each one of those actions optimized the discounted
future rewards), that reward would appear in the $Q(s, a)$ discounted by a factor of
$\gamma^{2}$. 

If we had perfect knowledge of the Q function, we would be able to choose
an optimal action in any given state. In practice, the space of state-action combinations
we'd have to explore is often impracticably large. For example, a very simple game like
tic tac toe has $3 \times 9$ possible board states and $9$ possible actions in any given state,
for a total of $243$ different pairs of $(s,a)$ we would have to explore. More complicated
games like chess have about $10^3$ possibilities for a pair of moves (one move by White
and one move by Black) alone, and [probably upwards of $10^{40}$ possible legal board configurations](https://math.stackexchange.com/questions/1406919/how-many-legal-states-of-chess-exists). For games with a continuous space of
actions, a brute-force search is not only computationally but theoretically intractable.

The minor miracle that powers deep Q learning is that, at least under certain conditions,
deep neural networks can learn to approximate the Q function!

## Getting up and running with RL

There's been a lot of activity recently focused on making techniques like DQNs
and DDPGs more accessible. Two particular things we'll focus on in this post
are [OpenAI's gym](https://gym.openai.com/), which defines a standard interface
for training environments (and includes many example games), and [Keras-RL](https://github.com/keras-rl/keras-rl),
which allows developers to implement a variety of reinforcement learning
algorithms using Keras's high-level model API. Both libraries can be installed
with `pip` and are reasonably well-documented.

## Defining our environment

One interesting aspect of how agents like AlphaZero were trained is that they are
essentially "self-taught": the environment only describes what moves are and aren't
legal, and what constitutes a winning game; AlphaZero did not learn to play Go
by studying previous games. We'll take a similar approach here. 

The various environments packaged with the OpenAI gym are generally "single player" games,
so we'll have to be a little clever to define a multiplayer game using the same framework.
Each OpenAI gym has to define four methods, as well as document some characteristics
of the observation and action space and reward range. Here's a stub of this:

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

We'll represent our tic tac toe board to the agent as a flattened $3 \times 3 \times 3$ tensor
of binary indicators. The first two dimensions are the board position, and the final dimension
indicates whether a space is unoccupied, occupied by player 1, or occupied by player 2. (Note
that in principle we would only need two binary variables to determine whether a space is
unoccupied, occupied by player 1, or occupied by player 2, but by one-hot encoding each of these
possibilities we are saving the agent from needing to spend time learning this more complicated
representation). Internally, we can track the board state as a $3 \times 3$ dimensional
array of integers for convenience.

The agent's action space consists of the integers 0, 1, .., 8, indicating the position of the
board the agent wants to move to. These correspond to the board positions in the obvious way,
i.e.

```python
[ 0, 1, 2,
  3, 4, 5,
  6, 7, 8 ]
```

The `seed` function is irrelevant for our purposes since tic tac toe is entirely deterministic.

The function `reset` resets the game to the initial state and returns the initial observation
of the environment. In our case, this will need to zero out our board and return the appropriate
empty board:

```python
    def reset(self):
        self.current_player = 0
        self.board = np.zeros(9, dtype='int')
        return self._one_hot_board()
```

Here, `_one_hot_board` is a function that transforms the environment's internal representation of
the board state (an array of integers) into the tensor expected by the agent.

The `step` function is probably the most important of these four: it takes in an action,
and returns the new state and reward. Its implementation is also the most complicated:

```python
    def step(self, action):
        exp = {"state": "in progress"}
        
        reward = 0
        done = False

        # handle illegal moves
        if self.board[action] != 0:
            reward = -10 # illegal moves are really bad
            exp = {"state": "done", 
                   "reason":"Illegal move"}
            done = True
            return self._one_hot_board(), reward, done, exp
        
        self.board[action] = self.current_player + 1
        
        # check if the other player can win on the next turn:
        for streak in self.winning_streaks:
            if ((self.board[streak] == 2 - self.current_player).sum() >= 2) \
                 and (self.board[streak] == 0).any():
                reward = -1
                exp = {
                "state": "in progress", 
                "reason": "{} can lose next turn".format(self.current_player)
                    }
                
        # check if we won
        for streak in self.winning_streaks:
            if (self.board[streak] == self.current_player + 1).all():
                reward = 1 # player wins!
                exp = {"state": "in progress", 
                       "reason": "{} has won".format(self.current_player)}
                done = True
        # check if we tied, which is also a win
        if (self.board != 0).all():
            reward = 1
            exp = {"state": "in progress", 
                   "reason": "{} has tied".format(self.current_player)}
            done = True
        
        # move to the next player
        self.current_player = 1 - self.current_player
        
        return self._one_hot_board(), reward, done, exp
```

The `step` function returns a tuple with the following information: 

* The new observed state (i.e. the one-hot-encoded board)
* The reward resulting from the agent's action
* A boolean indicating whether or not the game is over
* Any additional debug information (used in our case to indicate what happened that turn)

In addition, we also update the environment's internal state with each step.
In particular, each step represents one player's move, and we alternate with each step
which player is moving. (We'll have to write a special kind of "interleaved agent" to
do this, see below.)

Picking reward functions is one of the most difficult things to get right
when doing reinforcement learning. The reward function I settled on is the following:

* If the agent wins the game, they get a reward of `1`, and the game ends.
* If the agent makes an illegal move, they get a reward of `-10`, and the game ends.
* If the agent makes a move that allows the opposing player to win on the next turn,
  they get a reward of `-1`.

The last item (penalizing losing moves) required some trial and error to discover:
initially it seemed cleaner to me not to include such a penalty. However, I noticed
that this resulted in the two players racing to get three in a row without any
concern for the other player's moves. This is because without a penalty for making
a "losing" move, the agent does not learn to pay attention to how close the other
player is to winning.

It's possible that including a reward for "staying alive" might be another way to
incentivize avoiding losing moves, and it might be an interesting experiment to see
how two agents with alternative reward structures play against one another!

## Defining our agent

Recall that we're trying to train our agent to learn to play tic tac toe by
playing against itself. This is a kind of "multi-agent" (i.e. multiplayer) game.
Keras-RL doesn't include any multi-agent functionality out of the box. However,
there are some examples in the wild implementing this---I ended up shamelessly
copying [an interleaved agent implementation from here](https://github.com/velochy/rl-bargaining/blob/master/interleaved.py).

Adding a new type of agent to Keras-RL is actually not especially difficult,
and it's a testament to the library that I (or rather, GitHub user velochy) was
able to bend it to this purpose. The main work involved is just defining a
new class inheriting from the base Keras-RL `Agent` class, and overriding
some methods (specifically the `forward` and `backward` passes). I'll direct the
interested reader to velochy's implementation, and only show specifically the
`forward` and `backward` passes here:

```python
    def forward(self, observation):
        # forward updates the current agent first
        self.current_agent = (self.current_agent + 1) % self.n
        
        return self.agents[self.current_agent].forward(observation)
    
    def backward(self, reward, terminal):
        return (self.agents[self.current_agent]
                .backward(reward, terminal)[:len(self.m_names)])
```

What's going on here is that we're rotating through each individual agent in the
`MultiAgent` with each call to `forward`. Since our environment rotates through
players as well, this means that each time we train on a step, we are training the
appropriate agent. The fit method is actually implemented in the base Keras-RL agent
([see here](https://github.com/keras-rl/keras-rl/blob/master/rl/core.py#L169)), and
for each step `forward` gets called, then `backward`, for the same agent. 

## Building the model

For the rest of it, we can build our model just like one normally would with Keras-RL:

```python
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Embedding
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

def make_dqn():
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Activation('relu'))
    model.add(Dense(27))
    model.add(Activation('relu'))
    model.add(Dense(27))
    model.add(Activation('relu'))
    model.add(Dense(27))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    
    memory = SequentialMemory(limit=50000, window_length=1)
    policy = EpsGreedyQPolicy(eps=0.2)
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, 
                   nb_steps_warmup=100,
                   target_model_update=1e-2, 
                   policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    
    return dqn

dqn = make_dqn()

agent = MultiAgent([dqn, dqn])
```

Here, I've wrapped the model definition in a simple function to make it easy to compare
training the same model compared to training different models for player 1 and player 2.
(For this post, I've focused on the former case.)

Next, let's compile and train our agent:

```python
agent.compile(Adam(lr=1e-3), metrics=['mae'])

agent.fit(env, nb_steps=100000, visualize=False, verbose=1)
```

After about 13 minutes on my machine, training completes. The loss, error, and
reward values reported are not especially informative given that we're training
our DQN to play itself.

## Testing it out!

Now, let's see how our agent does in a "real" game. First, the set up:

```python
test_env = TicTacToe()

done = False
observation = test_env.reset()

agent.training = False
```

It's very important to set `agent.training = False`: in our agent definition,
we used an epsilon-greedy policy with $\epsilon = 0.2$, meaning that during
training the agent will take a random action with probability 0.2. Setting
the `training` parameter to false will disable this random action, which is
generally a good thing. In certain circumstances note that it can be desirable
to have _some_ amount of randomness in your agent's actions (e.g. to avoid
getting "stuck" in some games), but tic tac toe is definitely not one of those
cases.

Next let's play the game:

```python
step = 0
while not done:
    print("Turn: {}".format(step))
    action = agent.forward(observation)
    observation, reward, done, exp = test_env.step(action)
    test_env.render()
    print("\n")
    step += 1

print("A strange game. The only winning move is not to play.")
```
On my machine, this resulted in:
```
Turn: 0
0|0|0
-----
0|0|0
-----
0|1|0


Turn: 1
0|0|0
-----
0|0|0
-----
0|1|2


Turn: 2
0|0|1
-----
0|0|0
-----
0|1|2


Turn: 3
0|0|1
-----
0|0|2
-----
0|1|2


Turn: 4
0|0|1
-----
1|0|2
-----
0|1|2


Turn: 5
0|0|1
-----
1|0|2
-----
2|1|2


Turn: 6
0|0|1
-----
1|1|2
-----
2|1|2


Turn: 7
0|2|1
-----
1|1|2
-----
2|1|2


Turn: 8
1|2|1
-----
1|1|2
-----
2|1|2
```

Note that our game ends in a tie (although we play until the board is full).
Also note that a very interesting play happened with player 2's move in turn 5:

```
0|0|1    0|0|1
-----    -----
1|0|2 -> 1|0|2
-----    -----
0|1|2    2|1|2
```

Player 2 chooses a move that successfully blocks player 1 from completing a winning
streak on either the diagonal or left-hand vertical axis, even though this move
will not directly lead to any 3-in-a-row opportunities for player 2. This is
an interesting move, but likely non-optimal: playing in the upper-left corner instead
would have similarly blocked player 1, but also left player 2 open to possibly
get 3 in a row along the diagonal. Oh well!