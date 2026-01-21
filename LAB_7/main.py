"""
Sztuczna inteligencja wykorzystująca uczenie ze wzmocnieniem (Reinforcement Learning),
która uczy się grać w grę Flappy Bird przy użyciu algorytmu DQN.

Autorzy:
- Dominik Ludwiński
- Bartosz Dembowski

Środowisko FlappyBird wymaga Python 3.9

Wymagane biblioteki:
    pip install flappy_bird_gym
    pip install tensorflow==2.5.0
    pip install keras-rl2
"""

import flappy_bird_gym
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy


def build_model(obs, actions):
    """
    Funkcja budująca model sieci neuronowej.
    Model odpowiada za aproksymację funkcji Q,
    na podstawie której agent podejmuje decyzje w grze.
    """
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(1, obs)))
    model.add(Dense(128, activation='relu'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    model.summary()
    return model


# Utworzenie środowiska gry Flappy Bird
env = flappy_bird_gym.make("FlappyBird-v0")

# Pobranie liczby obserwacji oraz możliwych akcji z environmentu
obs = env.observation_space.shape[0]
actions = env.action_space.n

# Zbudowanie modelu sieci neuronowej
model = build_model(obs, actions)


def build_agent(model, actions):
    """
    Funkcja budująca agenta typu DQN (Deep Q-Network),
    który uczy się strategii gry na podstawie interakcji ze środowiskiem.
    """
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr='eps',
        value_max=0.5,
        value_min=0.0001,
        value_test=.0,
        nb_steps=6000000
    )

    # Pamięć doświadczeń (replay buffer)
    memory = SequentialMemory(limit=100000, window_length=1)

    # Konfiguracja agenta DQN
    dqn = DQNAgent(
        model=model,
        memory=memory,
        policy=policy,
        enable_dueling_network=True,
        dueling_type='avg',
        nb_actions=actions,
        nb_steps_warmup=500
    )

    return dqn


# Utworzenie agenta
dqn = build_agent(model, actions)

# Kompilacja modelu z użyciem optymalizatora Adam
dqn.compile(Adam(learning_rate=0.003))

"""
Po kompilacji sieć neuronowa jest gotowa do uczenia.
Poniższa linia (zakomentowana) odpowiada za trening agenta.
"""

#dqn.fit(env, nb_steps=5000000, visualize=False, verbose=1)

"""
Zapis wytrenowanych wag modelu do pliku.
"""

#dqn.save_weights("flappy.h5")

"""
Wczytanie zapisanych wag i uruchomienie testu agenta
z wizualizacją jednego epizodu gry.
"""

dqn.load_weights("testing.h5")
dqn.test(env, visualize=True, nb_episodes=10)
