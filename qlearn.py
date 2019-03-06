import snake
import argparse
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D
from keras import optimizers
import numpy as np
import skimage as skimage
import random
import json
from random import sample as rsample
import time


GAME_INPUT = [0, 1, 2, 3, 4]
RANDTRAINING = 1
RANDTRAINING_DECAY = 0.99
FINAL_RANDTRAINING = 0.3
INPUT_SHAPE = (80, 80, 2)
NB_ACTIONS = 5
BATCH = 50
GAMMA = 0.6



def build_model():
    model = Sequential()
    model.add(Convolution2D(16, (8, 8), strides=(4, 4), input_shape=INPUT_SHAPE))
    model.add(Activation("relu"))
    model.add(Convolution2D(32, (4, 4), strides=(2, 2)))
    model.add(Activation("relu"))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(Dense(NB_ACTIONS))
    sgd = optimizers.SGD(lr=1e-4, clipnorm=1)
    model.compile(loss="mean_squared_error", optimizer=sgd)
    return model


def replay(batch_size):
    memory = []
    while True:
        experience = (
            yield rsample(memory, batch_size) if batch_size <= len(memory) else None
        )
        memory.append(experience)


def image_preparation(game_image):
    x_t = skimage.transform.resize(skimage.color.rgb2gray(game_image), (80, 80))
    # put it in black and white and boost contrast
    x_t = skimage.exposure.rescale_intensity(x_t, out_range=(0, 255))
    # 2 images stacked
    s_t = np.stack((x_t, x_t), axis=2)
    # Reshape for keras
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])
    return s_t


def load_weights(model):
    print("loading weight")
    model.load_weights("weights")
    print("training is starting")
    train_network(model)


def train_network(model):
    epsilon = RANDTRAINING
    game_state = snake.Environment()  # Starting up a game
    game_state.set_initial_state()
    game_image, score, game_lost = game_state.run(
        0
    )  # The game is started but no action is performed
    s_t = image_preparation(game_image)
    terminal = False
    t = 0
    exp_replay = replay(BATCH)
    exp_replay.__next__()  # Start experience replay coroutine
    while True:
        loss = 0
        Q_sa = 0
        r_t = 0
        a_t = "no nothing"
        if terminal:
            game_state.set_initial_state()
        if random.random() <= epsilon:
            action_index = random.randrange(NB_ACTIONS)
            a_t = GAME_INPUT[action_index]
        else:
            action_index = np.argmax(model.predict(s_t))
            a_t = GAME_INPUT[action_index]
        if epsilon > FINAL_RANDTRAINING:
            epsilon = epsilon * RANDTRAINING_DECAY
        else:
            epsilon = FINAL_RANDTRAINING
        # run the selected action and observed next state and reward
        x_t1_colored, r_t, terminal = game_state.run(a_t)
        s_t1 = image_preparation(x_t1_colored)
        experience = (s_t, a_t, r_t, s_t1)
        batch = exp_replay.send(experience)
        s_t1 = image_preparation(x_t1_colored)
        if batch:
            inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))
            targets = np.zeros((BATCH, NB_ACTIONS))
            i = 0
            for s, a, r, s_pred in batch:
                inputs[i : i + 1] = s
                if r < 0:
                    targets[i, a] = r
                else:
                    Q_sa = model.predict(s_pred)
                    targets[i, a] = r + GAMMA * np.max(Q_sa)
                i += 1
            loss += model.train_on_batch(inputs, targets)
            # Exploration vs Exploitation

        t += 1
        # save progress every 10000 iterations
        if t % 1000 == 0:
            print("Now we save model")
            model.save_weights("weights", overwrite=True)

        if t % 500 == 0:
            print(
                "TIMESTEP",
                t,
                "/ RANDTRAINING",
                epsilon,
                "/ ACTION",
                action_index,
                "/ REWARD",
                r_t,
                "/ Q_MAX ",
                np.max(Q_sa),
                "/ Loss ",
                loss,
            )

    print("Episode finished!")
    print("************************")


def nn_playGame(model):
    print("Now we load weight")
    model.load_weights("weights")
    print("Weight load successfully")
    print("Let the game begin!")
    game_state = snake.Environment()  # Starting up a game
    game_state.set_initial_state()
    game_image, score, game_lost = game_state.run(
        4
    )  # The game is started but no action is performed
    s_t = image_preparation(game_image)
    s_t1 = s_t
    a_t = 4
    while True:

        if game_lost:
            print("Game lost")
            time.sleep(2)
            print("Game is restarting")
            game_state.set_initial_state()

        action_index = np.argmax(model.predict(s_t1))
        a_t = GAME_INPUT[action_index]
        x_t1_colored, _, terminal = game_state.run(a_t)
        s_t1 = image_preparation(x_t1_colored)
        game_lost = terminal


def playGame(args):
    model = build_model()
    if args["mode"] == "Run":
        nn_playGame(model)
    elif args["mode"] == "Re-train":
        load_weights(model)
    elif args["mode"] == "Train":
        train_network(model)


def main():
    parser = argparse.ArgumentParser(
        description="How you would like your program to run"
    )
    parser.add_argument("-m", "--mode", help="Train / Run / Re-train", required=True)
    args = vars(parser.parse_args())
    playGame(args)

main()
