import pickle


def save(name, value):
    with open(f'./Saved Data/{name}.pkl', 'wb') as file:
        pickle.dump(value, file)


def load(name):
    with open(f'./Saved Data/{name}.pkl', 'rb') as file:
        return pickle.load(file)