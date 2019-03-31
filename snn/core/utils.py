import os
import pickle


def deserialize(path):
    """
        A method to save entities.
    :param path: Where you want to save it
    """
    with open(path, 'rb') as f:
        entity = pickle.load(f)
    return entity


def serialize(entity, path, overwrite=False):
    """
        A method to save entities.
    :param entity: What you would like to save
    :param path: Where you want to save it
    :param overwrite: Overwrite previously saved entity?
    """
    if os.path.exists(path) and not overwrite:
        raise FileExistsError("File in {} already exists, use --overwrite in your script or run the experiment in a "
                              "different setting".format(path))
    else:
        save_dir = os.path.join(*os.path.split(path)[:-1])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(path, 'wb') as f:
            pickle.dump(entity, f)
