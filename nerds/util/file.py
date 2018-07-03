import os


def mkdir(directory):
    """ Makes a directory after checking whether it already exists.

        Parameters:
            `directory` (str): The name of the directory to be created.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def rmdir(directory):
    """ Removes an empty directory after checking whether it already exists.

        Parameters:
            `directory` (str): The name of the directory to be removed.
    """
    if os.path.exists(directory) and len(os.listdir(directory)) == 0:
        os.rmdir(directory)
