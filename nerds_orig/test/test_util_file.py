import os

from nose.tools import assert_true, assert_false

from nerds.util.file import mkdir, rmdir


def test_mkdir():
    directory1 = "data"
    directory2 = "data/foo"

    mkdir(directory1)
    mkdir(directory2)

    assert_true(os.path.exists(directory1))
    assert_true(os.path.exists(directory2))


def test_rmdir():
    directory1 = "data"
    directory2 = "data/foo"

    mkdir(directory1)
    mkdir(directory2)

    rmdir(directory2)
    rmdir(directory1)

    assert_false(os.path.exists(directory1))
    assert_false(os.path.exists(directory2))
