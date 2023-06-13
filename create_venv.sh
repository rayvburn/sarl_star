#!/usr/bin/env bash
#
# Creates a virtual environment that handles Python module dependencies for all packages necessary to run SARL*
#
SCRIPT_DIR=$(realpath $(dirname $0))

cd $SCRIPT_DIR
virtualenv .venv
source .venv/bin/activate

# Make sure that installing with python into the venv: https://stackoverflow.com/a/5979776
VENV_PYTHON=$SCRIPT_DIR/.venv/bin/python
VENV_PIP=$SCRIPT_DIR/.venv/bin/pip

# installing package dependencies
$VENV_PIP install torch
$VENV_PIP install torchvision
$VENV_PIP install scipy
$VENV_PIP install gym
$VENV_PIP install configparser
$VENV_PIP install matplotlib
$VENV_PIP install gitpython

# prepare RVO
cd $SCRIPT_DIR/Python-RVO2/
$VENV_PIP install -r requirements.txt
# creates .so library right in the main directory of the module
$VENV_PYTHON setup.py build_ext --inplace

# prepare CrowdNav
cd $SCRIPT_DIR/sarl_star_ros/CrowdNav/
$VENV_PYTHON setup.py install

deactivate
