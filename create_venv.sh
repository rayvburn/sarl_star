#!/usr/bin/env bash
#
# Creates a virtual environment that handles Python module dependencies for all packages necessary to run SARL*
#
SCRIPT_DIR=$(realpath $(dirname $0))

cd $SCRIPT_DIR
virtualenv --python="python2.7" .venv
source .venv/bin/activate

# Make sure that installing with python into the venv: https://stackoverflow.com/a/5979776
# Valid setup consists of Python 2.7.17 and pip 20.3.4
VENV_PYTHON=$SCRIPT_DIR/.venv/bin/python2.7
VENV_PIP=$SCRIPT_DIR/.venv/bin/pip2.7

# installing package dependencies
$VENV_PIP install torch==1.4.0
$VENV_PIP install torchvision==0.5.0
$VENV_PIP install scipy==1.2.3
$VENV_PIP install gym==0.16.0
$VENV_PIP install configparser==4.0.2
$VENV_PIP install matplotlib==2.2.5
$VENV_PIP install gitpython==2.1.15

# prepare RVO
cd $SCRIPT_DIR/Python-RVO2/
$VENV_PIP install -r requirements.txt
# creates .so library right in the main directory of the module
$VENV_PYTHON setup.py build_ext --inplace

# prepare CrowdNav
cd $SCRIPT_DIR/sarl_star_ros/CrowdNav/
$VENV_PYTHON setup.py install

deactivate
