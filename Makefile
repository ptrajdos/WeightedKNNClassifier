ROOTDIR=$(realpath $(dir $(firstword $(MAKEFILE_LIST))))


SRCDIR=${ROOTDIR}/weightedknn
TESTDIR=${ROOTDIR}/tests
COVDIR=${ROOTDIR}/htmlcov_p
COVERAGERC=${ROOTDIR}/.coveragerc
REQ_FILE=${ROOTDIR}/requirements_dev.txt
INSTALL_LOG_FILE=${ROOTDIR}/install.log
VENV_SUBDIR=${ROOTDIR}/venv
COVERAGERC=${ROOTDIR}/.coveragerc
DOCS_DIR=${ROOTDIR}/docs

COVERAGE = coverage
UNITTEST_PARALLEL = unittest-parallel
PDOC= pdoc3
PYTHON=python
PIP=pip
PYTEST=pytest
TOX=tox
TOX_CORES= auto

LOGDIR=${ROOTDIR}/testlogs
LOGFILE=${LOGDIR}/`date +'%y-%m-%d_%H-%M-%S'`.log


PYTHON_VERSION=3.9

ifeq ($(OS),Windows_NT)
	ACTIVATE:=. ${VENV_SUBDIR}/Scripts/activate
else
	ACTIVATE:=. ${VENV_SUBDIR}/bin/activate
endif

.PHONY: all clean test docs tox tox_parallel tox_list

clean:
	rm -rf ${VENV_SUBDIR} ${ROOTDIR}/.tox

venv:
	${PYTHON} -m venv ${VENV_SUBDIR}
	${ACTIVATE}; ${PIP} install pip -U
	${ACTIVATE}; ${PIP} install -e ${ROOTDIR}[test] --prefer-binary --log ${INSTALL_LOG_FILE}

test: venv
	mkdir -p ${LOGDIR}  
	${ACTIVATE}; ${COVERAGE} run --branch  --source=${SRCDIR} -m unittest discover -p '*_test.py' -v -s ${TESTDIR} 2>&1 |tee -a ${LOGFILE}
	${ACTIVATE}; ${COVERAGE} html --show-contexts


test_parallel: venv
	mkdir -p ${COVDIR} ${LOGDIR}
	${ACTIVATE}; ${UNITTEST_PARALLEL} --class-fixtures -v -t ${ROOTDIR} -s ${TESTDIR} -p '*_test.py' --coverage --coverage-rcfile ./.coveragerc --coverage-source ${SRCDIR} --coverage-html ${COVDIR}  2>&1 |tee -a ${LOGFILE}

docs: venv
	${ACTIVATE}; $(PDOC) --force --html ${SRCDIR} --output-dir ${DOCS_DIR}

profile: venv
	
	${ACTIVATE}; ${PYTEST} -n auto --cov-report=html --cov=${SRCDIR} --profile ${TESTDIR}

# Run the Python x scikit-learn matrix defined in tox.ini.
# Select a subset with e.g.: make tox TOX_ENVS=py311-sklearn19,py39-sklearn122
TOX_ENVS=
TOX_ENV_ARG=$(if ${TOX_ENVS},-e ${TOX_ENVS},)

tox: venv
	mkdir -p ${LOGDIR}
	${ACTIVATE}; ${TOX} -c ${ROOTDIR}/tox.ini ${TOX_ENV_ARG} 2>&1 |tee -a ${LOGFILE}

tox_parallel: venv
	mkdir -p ${LOGDIR}
	${ACTIVATE}; ${TOX} -c ${ROOTDIR}/tox.ini ${TOX_ENV_ARG} -p $(TOX_CORES) 2>&1 |tee -a ${LOGFILE}

tox_list: venv
	${ACTIVATE}; ${TOX} -c ${ROOTDIR}/tox.ini list
