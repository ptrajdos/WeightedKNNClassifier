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

LOGDIR=${ROOTDIR}/testlogs
LOGFILE=${LOGDIR}/`date +'%y-%m-%d_%H-%M-%S'`.log


PYTHON_VERSION=3.9

ifeq ($(OS),Windows_NT)
	ACTIVATE:=. ${VENV_SUBDIR}/Scripts/activate
else
	ACTIVATE:=. ${VENV_SUBDIR}/bin/activate
endif

.PHONY: all clean test docs

clean:
	rm -rf ${VENV_SUBDIR}

venv:
	${PYTHON} -m venv ${VENV_SUBDIR}
	${ACTIVATE}; ${PIP} install pip -U
	${ACTIVATE}; ${PIP} install -e ${ROOTDIR} --prefer-binary --log ${INSTALL_LOG_FILE} -r ${REQ_FILE}

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