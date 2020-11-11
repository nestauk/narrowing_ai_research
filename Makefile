.PHONY: git all clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = narrowing_ai_research
PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

# Define utility variable to help calling Python from the virtual environment
ifeq ($(CONDA_DEFAULT_ENV),$(PROJECT_NAME))
    ACTIVATE_ENV := true
else
		CONDA_BASE := $(shell conda info --base)
    ACTIVATE_ENV := source $(CONDA_BASE)/etc/profile.d/conda.sh && conda activate $(PROJECT_NAME) || source activate $(PROJECT_NAME)
endif

# Execute python related functionalities from within the project's environment
define execute_in_env
	$(ACTIVATE_ENV) && $1
endef

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Setup git filter for notebooks
git:
	echo -e "[filter \"nbstrip_full\"]\n clean = \"jq --indent 1 '(.cells[] | select(has(\\\"outputs\\\")) | .outputs) = []  | (.cells[] | select(has(\\\"execution_count\\\")) | .execution_count) = null  | .metadata = {\\\"language_info\\\": {\\\"name\\\": \\\"python\\\", \\\"pygments_lexer\\\": \\\"ipython3\\\"}} | .cells[].metadata = {} '\"\n smudge = cat\n required = true\n " >> .git/config


## Make new environment, install requirements and make datasets
all:
	make create_environment
	$(call execute_in_env, make requirements)
	$(call execute_in_env, make data)

## Install Python Dependencies
requirements: test_environment
	$(call execute_in_env, conda env update -f conda_environment.yaml)
	$(call execute_in_env, pip install -e .)

## Fetch data and sync raw to s3
fetch:
	$(PYTHON_INTERPRETER) narrowing_ai_research/fetch_data.py
	#make sync_data_to_s3

## Sync raw from s3 and make Dataset
data: 
	#sync_data_from_s3
	$(PYTHON_INTERPRETER) narrowing_ai_research/make_dataset.py


## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Format using black and lint w/ flake8
lint:
	black narrowing_ai_research; flake8 narrowing_ai_research

## Upload Data to S3
sync_data_to_s3:
ifeq (default,$(PROFILE))
	aws s3 sync data/raw s3://$(BUCKET)/data/raw
else
	aws s3 sync data/raw s3://$(BUCKET)/data/raw --profile $(PROFILE)
endif

## Download Data from S3
sync_data_from_s3:
ifeq (default,$(PROFILE))
	aws s3 sync s3://$(BUCKET)/data/raw data/raw
else
	aws s3 sync s3://$(BUCKET)/data/raw data/raw --profile $(PROFILE)
endif

## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
	conda env create -f conda_environment.yaml
	$(call execute_in_env, pip install -e .)
	$(call execute_in_env, jupyter contrib nbextension install --user)
	$(call execute_in_env, jupyter-nbextensions_configurator enable --user)
	@echo ">>> New conda env created. Activate with:\nconda activate $(PROJECT_NAME)"
else
	@echo ">>> Please install a conda distribution\n"
endif

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
