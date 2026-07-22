.PHONY: setup pipeline dashboard clean test

PYTHON ?= python3

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

# Part 1 loads the database; Parts 2-4 read it back out and write outputs/.
pipeline:
	$(PYTHON) load_data.py
	$(PYTHON) run_analysis.py

dashboard:
	$(PYTHON) -m streamlit run dashboard/app.py

test:
	$(PYTHON) -m pytest tests/ -v

clean:
	rm -f cell_count.db
	rm -rf outputs/
