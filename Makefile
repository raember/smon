all: install clean
install:
	pip install .
install_e:
	pip install -e .
clean:
	rm -rf build/ src/smon.egg-info/
