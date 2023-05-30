all: install clean
install:
	/usr/bin/python3 -m pip install .
	install -m755 ${CURDIR}/bin/smon /usr/local/bin
clean:
	rm -rf build/ src/smon.egg-info/
