all:
	javac -classpath opencv-412.jar *.java

test:
	$(MAKE) all
	java -classpath .:opencv-412.jar -Djava.library.path=. Test
