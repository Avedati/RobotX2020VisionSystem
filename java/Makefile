all:
	javac -classpath opencv-412.jar *.java

clean:
	rm -f *.class

test:
	$(MAKE) all
	java -classpath .:opencv-412.jar -Djava.library.path=. Test
