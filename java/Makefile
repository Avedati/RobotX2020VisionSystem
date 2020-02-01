all:
	javac -classpath libs/opencv-412.jar *.java

clean:
	rm -f *.class

test:
	$(MAKE) all
	java -classpath .:libs/opencv-412.jar -Djava.library.path=.:libs Test
