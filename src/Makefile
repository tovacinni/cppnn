OBJECTS = network.o test.o iris_loader.o
DEPENDS= ${OBJECTS:.o=.d}
EXEC = test
EIGEN_PATH = ../lib # change this to proper path

CXX = g++
CXXFLAGS = -std=c++14 -I ${EIGEN_PATH} -Wall -MMD
MAKEFILE_NAME = ${firstword ${MAKEFILE_LIST}}

.PHONY : clean

${EXEC} : ${OBJECTS}
	${CXX} ${CXXFLAGS} ${OBJECTS} -o ${EXEC}

-include ${DEPENDS}

clean :
	rm -f ${DEPENDS} ${OBJECTS} ${EXEC}

.PHONY : clean
