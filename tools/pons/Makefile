CXX=g++
CXXFLAGS=--std=c++11 -W -Wall -O3 -DNDEBUG

SRCS=Solver.cpp
OBJS=$(subst .cpp,.o,$(SRCS))

c4solver:$(OBJS) main.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o c4solver main.o $(OBJS) $(LDLIBS)

c4solver_big: main.cpp Solver.cpp
	$(CXX) $(CXXFLAGS) -DC4_TABLE_SIZE=28 -o c4solver_big main.cpp Solver.cpp

generator: generator.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o generator generator.o $(LDLIBS)

.depend: $(SRCS)
	$(CXX) $(CXXFLAGS) -MM $^ > ./.depend
	
-include .depend

clean:
	rm -f *.o .depend c4solver generator


