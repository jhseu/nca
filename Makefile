CXX = g++
CXXFLAGS = -O2 -Wall -I/usr/include/eigen2
LDFLAGS =
OBJ = nca.o

nca: $(OBJ)
	$(CXX) -o $@ $^ $(CXX_FLAGS) $(LDFLAGS)

clean:
	rm -f $(OBJ) nca
