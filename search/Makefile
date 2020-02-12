all: clean compile_search

compile_search:
	g++ -O3 -fopenmp -std=c++11 search.cpp -o search

runmegaface:
	./search ../face/megaface_sparse_1024.hdf5.bin

runfacescrub:
	./search ../face/megaface_sparse_1024.hdf5.bin ../face/facescrub_sparse_1024.hdf5.bin

clean:
	rm -f ./search
