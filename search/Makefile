all: clean compile_search

compile_search:
	g++ -O3 -fopenmp -std=c++11 search.cpp -o search

runmegaface:
	./search ../face/megaface_sparse_1024.hdf5.bin 1000 0.25

runfacescrub:
	./search ../face/megaface_sparse_1024.hdf5.bin 1000 0.25 ../face/facescrub_sparse_1024.hdf5.bin

clean:
	rm -f ./search
