
.PHONY: cpu mkldnn

cpu:
	g++ -o run_cpu -std=c++11 layer.cc activation.cc loss.cc network.cc app.cc -O3 -fopenmp

debug_cpu:
	g++ -o debug_run_cpu -std=c++11 layer.cc activation.cc loss.cc network.cc app.cc -g -gdwarf-3 -fopenmp

mkldnn:
	g++ -o run_mkldnn -std=c++11 layer_mkldnn.cc activation_mkldnn.cc loss_mkldnn.cc network_mkldnn.cc app.cc -fopenmp
