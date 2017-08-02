
.PHONY: cpu mkldnn

cpu:
	icpc -o run_cpu layer.cc activation.cc loss.cc network.cc app.cc -O3 -qopenmp

debug_cpu:
	icpc -o debug_run_cpu layer.cc activation.cc loss.cc network.cc app.cc -g -qopenmp

mkldnn:
	icpc -o run_mkldnn layer_mkldnn.cc activation_mkldnn.cc loss_mkldnn.cc network_mkldnn.cc app.cc -qopenmp
