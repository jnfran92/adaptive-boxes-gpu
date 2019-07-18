
#define CC(x) do { if((x) != cudaSuccess) { \
	      printf("Error at %s:%d\n",__FILE__,__LINE__); \
	      return EXIT_FAILURE;}} while(0)

