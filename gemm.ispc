task void gemm_ispc_block(uniform int m, uniform int n, uniform int k,
	uniform double A[], uniform double B[], uniform double C[], uniform double alpha,
	 uniform double beta, uniform int startBlockRow, uniform int endBlockRow,
	 uniform int startBlockColumn, uniform int endBlockColumn) {
	// YOUR IMPLEMENTATION HERE

	uniform int startRowIndex = startBlockRow + (uniform int)((endBlockRow - startBlockRow)/4 * (taskIndex/4));
	uniform int endRowIndex = min(startRowIndex + (uniform int)((endBlockRow - startBlockRow)/4), endBlockRow);
	uniform int startColumnIndex = startBlockColumn + (uniform int)((endBlockColumn - startBlockColumn)/4 * (taskIndex%4));
	uniform int endColumnIndex = min(startColumnIndex + (uniform int)((endBlockColumn - startBlockColumn)/4), endBlockColumn);


	foreach (i = startRowIndex ... endRowIndex){ 
		for(uniform int j=startColumnIndex; j<endColumnIndex; j++){
			double inner_prod = 0;
			for(uniform int kk=0; kk<k; kk++){
				inner_prod += A[i*k+kk] * B[j*k+kk];
			}
			C[i*n+j] = alpha * inner_prod + beta * C[i*n+j];
		}
	}
}

export void gemm_ispc(uniform int m, uniform int n, uniform int k,
	 uniform double A[], uniform double B[], uniform double C[], uniform double alpha, 
	 uniform double beta) {
	// YOUR IMPLEMENTATION HERE
	uniform int lenBlock = 256;
	if(m>=2048){
		lenBlock = 512;
	}
	uniform int numBlock = m / lenBlock;

	for(uniform int i = 0; i < numBlock; i++){
		uniform int startBlockRow = i * lenBlock;
		uniform int endBlockRow = min(startBlockRow + lenBlock, m);

		foreach (ai = startBlockRow ... endBlockRow){
			for(uniform int kk=0; kk<k; kk++){
				prefetch_l2(&A[ai*k+kk]);
			}
		}

		for(uniform int j = 0; j < numBlock; j++){
			uniform int startBlockColumn = j * lenBlock;
			uniform int endBlockColumn = min(startBlockColumn + lenBlock, m);

			foreach (bj = startBlockColumn ... endBlockColumn){
				for(uniform int kk=0; kk<k; kk++){
					prefetch_l2(&B[bj*k+kk]);
				}
			}

			foreach (ci = startBlockRow ... endBlockRow){ 
				for(int cj=startBlockColumn; cj<endBlockColumn; cj++){
					prefetch_l2(&C[ci*n+cj]);
				}
			}

			launch[16] gemm_ispc_block(m, n, k, A, B, C, alpha, beta, startBlockRow, endBlockRow, startBlockColumn, endBlockColumn);
		}
	}
}