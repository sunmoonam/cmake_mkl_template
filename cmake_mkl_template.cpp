// cmake_mkl_template.cpp : アプリケーションのエントリ ポイントを定義します。
// 

#include "cmake_mkl_template.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <random>       // 乱数用
#include <iomanip>      // 書式表示用
#include <chrono>       // タイマー用

#include <omp.h>        // Open MP
#include "mkl.h"

using namespace std;

typedef float   DBLorFLT;               // 演算精度（浮動小数double）

#define NUMBER_32BIT    4294967296      // 乱数の正規化（0.0～1.0）用
#define ALIGN_VAL       64              // メモリアラインメント（32byte = 256bit 単位）

// mkl_mallocによる1次元配列の動的確保
//
// ptrm[i * matrix_size + j]でアクセス
//
DBLorFLT* malloc_1d_array(int matrix_size) {
    DBLorFLT* ptrm;

    ptrm = (DBLorFLT*)mkl_malloc(sizeof(DBLorFLT) * matrix_size * matrix_size, ALIGN_VAL);

    if (ptrm == NULL) {
        printf("ERROR: Can't allocate 1d_array memory [%d Kbyte] \n", matrix_size * matrix_size / 1024);
        exit(-1);
    }

    return(ptrm);
}

// 1次元配列の表示
//    double/float  ptrm                表示する行列
//    int           i0, j0, i1, i1      表示する行列範囲(i0,j0)-(i1,j1)
//                                      i0,i1,j0,j1引数を省略した場合、i0 = 0, j0 = 0, i0 = matrix_size, j0 = matrix_size になる    
//
template<typename T>
void display_1d_array(int matrix_size, T ptrm, int i0 = 0, int j0 = 0, int i1 = 0, int j1 = 0) {
    int i, j;

    if (i0 == 0 && i1 == 0)	i1 = matrix_size - 1;
    if (j0 == 0 && j1 == 0)	j1 = matrix_size - 1;

    for (i = i0; i < i1 + 1; i++) {
        for (j = 0; j < j1 + 1; j++) {
            printf("%.10f , ", ptrm[i * matrix_size + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(void)
{
	const int	N = 4096;
    int Iteration = 100;

    DBLorFLT* da, * db, * dc;
    DBLorFLT alpha, beta;
    std::random_device rnd;

    std::chrono::system_clock::time_point start, end;   // タイマー計測ライブラリ

    da = malloc_1d_array(N);
    db = malloc_1d_array(N);
    dc = malloc_1d_array(N);

    // mklのスレッド数はOMPで制御する
    printf("Max thread number = %d \n", omp_get_max_threads());

    // 行列の初期化
    std::mt19937 mt(rnd()); // メルセンヌツイスターの乱数を利用（rnd()を利用するとWSL2(Ubuntu)では遅くなる）
#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            da[i * N + j] = DBLorFLT(DBLorFLT(mt()) / NUMBER_32BIT);
            db[i * N + j] = DBLorFLT(DBLorFLT(mt()) / NUMBER_32BIT); // rnd()は、0～2^32の整数を生成 → NUMBER_32BIT(2^32)で除算し、0～1.0に正規化
            dc[i * N + j] = 0.0;
        }
    }

    // allocate memoty
    display_1d_array(N, da, 0, 0, 3, 3);
    display_1d_array(N, db, 0, 0, 3, 3);
    display_1d_array(N, dc, 0, 0, 3, 3);

    printf("=====  Double Matrix =====\n");
    printf("Matrix Size = %3d x %3d  \n", N, N);

    // start timer
    start = std::chrono::system_clock::now();

    alpha = 1.0;
    beta = 0.0;

#pragma omp parallel for
    for (int loop = 0; loop < Iteration; loop++) {
        sgemm("N", "N", &N, &N, &N, &alpha, db, &N, da, &N, &beta, dc, &N);
        //dgemm("N", "N", &N, &N, &N, &alpha, db, &N, da, &N, &beta, dc, &N);
        //cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, alpha, da, N, db, N, beta, dc, N);
        //cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, alpha, da, N, db, N, beta, dc, N);
    }

    // end timer
    end = std::chrono::system_clock::now();
    
    // dsiplay benchmark result
    double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    double sec = elapsed / (Iteration * 1.0e9);
    double flops = pow(N, 3) / sec;
    std::cout << "  " << sec / Iteration << "s" << " , " << flops / 1.0e9 << "Gflops\n" << std::endl;

    // display calculation result
    display_1d_array(N, da, 0, 0, 3, 3);
    display_1d_array(N, db, 0, 0, 3, 3);
    display_1d_array(N, dc, 0, 0, 3, 3);
 
    // release memoty
    mkl_free(da);
    mkl_free(db);
    mkl_free(dc);

    return 0;
}
