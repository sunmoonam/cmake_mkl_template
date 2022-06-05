# CMakeList.txt : cmake_mkl_template の CMake プロジェクト。ソースを含めて、次を定義します:
# プロジェクト専用ロジックはこちらです。
#
cmake_minimum_required (VERSION 3.8)

project ("cmake_mkl_template")

# ソースをこのプロジェクトの実行可能ファイルに追加します。
add_executable (cmake_mkl_template "cmake_mkl_template.cpp" "cmake_mkl_template.h")

# TODO: テストを追加し、必要な場合は、ターゲットをインストールします。
#set(CMAKE_GENERATOR_TOOLSET "icx" CACHE STRING "Platform Toolset" FORCE)

# mklの静的リンク／動的リンク
#set(MKL_LINK static)   
#set(MKL_LINK dynamic)  #Windowsではdynamicのdgemm速度が遅い／WSL2(Ubuntu)ではdynamicしかコンパイルが通らない
set(MKL_LINK sdl)       #WIndowsでもWSL2(Ubuntu)でも動作する

# mklの逐次動作／並列動作
#set(MKL_THREADING sequential)
set(MKL_THREADING intel_thread) #tbb_theadよりも早い
#set(MKL_THREADING tbb_thread)  #intel_theadよりも遅い

# mklのポインタ種類
#set(MKL_INTERFACE ilp64)	# cblas_dgemm()は、ilp64では動作しない
set(MKL_INTERFACE lp64)		

# mklパッケージの探索
find_package(MKL CONFIG REQUIRED)
if(MKL_FOUND)
    message(STATUS "MyMesg:Found MKL package")
endif()

# For OpenMP（コンパイラに標準装備されているので、コンパイルオプションに、-openmp を加えるだけで良い？）
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    message(STATUS "MyMesg:Found OpenMP package")
endif()

message(STATUS "MyMesg:==============================")

if(UNIX)
    message(STATUS "MyMesg:OS = WSL2(Ubuntu)")
    message(STATUS "MyMesg: Default cxx flag =  ${CMAKE_CXX_FLAGS} , Default openmp_cxx_flag = ${OpenMP_CXX_FLAGS}")

    if (CMAKE_BUILD_TYPE STREQUAL "Release")
        message(STATUS "MyMesg:  Relsease compile")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp-simd -O3 -Wall -march=native -mavx2")
#        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -O3 -Wall -march=native -mavx2")
    else()
        message(STATUS "MyMesg:  Debug compile")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -O0 -Wall -march=native -mavx2")
    endif()

    # ライブラリのリンク
    # WSL2(Ubuntu)でのchrono（タイマー）#には、-lstdc++ が必要
    target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE ${MKL_IMPORTED_TARGETS} -lstdc++)
else()
    message(STATUS "MyMesg:OS = Windows")
    message(STATUS "MyMesg: Default cxx flag =  ${CMAKE_CXX_FLAGS} , Default openmp_cxx_flag = ${OpenMP_CXX_FLAGS}")

    if (CMAKE_CXX_COMPILER STREQUAL "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/Llvm/x64/bin/clang++.exe")
        message(STATUS "MyMesg: Compiler = clang")
        if (CMAKE_BUILD_TYPE STREQUAL "Release")
            message(STATUS "MyMesg: Compile mode = Relsease")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        else()
            message(STATUS "MyMesg: Compile mode = Debug")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        endif()
    else()
        message(STATUS "MyMesg: Compiler = MSVC")
        if (CMAKE_BUILD_TYPE STREQUAL "Release")
            message(STATUS "MyMesg: Compile mode = Relsease")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} /O2 /Ot /Wall /arch:AVX2 /GL /Qpar")
        else()
            message(STATUS "MyMesg: Compile mode = Debug")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} /Od /Wall")
        endif()
    endif()

    # ライブラリのリンク
    target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE ${MKL_IMPORTED_TARGETS})
endif()

# C++の仕様
enable_language(CXX)
set(CMAKE_CXX_STANDARD 17) # C++17...
set(CMAKE_CXX_STANDARD_REQUIRED ON) #...is required...
set(CMAKE_CXX_EXTENSIONS OFF) #...without compiler extensions like gnu++11

# CMAKE変数のデバッグ
message(STATUS "MyMesg:==============================")
message(STATUS "MyMesg:Project Name = ${CMAKE_PROJECT_NAME}")   # VisualStudioのプロジェクト名
message(STATUS "MyMesg:CPU Arch = ${MKL_ARCH}")             # mklのアーキテクチャ（ia32/intel64）
message(STATUS "MyMesg:MKL Root = ${MKL_ROOT}")             # mklのルートディレクトリ
message(STATUS "MyMesg:MKL Include = ${MKL_INCLUDE}")          # mklヘッダーのディレクトリ名
message(STATUS "MyMesg:MKL Library = ${MKL_IMPORTED_TARGETS}") # リンクするmklライブラリ（複数）
message(STATUS "MyMesg:Compiler = ${CMAKE_CXX_COMPILER}")   # コンパイラ）
message(STATUS "MyMesg:Compiler Option = ${CMAKE_CXX_FLAGS}")      # コンパイルオプション
message(STATUS "MyMesg:Build type = ${CMAKE_BUILD_TYPE}")     # コンパイルオプション
message(STATUS "MyMesg:==============================")
