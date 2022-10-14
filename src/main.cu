#include <iostream>
#include <chrono>
#include <cutf/cublas.hpp>
#include <cutf/curand.hpp>
#include <cutf/memory.hpp>
#include <cutf/type.hpp>
#include <wmma_extension/wmma_extension.hpp>
#include <mateval/comparison_cuda.hpp>

constexpr unsigned test_count = 100;

namespace {
template <class T, class storage_t = typename mtk::wmma::detail::common::storage_t<T>::type>
__global__ void split_kernel(
		storage_t* const M_low,
		storage_t* const dM_low,
		const float* const M,
		const unsigned m, const unsigned n, const unsigned ld
		) {
	const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= m * n) {
		return;
	}

	const auto mi = tid % m;
	const auto ni = tid / m;
	const auto mem_index = mi + ni * ld;

	const auto v = M[mem_index];
	const auto v_low = cutf::type::cast<storage_t>(v);
	const auto dv_low = cutf::type::cast<storage_t>((v - cutf::type::cast<float>(v_low)) * 2048);

	M_low[mem_index] = v_low;
	dM_low[mem_index] = dv_low;
}

template <class T, class storage_t = typename mtk::wmma::detail::common::storage_t<T>::type>
void split(
		storage_t* const M_low,
		storage_t* const dM_low,
		const float* const M,
		const unsigned m, const unsigned n, const unsigned ld
		) {
	constexpr std::size_t block_size = 256;
	const auto grid_size = (m * n + block_size - 1) / block_size;

	split_kernel<T><<<grid_size, block_size>>>(
			M_low, dM_low,
			M,
			m, n, ld
			);
}

template <class T, class storage_t = typename mtk::wmma::detail::common::storage_t<T>::type>
void matmul_tcec(
		cublasHandle_t cublas_handle,
		const unsigned m, const unsigned n, const unsigned k,
		float* const c_ptr, // m x n
		const float* const a_ptr, // m x k
		const float* const b_ptr, // k x n
		storage_t* a_low_ptr,
		storage_t* da_low_ptr,
		storage_t* b_low_ptr,
		storage_t* db_low_ptr
		) {
	split<T>(
			a_low_ptr, da_low_ptr,
			a_ptr,
			m, k, m
			);
	split<T>(
			b_low_ptr, db_low_ptr,
			b_ptr,
			k, n, k
			);

	// Error Correction : dA*B + A*db
	const float one = 1.0f, zero = 0.0f;
	CUTF_CHECK_ERROR(cublasGemmEx(
				cublas_handle,
				CUBLAS_OP_N, CUBLAS_OP_N,
				m, n, k,
				&one,
				a_low_ptr, cutf::type::get_data_type<storage_t>(), m,
				db_low_ptr, cutf::type::get_data_type<storage_t>(), k,
				&zero,
				c_ptr, cutf::type::get_data_type<float>(), m,
				std::is_same<T, half>::value ? CUBLAS_COMPUTE_32F_FAST_16F : CUBLAS_COMPUTE_32F_FAST_TF32,
				CUBLAS_GEMM_DEFAULT_TENSOR_OP
				));
	CUTF_CHECK_ERROR(cublasGemmEx(
				cublas_handle,
				CUBLAS_OP_N, CUBLAS_OP_N,
				m, n, k,
				&one,
				da_low_ptr, cutf::type::get_data_type<storage_t>(), m,
				b_low_ptr, cutf::type::get_data_type<storage_t>(), k,
				&one,
				c_ptr, cutf::type::get_data_type<float>(), m,
				std::is_same<T, half>::value ? CUBLAS_COMPUTE_32F_FAST_16F : CUBLAS_COMPUTE_32F_FAST_TF32,
				CUBLAS_GEMM_DEFAULT_TENSOR_OP
				));

	// Error Correction : A*B
	const float scale = 1 / 2048.f;
	CUTF_CHECK_ERROR(cublasGemmEx(
				cublas_handle,
				CUBLAS_OP_N, CUBLAS_OP_N,
				m, n, k,
				&one,
				a_low_ptr, cutf::type::get_data_type<storage_t>(), m,
				b_low_ptr, cutf::type::get_data_type<storage_t>(), k,
				&scale,
				c_ptr, cutf::type::get_data_type<float>(), m,
				std::is_same<T, half>::value ? CUBLAS_COMPUTE_32F_FAST_16F : CUBLAS_COMPUTE_32F_FAST_TF32,
				CUBLAS_GEMM_DEFAULT_TENSOR_OP
				));
}

template <class T>
void eval(
		const unsigned min_N, const unsigned max_N
		) {
	using storage_t = typename mtk::wmma::detail::common::storage_t<T>::type;
	const auto num_elements = max_N * max_N;

	auto cublas_handle = cutf::cublas::get_cublas_unique_ptr();

	auto mat_A = cutf::memory::get_device_unique_ptr<float>(num_elements);
	auto mat_B = cutf::memory::get_device_unique_ptr<float>(num_elements);
	auto mat_C = cutf::memory::get_device_unique_ptr<float>(num_elements);
	auto mat_A_low = cutf::memory::get_device_unique_ptr<storage_t>(num_elements);
	auto mat_B_low = cutf::memory::get_device_unique_ptr<storage_t>(num_elements);
	auto mat_C_low = cutf::memory::get_device_unique_ptr<storage_t>(num_elements);
	auto mat_dA_low = cutf::memory::get_device_unique_ptr<storage_t>(num_elements);
	auto mat_dB_low = cutf::memory::get_device_unique_ptr<storage_t>(num_elements);
	auto mat_dC_low = cutf::memory::get_device_unique_ptr<storage_t>(num_elements);

	unsigned long long seed = 10;
	auto cugen = cutf::curand::get_curand_unique_ptr(CURAND_RNG_PSEUDO_MT19937);
	CUTF_CHECK_ERROR(curandSetPseudoRandomGeneratorSeed(*cugen.get(), seed));
	CUTF_CHECK_ERROR(cutf::curand::generate_uniform(*cugen.get(), mat_A.get(), num_elements));
	CUTF_CHECK_ERROR(cutf::curand::generate_uniform(*cugen.get(), mat_B.get(), num_elements));
	CUTF_CHECK_ERROR(cutf::curand::generate_uniform(*cugen.get(), mat_C.get(), num_elements));

	std::printf("N,residual,throughput_in_tflops\n");
	for (unsigned N = min_N; N <= max_N; N <<= 1) {
		matmul_tcec<T>(
				*cublas_handle.get(),
				N, N, N,
				mat_C.get(),
				mat_A.get(),
				mat_B.get(),
				mat_A_low.get(),
				mat_B_low.get(),
				mat_dA_low.get(),
				mat_dB_low.get()
				);
		auto error = mtk::mateval::cuda::get_error_AxB(
				mtk::mateval::relative_residual,
				N, N, N,
				mtk::mateval::col_major,
				mtk::mateval::col_major,
				mtk::mateval::col_major,
				mat_A.get(), N,
				mat_B.get(), N,
				mat_C.get(), N
				);
		cudaDeviceSynchronize();
		const auto start_clock = std::chrono::system_clock::now();
		for (unsigned i = 0; i < test_count; i++) {
			matmul_tcec<T>(
					*cublas_handle.get(),
					N, N, N,
					mat_C.get(),
					mat_A.get(),
					mat_B.get(),
					mat_A_low.get(),
					mat_B_low.get(),
					mat_dA_low.get(),
					mat_dB_low.get()
					);
		}
		cudaDeviceSynchronize();
		const auto end_clock = std::chrono::system_clock::now();
		const auto elapsed_tiem = std::chrono::duration_cast<std::chrono::nanoseconds>(end_clock - start_clock).count() * 1e-9;
		const auto complexity = 2lu * N * N * N * test_count;

		std::printf("%u,%e,%e\n",
				N,
				error.at(mtk::mateval::relative_residual),
				complexity / elapsed_tiem * 1e-12
				);
	}
}

} // unnamed namespace

int main() {
	eval<half>(1u << 10, 1u << 14);
}
