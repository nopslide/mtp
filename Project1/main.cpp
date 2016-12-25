
#include <string.h>
#include <iostream>
#include <vector>
#include <stdint.h>
#include <stdlib.h>
#include <inttypes.h>
#include <emmintrin.h> 
#include <random>
#include "common.h"
#include "uint256.h"
#include "merkle.h"
#define _CRT_SECURE_NO_WARNINGS

/*For memory wiping*/
#ifdef _MSC_VER
#include <windows.h>
#include <winbase.h> /* For SecureZeroMemory */
#endif

#define VC_GE_2005(version) (version >= 1400)
#define BLAKE2_INLINE __inline
#if defined(__clang__)
#if __has_attribute(optnone)
#define NOT_OPTIMIZED __attribute__((optnone))
#endif
#elif defined(__GNUC__)
#define GCC_VERSION                                                            \
    (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#if GCC_VERSION >= 40400
#define NOT_OPTIMIZED __attribute__((optimize("O0")))
#endif
#endif
#ifndef NOT_OPTIMIZED
#define NOT_OPTIMIZED
#endif

/* Memory allocator types --- for external allocation */
typedef int(*allocate_fptr)(uint8_t **memory, size_t bytes_to_allocate);
typedef void(*deallocate_fptr)(uint8_t *memory, size_t bytes_to_allocate);

const __m128i r16 = _mm_setr_epi8(2, 3, 4, 5, 6, 7, 0, 1, 10, 11, 12, 13, 14, 15, 8, 9);
const __m128i r24 = _mm_setr_epi8(3, 4, 5, 6, 7, 0, 1, 2, 11, 12, 13, 14, 15, 8, 9, 10);
__m128i t0, t1;

#define _mm_roti_epi64(x, c) \
	(-(c) == 32) ? _mm_shuffle_epi32((x), _MM_SHUFFLE(2,3,0,1))  \
	: (-(c) == 24) ? _mm_shuffle_epi8((x), r24) \
	: (-(c) == 16) ? _mm_shuffle_epi8((x), r16) \
	: (-(c) == 63) ? _mm_xor_si128(_mm_srli_epi64((x), -(c)), _mm_add_epi64((x), (x)))  \
	: _mm_xor_si128(_mm_srli_epi64((x), -(c)), _mm_slli_epi64((x), 64-(-(c))))

static inline __m128i fBlaMka(__m128i x, __m128i y) {
	__m128i z = _mm_mul_epu32(x, y);

	z = _mm_slli_epi64(z, 1);

	z = _mm_add_epi64(z, x);
	z = _mm_add_epi64(z, y);

	return z;
}

#define G1(row1l,row2l,row3l,row4l,row1h,row2h,row3h,row4h) \
	row1l = fBlaMka(row1l, row2l); \
	row1h = fBlaMka(row1h, row2h); \
	\
	row4l = _mm_xor_si128(row4l, row1l); \
	row4h = _mm_xor_si128(row4h, row1h); \
	\
	row4l = _mm_roti_epi64(row4l, -32); \
	row4h = _mm_roti_epi64(row4h, -32); \
	\
	row3l = fBlaMka(row3l, row4l); \
	row3h = fBlaMka(row3h, row4h); \
	\
	row2l = _mm_xor_si128(row2l, row3l); \
	row2h = _mm_xor_si128(row2h, row3h); \
	\
	row2l = _mm_roti_epi64(row2l, -24); \
	row2h = _mm_roti_epi64(row2h, -24); \

#define G2(row1l,row2l,row3l,row4l,row1h,row2h,row3h,row4h) \
	row1l = fBlaMka(row1l, row2l); \
	row1h = fBlaMka(row1h, row2h); \
	\
	row4l = _mm_xor_si128(row4l, row1l); \
	row4h = _mm_xor_si128(row4h, row1h); \
	\
	row4l = _mm_roti_epi64(row4l, -16); \
	row4h = _mm_roti_epi64(row4h, -16); \
	\
	row3l = fBlaMka(row3l, row4l); \
	row3h = fBlaMka(row3h, row4h); \
	\
	row2l = _mm_xor_si128(row2l, row3l); \
	row2h = _mm_xor_si128(row2h, row3h); \
	\
	row2l = _mm_roti_epi64(row2l, -63); \
	row2h = _mm_roti_epi64(row2h, -63); \


#define DIAGONALIZE(row1l,row2l,row3l,row4l,row1h,row2h,row3h,row4h) \
	t0 = _mm_alignr_epi8(row2h, row2l, 8); \
	t1 = _mm_alignr_epi8(row2l, row2h, 8); \
	row2l = t0; \
	row2h = t1; \
	\
	t0 = row3l; \
	row3l = row3h; \
	row3h = t0;    \
	\
	t0 = _mm_alignr_epi8(row4h, row4l, 8); \
	t1 = _mm_alignr_epi8(row4l, row4h, 8); \
	row4l = t1; \
	row4h = t0;

#define UNDIAGONALIZE(row1l,row2l,row3l,row4l,row1h,row2h,row3h,row4h) \
	t0 = _mm_alignr_epi8(row2l, row2h, 8); \
	t1 = _mm_alignr_epi8(row2h, row2l, 8); \
	row2l = t0; \
	row2h = t1; \
	\
	t0 = row3l; \
	row3l = row3h; \
	row3h = t0; \
	\
	t0 = _mm_alignr_epi8(row4l, row4h, 8); \
	t1 = _mm_alignr_epi8(row4h, row4l, 8); \
	row4l = t1; \
	row4h = t0;

#define BLAKE2_ROUND(row1l,row1h,row2l,row2h,row3l,row3h,row4l,row4h) \
	G1(row1l,row2l,row3l,row4l,row1h,row2h,row3h,row4h); \
	G2(row1l,row2l,row3l,row4l,row1h,row2h,row3h,row4h); \
	\
	DIAGONALIZE(row1l,row2l,row3l,row4l,row1h,row2h,row3h,row4h); \
	\
	G1(row1l,row2l,row3l,row4l,row1h,row2h,row3h,row4h); \
	G2(row1l,row2l,row3l,row4l,row1h,row2h,row3h,row4h); \
	\
	UNDIAGONALIZE(row1l,row2l,row3l,row4l,row1h,row2h,row3h,row4h);

/*
Here we implement an abstraction layer for the simpĺe requirements
of the Argon2 code. We only require 3 primitives---thread creation,
joining, and termination---so full emulation of the pthreads API
is unwarranted. Currently we wrap pthreads and Win32 threads.

The API defines 2 types: the function pointer type,
argon2_thread_func_t,
and the type of the thread handle---argon2_thread_handle_t.
*/
#if defined(_WIN32)
#include <process.h>
typedef unsigned(__stdcall *argon2_thread_func_t)(void *);
typedef uintptr_t argon2_thread_handle_t;
#else
#include <pthread.h>
typedef void *(*argon2_thread_func_t)(void *);
typedef pthread_t argon2_thread_handle_t;
#endif

#define SecureZeroMemory RtlSecureZeroMemory

/* Symbols visibility control */
#ifdef A2_VISCTL
#define ARGON2_PUBLIC __attribute__((visibility("default")))
#elif _MSC_VER
#define ARGON2_PUBLIC __declspec(dllexport)
#else
#define ARGON2_PUBLIC
#endif

/*
* Argon2 input parameter restrictions
*/

/* Minimum and maximum number of lanes (degree of parallelism) */
#define ARGON2_MIN_LANES UINT32_C(1)
#define ARGON2_MAX_LANES UINT32_C(0xFFFFFF)

/* Minimum and maximum number of threads */
#define ARGON2_MIN_THREADS UINT32_C(1)
#define ARGON2_MAX_THREADS UINT32_C(0xFFFFFF)

/* Number of synchronization points between lanes per pass */
#define ARGON2_SYNC_POINTS UINT32_C(4)

/* Minimum and maximum digest size in bytes */
#define ARGON2_MIN_OUTLEN UINT32_C(4)
#define ARGON2_MAX_OUTLEN UINT32_C(0xFFFFFFFF)

/* Minimum and maximum number of memory blocks (each of BLOCK_SIZE bytes) */
#define ARGON2_MIN_MEMORY (2 * ARGON2_SYNC_POINTS) /* 2 blocks per slice */

#define ARGON2_MIN(a, b) ((a) < (b) ? (a) : (b))
/* Max memory size is addressing-space/2, topping at 2^32 blocks (4 TB) */
#define ARGON2_MAX_MEMORY_BITS                                                 \
    ARGON2_MIN(UINT32_C(32), (sizeof(void *) * CHAR_BIT - 10 - 1))
#define ARGON2_MAX_MEMORY                                                      \
    ARGON2_MIN(UINT32_C(0xFFFFFFFF), UINT64_C(1) << ARGON2_MAX_MEMORY_BITS)

/* Minimum and maximum number of passes */
#define ARGON2_MIN_TIME UINT32_C(1)
#define ARGON2_MAX_TIME UINT32_C(0xFFFFFFFF)

/* Minimum and maximum password length in bytes */
#define ARGON2_MIN_PWD_LENGTH UINT32_C(0)
#define ARGON2_MAX_PWD_LENGTH UINT32_C(0xFFFFFFFF)

/* Minimum and maximum associated data length in bytes */
#define ARGON2_MIN_AD_LENGTH UINT32_C(0)
#define ARGON2_MAX_AD_LENGTH UINT32_C(0xFFFFFFFF)

/* Minimum and maximum salt length in bytes */
#define ARGON2_MIN_SALT_LENGTH UINT32_C(8)
#define ARGON2_MAX_SALT_LENGTH UINT32_C(0xFFFFFFFF)

/* Minimum and maximum key length in bytes */
#define ARGON2_MIN_SECRET UINT32_C(0)
#define ARGON2_MAX_SECRET UINT32_C(0xFFFFFFFF)

/* Flags to determine which fields are securely wiped (default = no wipe). */
#define ARGON2_DEFAULT_FLAGS UINT32_C(0)
#define ARGON2_FLAG_CLEAR_PASSWORD (UINT32_C(1) << 0)
#define ARGON2_FLAG_CLEAR_SECRET (UINT32_C(1) << 1)

/* Global flag to determine if we are wiping internal memory buffers. This flag
* is defined in core.c and deafults to 1 (wipe internal memory). */
extern int FLAG_clear_internal_memory;

/* Error codes */
typedef enum Argon2_ErrorCodes {
	ARGON2_OK = 0,

	ARGON2_OUTPUT_PTR_NULL = -1,

	ARGON2_OUTPUT_TOO_SHORT = -2,
	ARGON2_OUTPUT_TOO_LONG = -3,

	ARGON2_PWD_TOO_SHORT = -4,
	ARGON2_PWD_TOO_LONG = -5,

	ARGON2_SALT_TOO_SHORT = -6,
	ARGON2_SALT_TOO_LONG = -7,

	ARGON2_AD_TOO_SHORT = -8,
	ARGON2_AD_TOO_LONG = -9,

	ARGON2_SECRET_TOO_SHORT = -10,
	ARGON2_SECRET_TOO_LONG = -11,

	ARGON2_TIME_TOO_SMALL = -12,
	ARGON2_TIME_TOO_LARGE = -13,

	ARGON2_MEMORY_TOO_LITTLE = -14,
	ARGON2_MEMORY_TOO_MUCH = -15,

	ARGON2_LANES_TOO_FEW = -16,
	ARGON2_LANES_TOO_MANY = -17,

	ARGON2_PWD_PTR_MISMATCH = -18,    /* NULL ptr with non-zero length */
	ARGON2_SALT_PTR_MISMATCH = -19,   /* NULL ptr with non-zero length */
	ARGON2_SECRET_PTR_MISMATCH = -20, /* NULL ptr with non-zero length */
	ARGON2_AD_PTR_MISMATCH = -21,     /* NULL ptr with non-zero length */

	ARGON2_MEMORY_ALLOCATION_ERROR = -22,

	ARGON2_FREE_MEMORY_CBK_NULL = -23,
	ARGON2_ALLOCATE_MEMORY_CBK_NULL = -24,

	ARGON2_INCORRECT_PARAMETER = -25,
	ARGON2_INCORRECT_TYPE = -26,

	ARGON2_OUT_PTR_MISMATCH = -27,

	ARGON2_THREADS_TOO_FEW = -28,
	ARGON2_THREADS_TOO_MANY = -29,

	ARGON2_MISSING_ARGS = -30,

	ARGON2_ENCODING_FAIL = -31,

	ARGON2_DECODING_FAIL = -32,

	ARGON2_THREAD_FAIL = -33,

	ARGON2_DECODING_LENGTH_FAIL = -34,

	ARGON2_VERIFY_MISMATCH = -35
} argon2_error_codes;

/* Memory allocator types --- for external allocation */
typedef int(*allocate_fptr)(uint8_t **memory, size_t bytes_to_allocate);
typedef void(*deallocate_fptr)(uint8_t *memory, size_t bytes_to_allocate);

/* Argon2 external data structures */

/*
*****
* Context: structure to hold Argon2 inputs:
*  output array and its length,
*  password and its length,
*  salt and its length,
*  secret and its length,
*  associated data and its length,
*  number of passes, amount of used memory (in KBytes, can be rounded up a bit)
*  number of parallel threads that will be run.
* All the parameters above affect the output hash value.
* Additionally, two function pointers can be provided to allocate and
* deallocate the memory (if NULL, memory will be allocated internally).
* Also, three flags indicate whether to erase password, secret as soon as they
* are pre-hashed (and thus not needed anymore), and the entire memory
*****
* Simplest situation: you have output array out[8], password is stored in
* pwd[32], salt is stored in salt[16], you do not have keys nor associated
* data. You need to spend 1 GB of RAM and you run 5 passes of Argon2d with
* 4 parallel lanes.
* You want to erase the password, but you're OK with last pass not being
* erased. You want to use the default memory allocator.
* Then you initialize:
Argon2_Context(out,8,pwd,32,salt,16,NULL,0,NULL,0,5,1<<20,4,4,NULL,NULL,true,false,false,false)
*/
typedef struct Argon2_Context {
	uint8_t *out;    /* output array */
	uint32_t outlen; /* digest length */

	uint8_t *pwd;    /* password array */
	uint32_t pwdlen; /* password length */

	uint8_t *salt;    /* salt array */
	uint32_t saltlen; /* salt length */

	uint8_t *secret;    /* key array */
	uint32_t secretlen; /* key length */

	uint8_t *ad;    /* associated data array */
	uint32_t adlen; /* associated data length */

	uint32_t t_cost;  /* number of passes */
	uint32_t m_cost;  /* amount of memory requested (KB) */
	uint32_t lanes;   /* number of lanes */
	uint32_t threads; /* maximum number of threads */

	uint32_t version; /* version number */

	allocate_fptr allocate_cbk; /* pointer to memory allocator */
	deallocate_fptr free_cbk;   /* pointer to memory deallocator */

	uint32_t flags; /* array of bool options */
} argon2_context;



/* Argon2 primitive type */
typedef enum Argon2_type {
	Argon2_d = 0,
	Argon2_i = 1,
	Argon2_id = 2
} argon2_type;

/* Version of the algorithm */
typedef enum Argon2_version {
	ARGON2_VERSION_10 = 0x10,
	ARGON2_VERSION_13 = 0x13,
	ARGON2_VERSION_NUMBER = ARGON2_VERSION_13
} argon2_version;

/*
* Function that gives the string representation of an argon2_type.
* @param type The argon2_type that we want the string for
* @param uppercase Whether the string should have the first letter uppercase
* @return NULL if invalid type, otherwise the string representation.
*/
ARGON2_PUBLIC const char *argon2_type2string(argon2_type type, int uppercase);

/*
* Function that performs memory-hard hashing with certain degree of parallelism
* @param  context  Pointer to the Argon2 internal structure
* @return Error code if smth is wrong, ARGON2_OK otherwise
*/
ARGON2_PUBLIC int argon2_ctx(argon2_context *context, argon2_type type);

/**
* Hashes a password with Argon2i, producing an encoded hash
* @param t_cost Number of iterations
* @param m_cost Sets memory usage to m_cost kibibytes
* @param parallelism Number of threads and compute lanes
* @param pwd Pointer to password
* @param pwdlen Password size in bytes
* @param salt Pointer to salt
* @param saltlen Salt size in bytes
* @param hashlen Desired length of the hash in bytes
* @param encoded Buffer where to write the encoded hash
* @param encodedlen Size of the buffer (thus max size of the encoded hash)
* @pre   Different parallelism levels will give different results
* @pre   Returns ARGON2_OK if successful
*/
ARGON2_PUBLIC int argon2i_hash_encoded(const uint32_t t_cost,
	const uint32_t m_cost,
	const uint32_t parallelism,
	const void *pwd, const size_t pwdlen,
	const void *salt, const size_t saltlen,
	const size_t hashlen, char *encoded,
	const size_t encodedlen);

/**
* Hashes a password with Argon2i, producing a raw hash by allocating memory at
* @hash
* @param t_cost Number of iterations
* @param m_cost Sets memory usage to m_cost kibibytes
* @param parallelism Number of threads and compute lanes
* @param pwd Pointer to password
* @param pwdlen Password size in bytes
* @param salt Pointer to salt
* @param saltlen Salt size in bytes
* @param hash Buffer where to write the raw hash - updated by the function
* @param hashlen Desired length of the hash in bytes
* @pre   Different parallelism levels will give different results
* @pre   Returns ARGON2_OK if successful
*/
ARGON2_PUBLIC int argon2i_hash_raw(const uint32_t t_cost, const uint32_t m_cost,
	const uint32_t parallelism, const void *pwd,
	const size_t pwdlen, const void *salt,
	const size_t saltlen, void *hash,
	const size_t hashlen);

ARGON2_PUBLIC int argon2d_hash_encoded(const uint32_t t_cost,
	const uint32_t m_cost,
	const uint32_t parallelism,
	const void *pwd, const size_t pwdlen,
	const void *salt, const size_t saltlen,
	const size_t hashlen, char *encoded,
	const size_t encodedlen);

ARGON2_PUBLIC int argon2d_hash_raw(const uint32_t t_cost, const uint32_t m_cost,
	const uint32_t parallelism, const void *pwd,
	const size_t pwdlen, const void *salt,
	const size_t saltlen, void *hash,
	const size_t hashlen);

ARGON2_PUBLIC int argon2id_hash_encoded(const uint32_t t_cost,
	const uint32_t m_cost,
	const uint32_t parallelism,
	const void *pwd, const size_t pwdlen,
	const void *salt, const size_t saltlen,
	const size_t hashlen, char *encoded,
	const size_t encodedlen);

ARGON2_PUBLIC int argon2id_hash_raw(const uint32_t t_cost,
	const uint32_t m_cost,
	const uint32_t parallelism, const void *pwd,
	const size_t pwdlen, const void *salt,
	const size_t saltlen, void *hash,
	const size_t hashlen);

/* generic function underlying the above ones */
ARGON2_PUBLIC int argon2_hash(const uint32_t t_cost, const uint32_t m_cost,
	const uint32_t parallelism, const void *pwd,
	const size_t pwdlen, const void *salt,
	const size_t saltlen, void *hash,
	const size_t hashlen, char *encoded,
	const size_t encodedlen, argon2_type type,
	const uint32_t version);

/**
* Verifies a password against an encoded string
* Encoded string is restricted as in validate_inputs()
* @param encoded String encoding parameters, salt, hash
* @param pwd Pointer to password
* @pre   Returns ARGON2_OK if successful
*/
ARGON2_PUBLIC int argon2i_verify(const char *encoded, const void *pwd,
	const size_t pwdlen);

ARGON2_PUBLIC int argon2d_verify(const char *encoded, const void *pwd,
	const size_t pwdlen);

ARGON2_PUBLIC int argon2id_verify(const char *encoded, const void *pwd,
	const size_t pwdlen);

/* generic function underlying the above ones */
ARGON2_PUBLIC int argon2_verify(const char *encoded, const void *pwd,
	const size_t pwdlen, argon2_type type);

/**
* Argon2d: Version of Argon2 that picks memory blocks depending
* on the password and salt. Only for side-channel-free
* environment!!
*****
* @param  context  Pointer to current Argon2 context
* @return  Zero if successful, a non zero error code otherwise
*/
ARGON2_PUBLIC int argon2d_ctx(argon2_context *context);

/**
* Argon2i: Version of Argon2 that picks memory blocks
* independent on the password and salt. Good for side-channels,
* but worse w.r.t. tradeoff attacks if only one pass is used.
*****
* @param  context  Pointer to current Argon2 context
* @return  Zero if successful, a non zero error code otherwise
*/
ARGON2_PUBLIC int argon2i_ctx(argon2_context *context);

/**
* Argon2id: Version of Argon2 where the first half-pass over memory is
* password-independent, the rest are password-dependent (on the password and
* salt). OK against side channels (they reduce to 1/2-pass Argon2i), and
* better with w.r.t. tradeoff attacks (similar to Argon2d).
*****
* @param  context  Pointer to current Argon2 context
* @return  Zero if successful, a non zero error code otherwise
*/
ARGON2_PUBLIC int argon2id_ctx(argon2_context *context);

/**
* Verify if a given password is correct for Argon2d hashing
* @param  context  Pointer to current Argon2 context
* @param  hash  The password hash to verify. The length of the hash is
* specified by the context outlen member
* @return  Zero if successful, a non zero error code otherwise
*/
ARGON2_PUBLIC int argon2d_verify_ctx(argon2_context *context, const char *hash);

/**
* Verify if a given password is correct for Argon2i hashing
* @param  context  Pointer to current Argon2 context
* @param  hash  The password hash to verify. The length of the hash is
* specified by the context outlen member
* @return  Zero if successful, a non zero error code otherwise
*/
ARGON2_PUBLIC int argon2i_verify_ctx(argon2_context *context, const char *hash);

/**
* Verify if a given password is correct for Argon2id hashing
* @param  context  Pointer to current Argon2 context
* @param  hash  The password hash to verify. The length of the hash is
* specified by the context outlen member
* @return  Zero if successful, a non zero error code otherwise
*/
ARGON2_PUBLIC int argon2id_verify_ctx(argon2_context *context,
	const char *hash);

/* generic function underlying the above ones */
ARGON2_PUBLIC int argon2_verify_ctx(argon2_context *context, const char *hash,
	argon2_type type);

/**
* Get the associated error message for given error code
* @return  The error message associated with the given error code
*/
ARGON2_PUBLIC const char *argon2_error_message(int error_code);

/**
* Returns the encoded hash length for the given input parameters
* @param t_cost  Number of iterations
* @param m_cost  Memory usage in kibibytes
* @param parallelism  Number of threads; used to compute lanes
* @param saltlen  Salt size in bytes
* @param hashlen  Hash size in bytes
* @param type The argon2_type that we want the encoded length for
* @return  The encoded hash length in bytes
*/
ARGON2_PUBLIC size_t argon2_encodedlen(uint32_t t_cost, uint32_t m_cost,
	uint32_t parallelism, uint32_t saltlen,
	uint32_t hashlen, argon2_type type);



int validate_inputs(const argon2_context *context) {
	if (NULL == context) {
		return ARGON2_INCORRECT_PARAMETER;
	}

	if (NULL == context->out) {
		return ARGON2_OUTPUT_PTR_NULL;
	}

	/* Validate output length */
	if (ARGON2_MIN_OUTLEN > context->outlen) {
		return ARGON2_OUTPUT_TOO_SHORT;
	}

	if (ARGON2_MAX_OUTLEN < context->outlen) {
		return ARGON2_OUTPUT_TOO_LONG;
	}

	/* Validate password (required param) */
	if (NULL == context->pwd) {
		if (0 != context->pwdlen) {
			return ARGON2_PWD_PTR_MISMATCH;
		}
	}

	if (ARGON2_MIN_PWD_LENGTH > context->pwdlen) {
		return ARGON2_PWD_TOO_SHORT;
	}

	if (ARGON2_MAX_PWD_LENGTH < context->pwdlen) {
		return ARGON2_PWD_TOO_LONG;
	}

	/* Validate salt (required param) */
	if (NULL == context->salt) {
		if (0 != context->saltlen) {
			return ARGON2_SALT_PTR_MISMATCH;
		}
	}

	if (ARGON2_MIN_SALT_LENGTH > context->saltlen) {
		return ARGON2_SALT_TOO_SHORT;
	}

	if (ARGON2_MAX_SALT_LENGTH < context->saltlen) {
		return ARGON2_SALT_TOO_LONG;
	}

	/* Validate secret (optional param) */
	if (NULL == context->secret) {
		if (0 != context->secretlen) {
			return ARGON2_SECRET_PTR_MISMATCH;
		}
	}
	else {
		if (ARGON2_MIN_SECRET > context->secretlen) {
			return ARGON2_SECRET_TOO_SHORT;
		}
		if (ARGON2_MAX_SECRET < context->secretlen) {
			return ARGON2_SECRET_TOO_LONG;
		}
	}

	/* Validate associated data (optional param) */
	if (NULL == context->ad) {
		if (0 != context->adlen) {
			return ARGON2_AD_PTR_MISMATCH;
		}
	}
	else {
		if (ARGON2_MIN_AD_LENGTH > context->adlen) {
			return ARGON2_AD_TOO_SHORT;
		}
		if (ARGON2_MAX_AD_LENGTH < context->adlen) {
			return ARGON2_AD_TOO_LONG;
		}
	}

	/* Validate memory cost */
	if (ARGON2_MIN_MEMORY > context->m_cost) {
		return ARGON2_MEMORY_TOO_LITTLE;
	}

	if (ARGON2_MAX_MEMORY < context->m_cost) {
		return ARGON2_MEMORY_TOO_MUCH;
	}

	if (context->m_cost < 8 * context->lanes) {
		return ARGON2_MEMORY_TOO_LITTLE;
	}

	/* Validate time cost */
	if (ARGON2_MIN_TIME > context->t_cost) {
		return ARGON2_TIME_TOO_SMALL;
	}

	if (ARGON2_MAX_TIME < context->t_cost) {
		return ARGON2_TIME_TOO_LARGE;
	}

	/* Validate lanes */
	if (ARGON2_MIN_LANES > context->lanes) {
		return ARGON2_LANES_TOO_FEW;
	}

	if (ARGON2_MAX_LANES < context->lanes) {
		return ARGON2_LANES_TOO_MANY;
	}

	/* Validate threads */
	if (ARGON2_MIN_THREADS > context->threads) {
		return ARGON2_THREADS_TOO_FEW;
	}

	if (ARGON2_MAX_THREADS < context->threads) {
		return ARGON2_THREADS_TOO_MANY;
	}

	if (NULL != context->allocate_cbk && NULL == context->free_cbk) {
		return ARGON2_FREE_MEMORY_CBK_NULL;
	}

	if (NULL == context->allocate_cbk && NULL != context->free_cbk) {
		return ARGON2_ALLOCATE_MEMORY_CBK_NULL;
	}

	return ARGON2_OK;
}

/**********************Argon2 internal constants*******************************/

enum argon2_core_constants {
	/* Memory block size in bytes */
	ARGON2_BLOCK_SIZE = 1024,
	ARGON2_QWORDS_IN_BLOCK = ARGON2_BLOCK_SIZE / 8,
	ARGON2_OWORDS_IN_BLOCK = ARGON2_BLOCK_SIZE / 16,

	/* Number of pseudo-random values generated by one call to Blake in Argon2i
	to
	generate reference block positions */
	ARGON2_ADDRESSES_IN_BLOCK = 128,

	/* Pre-hashing digest length and its extension*/
	ARGON2_PREHASH_DIGEST_LENGTH = 64,
	ARGON2_PREHASH_SEED_LENGTH = 72
};

/*
* Structure for the (1KB) memory block implemented as 128 64-bit words.
* Memory blocks can be copied, XORed. Internal words can be accessed by [] (no
* bounds checking).
*/
typedef struct block_ { uint64_t v[ARGON2_QWORDS_IN_BLOCK]; } block;


/*
* Argon2 instance: memory pointer, number of passes, amount of memory, type,
* and derived values.
* Used to evaluate the number and location of blocks to construct in each
* thread
*/
typedef struct Argon2_instance_t {
	block *memory;          /* Memory pointer */
	uint32_t version;
	uint32_t passes;        /* Number of passes */
	uint32_t memory_blocks; /* Number of blocks in memory */
	uint32_t segment_length;
	uint32_t lane_length;
	uint32_t lanes;
	uint32_t threads;
	argon2_type type;
	int print_internals; /* whether to print the memory blocks */
	argon2_context *context_ptr; /* points back to original context */
} argon2_instance_t;


const char *argon2_type2string(argon2_type type, int uppercase) {
	switch (type) {
	case Argon2_d:
		return uppercase ? "Argon2d" : "argon2d";
	case Argon2_i:
		return uppercase ? "Argon2i" : "argon2i";
	case Argon2_id:
		return uppercase ? "Argon2id" : "argon2id";
	}

	return NULL;
}

void initial_kat(const uint8_t *blockhash, const argon2_context *context,
	argon2_type type) {
	unsigned i;

	if (blockhash != NULL && context != NULL) {
		printf("=======================================\n");

		printf("%s version number %d\n", argon2_type2string(type, 1),
			context->version);

		printf("=======================================\n");


		printf("Memory: %u KiB, Iterations: %u, Parallelism: %u lanes, Tag "
			"length: %u bytes\n",
			context->m_cost, context->t_cost, context->lanes,
			context->outlen);

		printf("Password[%u]: ", context->pwdlen);

		if (context->flags & ARGON2_FLAG_CLEAR_PASSWORD) {
			printf("CLEARED\n");
		}
		else {
			for (i = 0; i < context->pwdlen; ++i) {
				printf("%2.2x ", ((unsigned char *)context->pwd)[i]);
			}

			printf("\n");
		}

		printf("Salt[%u]: ", context->saltlen);

		for (i = 0; i < context->saltlen; ++i) {
			printf("%2.2x ", ((unsigned char *)context->salt)[i]);
		}

		printf("\n");

		printf("Secret[%u]: ", context->secretlen);

		if (context->flags & ARGON2_FLAG_CLEAR_SECRET) {
			printf("CLEARED\n");
		}
		else {
			for (i = 0; i < context->secretlen; ++i) {
				printf("%2.2x ", ((unsigned char *)context->secret)[i]);
			}

			printf("\n");
		}

		printf("Associated data[%u]: ", context->adlen);

		for (i = 0; i < context->adlen; ++i) {
			printf("%2.2x ", ((unsigned char *)context->ad)[i]);
		}

		printf("\n");

		printf("Pre-hashing digest: ");

		for (i = 0; i < ARGON2_PREHASH_DIGEST_LENGTH; ++i) {
			printf("%2.2x ", ((unsigned char *)blockhash)[i]);
		}

		printf("\n");
	}
}

void print_tag(const void *out, uint32_t outlen) {
	unsigned i;
	if (out != NULL) {
		printf("Tag: ");

		for (i = 0; i < outlen; ++i) {
			printf("%2.2x ", ((uint8_t *)out)[i]);
		}

		printf("\n");
	}
}



void internal_kat(const argon2_instance_t *instance, uint32_t pass) {

	if (instance != NULL) {
		uint32_t i, j;
		printf("\n After pass %u:\n", pass);

		for (i = 0; i < instance->memory_blocks; ++i) {
			uint32_t how_many_words =
				(instance->memory_blocks > ARGON2_QWORDS_IN_BLOCK)
				? 1
				: ARGON2_QWORDS_IN_BLOCK;

			for (j = 0; j < how_many_words; ++j)
				printf("Block %.4u [%3u]: %016" PRIx64 "\n", i, j,
					instance->memory[i].v[j]);
		}
	}
}


static void fatal(const char *error) {
	fprintf(stderr, "Error: %s\n", error);
	exit(1);
}



/***************Memory functions*****************/

int allocate_memory(const argon2_context *context, uint8_t **memory,
	size_t num, size_t size) {
	size_t memory_size = num*size;
	if (memory == NULL) {
		return ARGON2_MEMORY_ALLOCATION_ERROR;
	}

	/* 1. Check for multiplication overflow */
	if (size != 0 && memory_size / size != num) {
		return ARGON2_MEMORY_ALLOCATION_ERROR;
	}

	/* 2. Try to allocate with appropriate allocator */
	if (context->allocate_cbk) {
		(context->allocate_cbk)(memory, memory_size);
	}
	else {
		*memory = (uint8_t*)malloc(memory_size);
	}

	if (*memory == NULL) {
		return ARGON2_MEMORY_ALLOCATION_ERROR;
	}

	return ARGON2_OK;
}

enum blake2b_constant {
	BLAKE2B_BLOCKBYTES = 128,
	BLAKE2B_OUTBYTES = 64,
	BLAKE2B_KEYBYTES = 64,
	BLAKE2B_SALTBYTES = 16,
	BLAKE2B_PERSONALBYTES = 16
};

typedef struct __blake2b_state {
	uint64_t h[8];
	uint64_t t[2];
	uint64_t f[2];
	uint8_t buf[BLAKE2B_BLOCKBYTES];
	unsigned buflen;
	unsigned outlen;
	uint8_t last_node;
} blake2b_state;

typedef struct __blake2b_param {
	uint8_t digest_length;                   /* 1 */
	uint8_t key_length;                      /* 2 */
	uint8_t fanout;                          /* 3 */
	uint8_t depth;                           /* 4 */
	uint32_t leaf_length;                    /* 8 */
	uint64_t node_offset;                    /* 16 */
	uint8_t node_depth;                      /* 17 */
	uint8_t inner_length;                    /* 18 */
	uint8_t reserved[14];                    /* 32 */
	uint8_t salt[BLAKE2B_SALTBYTES];         /* 48 */
	uint8_t personal[BLAKE2B_PERSONALBYTES]; /* 64 */
} blake2b_param;


void NOT_OPTIMIZED secure_wipe_memory(void *v, size_t n) {
#if defined(_MSC_VER) && VC_GE_2005(_MSC_VER)
	SecureZeroMemory(v, n);
#elif defined memset_s
	memset_s(v, n, 0, n);
#elif defined(__OpenBSD__)
	explicit_bzero(v, n);
#else
	static void *(*const volatile memset_sec)(void *, int, size_t) = &memset;
	memset_sec(v, 0, n);
#endif
}


/* Memory clear flag defaults to true. */
int FLAG_clear_internal_memory = 1;
void clear_internal_memory(void *v, size_t n) {
	if (FLAG_clear_internal_memory && v) {
		secure_wipe_memory(v, n);
	}
}


static BLAKE2_INLINE void blake2b_set_lastnode(blake2b_state *S) {
	S->f[1] = (uint64_t)-1;
}

static BLAKE2_INLINE void blake2b_set_lastblock(blake2b_state *S) {
	if (S->last_node) {
		blake2b_set_lastnode(S);
	}
	S->f[0] = (uint64_t)-1;
}

static BLAKE2_INLINE void blake2b_invalidate_state(blake2b_state *S) {
	clear_internal_memory(S, sizeof(*S));      /* wipe */
	blake2b_set_lastblock(S); /* invalidate for further use */
}

static const uint64_t blake2b_IV[8] = {
	UINT64_C(0x6a09e667f3bcc908), UINT64_C(0xbb67ae8584caa73b),
	UINT64_C(0x3c6ef372fe94f82b), UINT64_C(0xa54ff53a5f1d36f1),
	UINT64_C(0x510e527fade682d1), UINT64_C(0x9b05688c2b3e6c1f),
	UINT64_C(0x1f83d9abfb41bd6b), UINT64_C(0x5be0cd19137e2179) };

static BLAKE2_INLINE void blake2b_init0(blake2b_state *S) {
	memset(S, 0, sizeof(*S));
	memcpy(S->h, blake2b_IV, sizeof(S->h));
}

static BLAKE2_INLINE uint64_t load64(const void *src) {
#if defined(NATIVE_LITTLE_ENDIAN)
	uint64_t w;
	memcpy(&w, src, sizeof w);
	return w;
#else
	const uint8_t *p = (const uint8_t *)src;
	uint64_t w = *p++;
	w |= (uint64_t)(*p++) << 8;
	w |= (uint64_t)(*p++) << 16;
	w |= (uint64_t)(*p++) << 24;
	w |= (uint64_t)(*p++) << 32;
	w |= (uint64_t)(*p++) << 40;
	w |= (uint64_t)(*p++) << 48;
	w |= (uint64_t)(*p++) << 56;
	return w;
#endif
}


int blake2b_init_param(blake2b_state *S, const blake2b_param *P) {
	const unsigned char *p = (const unsigned char *)P;
	unsigned int i;

	if (NULL == P || NULL == S) {
		return -1;
	}

	blake2b_init0(S);
	/* IV XOR Parameter Block */
	for (i = 0; i < 8; ++i) {
		S->h[i] ^= load64(&p[i * sizeof(S->h[i])]);
	}
	S->outlen = P->digest_length;
	return 0;
}

/* Sequential blake2b initialization */
int blake2b_init(blake2b_state *S, size_t outlen) {
	blake2b_param P;

	if (S == NULL) {
		return -1;
	}

	if ((outlen == 0) || (outlen > BLAKE2B_OUTBYTES)) {
		blake2b_invalidate_state(S);
		return -1;
	}

	/* Setup Parameter Block for unkeyed BLAKE2 */
	P.digest_length = (uint8_t)outlen;
	P.key_length = 0;
	P.fanout = 1;
	P.depth = 1;
	P.leaf_length = 0;
	P.node_offset = 0;
	P.node_depth = 0;
	P.inner_length = 0;
	memset(P.reserved, 0, sizeof(P.reserved));
	memset(P.salt, 0, sizeof(P.salt));
	memset(P.personal, 0, sizeof(P.personal));

	return blake2b_init_param(S, &P);
}


static const unsigned int blake2b_sigma[12][16] = {
	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
	{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
	{ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
	{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
	{ 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
	{ 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
	{ 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
	{ 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
	{ 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
	{ 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 },
	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
	{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
};

static BLAKE2_INLINE uint32_t rotr32(const uint32_t w, const unsigned c) {
	return (w >> c) | (w << (32 - c));
}

static BLAKE2_INLINE uint64_t rotr64(const uint64_t w, const unsigned c) {
	return (w >> c) | (w << (64 - c));
}

static void blake2b_compress(blake2b_state *S, const uint8_t *block) {
	uint64_t m[16];
	uint64_t v[16];
	unsigned int i, r;

	for (i = 0; i < 16; ++i) {
		m[i] = load64(block + i * sizeof(m[i]));
	}

	for (i = 0; i < 8; ++i) {
		v[i] = S->h[i];
	}

	v[8] = blake2b_IV[0];
	v[9] = blake2b_IV[1];
	v[10] = blake2b_IV[2];
	v[11] = blake2b_IV[3];
	v[12] = blake2b_IV[4] ^ S->t[0];
	v[13] = blake2b_IV[5] ^ S->t[1];
	v[14] = blake2b_IV[6] ^ S->f[0];
	v[15] = blake2b_IV[7] ^ S->f[1];

#define G(r, i, a, b, c, d)                                                    \
    do {                                                                       \
        a = a + b + m[blake2b_sigma[r][2 * i + 0]];                            \
        d = rotr64(d ^ a, 32);                                                 \
        c = c + d;                                                             \
        b = rotr64(b ^ c, 24);                                                 \
        a = a + b + m[blake2b_sigma[r][2 * i + 1]];                            \
        d = rotr64(d ^ a, 16);                                                 \
        c = c + d;                                                             \
        b = rotr64(b ^ c, 63);                                                 \
    } while ((void)0, 0)

#define ROUND(r)                                                               \
    do {                                                                       \
        G(r, 0, v[0], v[4], v[8], v[12]);                                      \
        G(r, 1, v[1], v[5], v[9], v[13]);                                      \
        G(r, 2, v[2], v[6], v[10], v[14]);                                     \
        G(r, 3, v[3], v[7], v[11], v[15]);                                     \
        G(r, 4, v[0], v[5], v[10], v[15]);                                     \
        G(r, 5, v[1], v[6], v[11], v[12]);                                     \
        G(r, 6, v[2], v[7], v[8], v[13]);                                      \
        G(r, 7, v[3], v[4], v[9], v[14]);                                      \
    } while ((void)0, 0)

	for (r = 0; r < 12; ++r) {
		ROUND(r);
	}

	for (i = 0; i < 8; ++i) {
		S->h[i] = S->h[i] ^ v[i] ^ v[i + 8];
	}

#undef G
#undef ROUND
}



static BLAKE2_INLINE void blake2b_increment_counter(blake2b_state *S,
	uint64_t inc) {
	S->t[0] += inc;
	S->t[1] += (S->t[0] < inc);
}


static BLAKE2_INLINE void store64(void *dst, uint64_t w) {
#if defined(NATIVE_LITTLE_ENDIAN)
	memcpy(dst, &w, sizeof w);
#else
	uint8_t *p = (uint8_t *)dst;
	*p++ = (uint8_t)w;
	w >>= 8;
	*p++ = (uint8_t)w;
	w >>= 8;
	*p++ = (uint8_t)w;
	w >>= 8;
	*p++ = (uint8_t)w;
	w >>= 8;
	*p++ = (uint8_t)w;
	w >>= 8;
	*p++ = (uint8_t)w;
	w >>= 8;
	*p++ = (uint8_t)w;
	w >>= 8;
	*p++ = (uint8_t)w;
#endif
}

int blake2b_final(blake2b_state *S, void *out, size_t outlen) {
	uint8_t buffer[BLAKE2B_OUTBYTES] = { 0 };
	unsigned int i;

	/* Sanity checks */
	if (S == NULL || out == NULL || outlen < S->outlen) {
		return -1;
	}

	/* Is this a reused state? */
	if (S->f[0] != 0) {
		return -1;
	}

	blake2b_increment_counter(S, S->buflen);
	blake2b_set_lastblock(S);
	memset(&S->buf[S->buflen], 0, BLAKE2B_BLOCKBYTES - S->buflen); /* Padding */
	blake2b_compress(S, S->buf);

	for (i = 0; i < 8; ++i) { /* Output full hash to temp buffer */
		store64(buffer + sizeof(S->h[i]) * i, S->h[i]);
	}

	memcpy(out, buffer, S->outlen);
	clear_internal_memory(buffer, sizeof(buffer));
	clear_internal_memory(S->buf, sizeof(S->buf));
	clear_internal_memory(S->h, sizeof(S->h));
	return 0;
}


static BLAKE2_INLINE void store32(void *dst, uint32_t w) {
#if defined(NATIVE_LITTLE_ENDIAN)
	memcpy(dst, &w, sizeof w);
#else
	uint8_t *p = (uint8_t *)dst;
	*p++ = (uint8_t)w;
	w >>= 8;
	*p++ = (uint8_t)w;
	w >>= 8;
	*p++ = (uint8_t)w;
	w >>= 8;
	*p++ = (uint8_t)w;
#endif
}


int blake2b_update(blake2b_state *S, const void *in, size_t inlen) {
	const uint8_t *pin = (const uint8_t *)in;

	if (inlen == 0) {
		return 0;
	}

	/* Sanity check */
	if (S == NULL || in == NULL) {
		return -1;
	}

	/* Is this a reused state? */
	if (S->f[0] != 0) {
		return -1;
	}

	if (S->buflen + inlen > BLAKE2B_BLOCKBYTES) {
		/* Complete current block */
		size_t left = S->buflen;
		size_t fill = BLAKE2B_BLOCKBYTES - left;
		memcpy(&S->buf[left], pin, fill);
		blake2b_increment_counter(S, BLAKE2B_BLOCKBYTES);
		blake2b_compress(S, S->buf);
		S->buflen = 0;
		inlen -= fill;
		pin += fill;
		/* Avoid buffer copies when possible */
		while (inlen > BLAKE2B_BLOCKBYTES) {
			blake2b_increment_counter(S, BLAKE2B_BLOCKBYTES);
			blake2b_compress(S, pin);
			inlen -= BLAKE2B_BLOCKBYTES;
			pin += BLAKE2B_BLOCKBYTES;
		}
	}
	memcpy(&S->buf[S->buflen], pin, inlen);
	S->buflen += (unsigned int)inlen;
	return 0;
}


void initial_hash(uint8_t *blockhash, argon2_context *context,
	argon2_type type) {
	blake2b_state BlakeHash;
	uint8_t value[sizeof(uint32_t)];

	if (NULL == context || NULL == blockhash) {
		return;
	}

	blake2b_init(&BlakeHash, ARGON2_PREHASH_DIGEST_LENGTH);

	store32(&value, context->lanes);
	blake2b_update(&BlakeHash, (const uint8_t *)&value, sizeof(value));

	store32(&value, context->outlen);
	blake2b_update(&BlakeHash, (const uint8_t *)&value, sizeof(value));

	store32(&value, context->m_cost);
	blake2b_update(&BlakeHash, (const uint8_t *)&value, sizeof(value));

	store32(&value, context->t_cost);
	blake2b_update(&BlakeHash, (const uint8_t *)&value, sizeof(value));

	store32(&value, context->version);
	blake2b_update(&BlakeHash, (const uint8_t *)&value, sizeof(value));

	store32(&value, (uint32_t)type);
	blake2b_update(&BlakeHash, (const uint8_t *)&value, sizeof(value));

	store32(&value, context->pwdlen);
	blake2b_update(&BlakeHash, (const uint8_t *)&value, sizeof(value));

	if (context->pwd != NULL) {
		blake2b_update(&BlakeHash, (const uint8_t *)context->pwd,
			context->pwdlen);

		if (context->flags & ARGON2_FLAG_CLEAR_PASSWORD) {
			secure_wipe_memory(context->pwd, context->pwdlen);
			context->pwdlen = 0;
		}
	}

	store32(&value, context->saltlen);
	blake2b_update(&BlakeHash, (const uint8_t *)&value, sizeof(value));

	if (context->salt != NULL) {
		blake2b_update(&BlakeHash, (const uint8_t *)context->salt,
			context->saltlen);
	}

	store32(&value, context->secretlen);
	blake2b_update(&BlakeHash, (const uint8_t *)&value, sizeof(value));

	if (context->secret != NULL) {
		blake2b_update(&BlakeHash, (const uint8_t *)context->secret,
			context->secretlen);

		if (context->flags & ARGON2_FLAG_CLEAR_SECRET) {
			secure_wipe_memory(context->secret, context->secretlen);
			context->secretlen = 0;
		}
	}

	store32(&value, context->adlen);
	blake2b_update(&BlakeHash, (const uint8_t *)&value, sizeof(value));

	if (context->ad != NULL) {
		blake2b_update(&BlakeHash, (const uint8_t *)context->ad,
			context->adlen);
	}

	blake2b_final(&BlakeHash, blockhash, ARGON2_PREHASH_DIGEST_LENGTH);
}


int blake2b_init_key(blake2b_state *S, size_t outlen, const void *key,
	size_t keylen) {
	blake2b_param P;

	if (S == NULL) {
		return -1;
	}

	if ((outlen == 0) || (outlen > BLAKE2B_OUTBYTES)) {
		blake2b_invalidate_state(S);
		return -1;
	}

	if ((key == 0) || (keylen == 0) || (keylen > BLAKE2B_KEYBYTES)) {
		blake2b_invalidate_state(S);
		return -1;
	}

	/* Setup Parameter Block for keyed BLAKE2 */
	P.digest_length = (uint8_t)outlen;
	P.key_length = (uint8_t)keylen;
	P.fanout = 1;
	P.depth = 1;
	P.leaf_length = 0;
	P.node_offset = 0;
	P.node_depth = 0;
	P.inner_length = 0;
	memset(P.reserved, 0, sizeof(P.reserved));
	memset(P.salt, 0, sizeof(P.salt));
	memset(P.personal, 0, sizeof(P.personal));

	if (blake2b_init_param(S, &P) < 0) {
		blake2b_invalidate_state(S);
		return -1;
	}

	{
		uint8_t block[BLAKE2B_BLOCKBYTES];
		memset(block, 0, BLAKE2B_BLOCKBYTES);
		memcpy(block, key, keylen);
		blake2b_update(S, block, BLAKE2B_BLOCKBYTES);
		/* Burn the key from stack */
		clear_internal_memory(block, BLAKE2B_BLOCKBYTES);
	}
	return 0;
}

int blake2b(void *out, size_t outlen, const void *in, size_t inlen,
	const void *key, size_t keylen) {
	blake2b_state S;
	int ret = -1;

	/* Verify parameters */
	if (NULL == in && inlen > 0) {
		goto fail;
	}

	if (NULL == out || outlen == 0 || outlen > BLAKE2B_OUTBYTES) {
		goto fail;
	}

	if ((NULL == key && keylen > 0) || keylen > BLAKE2B_KEYBYTES) {
		goto fail;
	}

	if (keylen > 0) {
		if (blake2b_init_key(&S, outlen, key, keylen) < 0) {
			goto fail;
		}
	}
	else {
		if (blake2b_init(&S, outlen) < 0) {
			goto fail;
		}
	}

	if (blake2b_update(&S, in, inlen) < 0) {
		goto fail;
	}
	ret = blake2b_final(&S, out, outlen);

fail:
	clear_internal_memory(&S, sizeof(S));
	return ret;
}

/* Argon2 Team - Begin Code */
int blake2b_long(void *pout, size_t outlen, const void *in, size_t inlen) {
	uint8_t *out = (uint8_t *)pout;
	blake2b_state blake_state;
	uint8_t outlen_bytes[sizeof(uint32_t)] = { 0 };
	int ret = -1;

	if (outlen > UINT32_MAX) {
		goto fail;
	}

	/* Ensure little-endian byte order! */
	store32(outlen_bytes, (uint32_t)outlen);

#define TRY(statement)                                                         \
    do {                                                                       \
        ret = statement;                                                       \
        if (ret < 0) {                                                         \
            goto fail;                                                         \
        }                                                                      \
    } while ((void)0, 0)

	if (outlen <= BLAKE2B_OUTBYTES) {
		TRY(blake2b_init(&blake_state, outlen));
		TRY(blake2b_update(&blake_state, outlen_bytes, sizeof(outlen_bytes)));
		TRY(blake2b_update(&blake_state, in, inlen));
		TRY(blake2b_final(&blake_state, out, outlen));
	}
	else {
		uint32_t toproduce;
		uint8_t out_buffer[BLAKE2B_OUTBYTES];
		uint8_t in_buffer[BLAKE2B_OUTBYTES];
		TRY(blake2b_init(&blake_state, BLAKE2B_OUTBYTES));
		TRY(blake2b_update(&blake_state, outlen_bytes, sizeof(outlen_bytes)));
		TRY(blake2b_update(&blake_state, in, inlen));
		TRY(blake2b_final(&blake_state, out_buffer, BLAKE2B_OUTBYTES));
		memcpy(out, out_buffer, BLAKE2B_OUTBYTES / 2);
		out += BLAKE2B_OUTBYTES / 2;
		toproduce = (uint32_t)outlen - BLAKE2B_OUTBYTES / 2;

		while (toproduce > BLAKE2B_OUTBYTES) {
			memcpy(in_buffer, out_buffer, BLAKE2B_OUTBYTES);
			TRY(blake2b(out_buffer, BLAKE2B_OUTBYTES, in_buffer,
				BLAKE2B_OUTBYTES, NULL, 0));
			memcpy(out, out_buffer, BLAKE2B_OUTBYTES / 2);
			out += BLAKE2B_OUTBYTES / 2;
			toproduce -= BLAKE2B_OUTBYTES / 2;
		}

		memcpy(in_buffer, out_buffer, BLAKE2B_OUTBYTES);
		TRY(blake2b(out_buffer, toproduce, in_buffer, BLAKE2B_OUTBYTES, NULL,
			0));
		memcpy(out, out_buffer, toproduce);
	}
fail:
	clear_internal_memory(&blake_state, sizeof(blake_state));
	return ret;
#undef TRY
}
/* Argon2 Team - End Code */


static void load_block(block *dst, const void *input) {
	unsigned i;
	for (i = 0; i < ARGON2_QWORDS_IN_BLOCK; ++i) {
		dst->v[i] = load64((const uint8_t *)input + i * sizeof(dst->v[i]));
	}
}


void fill_first_blocks(uint8_t *blockhash, const argon2_instance_t *instance) {
	uint32_t l;
	/* Make the first and second block in each lane as G(H0||i||0) or
	G(H0||i||1) */
	uint8_t blockhash_bytes[ARGON2_BLOCK_SIZE];
	for (l = 0; l < instance->lanes; ++l) {

		store32(blockhash + ARGON2_PREHASH_DIGEST_LENGTH, 0);
		store32(blockhash + ARGON2_PREHASH_DIGEST_LENGTH + 4, l);
		blake2b_long(blockhash_bytes, ARGON2_BLOCK_SIZE, blockhash,
			ARGON2_PREHASH_SEED_LENGTH);
		load_block(&instance->memory[l * instance->lane_length + 0],
			blockhash_bytes);

		store32(blockhash + ARGON2_PREHASH_DIGEST_LENGTH, 1);
		blake2b_long(blockhash_bytes, ARGON2_BLOCK_SIZE, blockhash,
			ARGON2_PREHASH_SEED_LENGTH);
		load_block(&instance->memory[l * instance->lane_length + 1],
			blockhash_bytes);
	}
	clear_internal_memory(blockhash_bytes, ARGON2_BLOCK_SIZE);
}




int initialize(argon2_instance_t *instance, argon2_context *context) {
	uint8_t blockhash[ARGON2_PREHASH_SEED_LENGTH];
	int result = ARGON2_OK;

	if (instance == NULL || context == NULL)
		return ARGON2_INCORRECT_PARAMETER;
	instance->context_ptr = context;

	/* 1. Memory allocation */
	result = allocate_memory(context, (uint8_t **)&(instance->memory),
		instance->memory_blocks, sizeof(block));
	if (result != ARGON2_OK) {
		return result;
	}

	/* 2. Initial hashing */
	/* H_0 + 8 extra bytes to produce the first blocks */
	/* uint8_t blockhash[ARGON2_PREHASH_SEED_LENGTH]; */
	/* Hashing all inputs */
	initial_hash(blockhash, context, instance->type);
	/* Zeroing 8 extra bytes */
	clear_internal_memory(blockhash + ARGON2_PREHASH_DIGEST_LENGTH,
		ARGON2_PREHASH_SEED_LENGTH -
		ARGON2_PREHASH_DIGEST_LENGTH);

#ifdef GENKAT
	initial_kat(blockhash, context, instance->type);
#endif

	/* 3. Creating first blocks, we always have at least two blocks in a slice
	*/
	fill_first_blocks(blockhash, instance);
	/* Clearing the hash */
	clear_internal_memory(blockhash, ARGON2_PREHASH_SEED_LENGTH);

	return ARGON2_OK;
}

/*
* Argon2 position: where we construct the block right now. Used to distribute
* work between threads.
*/
typedef struct Argon2_position_t {
	uint32_t pass;
	uint32_t lane;
	uint8_t slice;
	uint32_t index;
} argon2_position_t;

/*Struct that holds the inputs for thread handling FillSegment*/
typedef struct Argon2_thread_data {
	argon2_instance_t *instance_ptr;
	argon2_position_t pos;
} argon2_thread_data;

int argon2_thread_join(argon2_thread_handle_t handle) {
#if defined(_WIN32)
	if (WaitForSingleObject((HANDLE)handle, INFINITE) == WAIT_OBJECT_0) {
		return CloseHandle((HANDLE)handle) != 0 ? 0 : -1;
	}
	return -1;
#else
	return pthread_join(handle, NULL);
#endif
}

int argon2_thread_create(argon2_thread_handle_t *handle,
	argon2_thread_func_t func, void *args) {
	if (NULL == handle || func == NULL) {
		return -1;
	}
#if defined(_WIN32)
	*handle = _beginthreadex(NULL, 0, func, args, 0, NULL);
	return *handle != 0 ? 0 : -1;
#else
	return pthread_create(handle, NULL, func, args);
#endif
}

void init_block_value(block *b, uint8_t in) { memset(b->v, in, sizeof(b->v)); }



void fill_block(__m128i *state, const block *ref_block, block *next_block,
	int with_xor) {
	__m128i block_XY[ARGON2_OWORDS_IN_BLOCK];
	unsigned int i;

	if (with_xor) {
		for (i = 0; i < ARGON2_OWORDS_IN_BLOCK; i++) {
			state[i] = _mm_xor_si128(
				state[i], _mm_loadu_si128((const __m128i *)ref_block->v + i));
			block_XY[i] = _mm_xor_si128(
				state[i], _mm_loadu_si128((const __m128i *)next_block->v + i));
		}
	}
	else {
		for (i = 0; i < ARGON2_OWORDS_IN_BLOCK; i++) {
			block_XY[i] = state[i] = _mm_xor_si128(
				state[i], _mm_loadu_si128((const __m128i *)ref_block->v + i));
		}
	}

	for (i = 0; i < 8; ++i) {
		BLAKE2_ROUND(state[8 * i + 0], state[8 * i + 1], state[8 * i + 2],
			state[8 * i + 3], state[8 * i + 4], state[8 * i + 5],
			state[8 * i + 6], state[8 * i + 7]);
	}

	for (i = 0; i < 8; ++i) {
		BLAKE2_ROUND(state[8 * 0 + i], state[8 * 1 + i], state[8 * 2 + i],
			state[8 * 3 + i], state[8 * 4 + i], state[8 * 5 + i],
			state[8 * 6 + i], state[8 * 7 + i]);
	}

	for (i = 0; i < ARGON2_OWORDS_IN_BLOCK; i++) {
		state[i] = _mm_xor_si128(state[i], block_XY[i]);
		_mm_storeu_si128((__m128i *)next_block->v + i, state[i]);
	}
}

static void next_addresses(block *address_block, block *input_block) {
	/*Temporary zero-initialized blocks*/
	__m128i zero_block[ARGON2_OWORDS_IN_BLOCK];
	__m128i zero2_block[ARGON2_OWORDS_IN_BLOCK];

	memset(zero_block, 0, sizeof(zero_block));
	memset(zero2_block, 0, sizeof(zero2_block));

	/*Increasing index counter*/
	input_block->v[6]++;

	/*First iteration of G*/
	fill_block(zero_block, input_block, address_block, 0);

	/*Second iteration of G*/
	fill_block(zero2_block, address_block, address_block, 0);
}


uint32_t index_alpha(const argon2_instance_t *instance,
	const argon2_position_t *position, uint32_t pseudo_rand,
	int same_lane) {
	/*
	* Pass 0:
	*      This lane : all already finished segments plus already constructed
	* blocks in this segment
	*      Other lanes : all already finished segments
	* Pass 1+:
	*      This lane : (SYNC_POINTS - 1) last segments plus already constructed
	* blocks in this segment
	*      Other lanes : (SYNC_POINTS - 1) last segments
	*/
	uint32_t reference_area_size;
	uint64_t relative_position;
	uint32_t start_position, absolute_position;

	if (0 == position->pass) {
		/* First pass */
		if (0 == position->slice) {
			/* First slice */
			reference_area_size =
				position->index - 1; /* all but the previous */
		}
		else {
			if (same_lane) {
				/* The same lane => add current segment */
				reference_area_size =
					position->slice * instance->segment_length +
					position->index - 1;
			}
			else {
				reference_area_size =
					position->slice * instance->segment_length +
					((position->index == 0) ? (-1) : 0);
			}
		}
	}
	else {
		/* Second pass */
		if (same_lane) {
			reference_area_size = instance->lane_length -
				instance->segment_length + position->index -
				1;
		}
		else {
			reference_area_size = instance->lane_length -
				instance->segment_length +
				((position->index == 0) ? (-1) : 0);
		}
	}

	/* 1.2.4. Mapping pseudo_rand to 0..<reference_area_size-1> and produce
	* relative position */
	relative_position = pseudo_rand;
	relative_position = relative_position * relative_position >> 32;
	relative_position = reference_area_size - 1 -
		(reference_area_size * relative_position >> 32);

	/* 1.2.5 Computing starting position */
	start_position = 0;

	if (0 != position->pass) {
		start_position = (position->slice == ARGON2_SYNC_POINTS - 1)
			? 0
			: (position->slice + 1) * instance->segment_length;
	}

	/* 1.2.6. Computing absolute position */
	absolute_position = (start_position + relative_position) %
		instance->lane_length; /* absolute position */
	return absolute_position;
}

void fill_segment(const argon2_instance_t *instance,
	argon2_position_t position) {
	block *ref_block = NULL, *curr_block = NULL;
	block address_block, input_block;
	uint64_t pseudo_rand, ref_index, ref_lane;
	uint32_t prev_offset, curr_offset;
	uint32_t starting_index, i;
	__m128i state[64];
	int data_independent_addressing;

	if (instance == NULL) {
		return;
	}

	data_independent_addressing =
		(instance->type == Argon2_i) ||
		(instance->type == Argon2_id && (position.pass == 0) &&
		(position.slice < ARGON2_SYNC_POINTS / 2));

	if (data_independent_addressing) {
		init_block_value(&input_block, 0);

		input_block.v[0] = position.pass;
		input_block.v[1] = position.lane;
		input_block.v[2] = position.slice;
		input_block.v[3] = instance->memory_blocks;
		input_block.v[4] = instance->passes;
		input_block.v[5] = instance->type;
	}

	starting_index = 0;

	if ((0 == position.pass) && (0 == position.slice)) {
		starting_index = 2; /* we have already generated the first two blocks */

							/* Don't forget to generate the first block of addresses: */
		if (data_independent_addressing) {
			next_addresses(&address_block, &input_block);
		}
	}

	/* Offset of the current block */
	curr_offset = position.lane * instance->lane_length +
		position.slice * instance->segment_length + starting_index;

	if (0 == curr_offset % instance->lane_length) {
		/* Last block in this lane */
		prev_offset = curr_offset + instance->lane_length - 1;
	}
	else {
		/* Previous block */
		prev_offset = curr_offset - 1;
	}

	memcpy(state, ((instance->memory + prev_offset)->v), ARGON2_BLOCK_SIZE);

	for (i = starting_index; i < instance->segment_length;
		++i, ++curr_offset, ++prev_offset) {
		/*1.1 Rotating prev_offset if needed */
		if (curr_offset % instance->lane_length == 1) {
			prev_offset = curr_offset - 1;
		}

		/* 1.2 Computing the index of the reference block */
		/* 1.2.1 Taking pseudo-random value from the previous block */
		if (data_independent_addressing) {
			if (i % ARGON2_ADDRESSES_IN_BLOCK == 0) {
				next_addresses(&address_block, &input_block);
			}
			pseudo_rand = address_block.v[i % ARGON2_ADDRESSES_IN_BLOCK];
		}
		else {
			pseudo_rand = instance->memory[prev_offset].v[0];
		}

		/* 1.2.2 Computing the lane of the reference block */
		ref_lane = ((pseudo_rand >> 32)) % instance->lanes;

		if ((position.pass == 0) && (position.slice == 0)) {
			/* Can not reference other lanes yet */
			ref_lane = position.lane;
		}

		/* 1.2.3 Computing the number of possible reference block within the
		* lane.
		*/
		position.index = i;
		ref_index = index_alpha(instance, &position, pseudo_rand & 0xFFFFFFFF,
			ref_lane == position.lane);

		/* 2 Creating a new block */
		ref_block =
			instance->memory + instance->lane_length * ref_lane + ref_index;
		curr_block = instance->memory + curr_offset;
		if (ARGON2_VERSION_10 == instance->version) {
			/* version 1.2.1 and earlier: overwrite, not XOR */
			fill_block(state, ref_block, curr_block, 0);
		}
		else {
			if (0 == position.pass) {
				fill_block(state, ref_block, curr_block, 0);
			}
			else {
				fill_block(state, ref_block, curr_block, 1);
			}
		}
	}
}

void argon2_thread_exit(void) {
#if defined(_WIN32)
	_endthreadex(0);
#else
	pthread_exit(NULL);
#endif
}

#ifdef _WIN32
static unsigned __stdcall fill_segment_thr(void *thread_data)
#else
static void *fill_segment_thr(void *thread_data)
#endif
{
	argon2_thread_data *my_data = (argon2_thread_data*)thread_data;
	fill_segment(my_data->instance_ptr, my_data->pos);
	argon2_thread_exit();
	return 0;
}


void xor_block(block *dst, const block *src) {
	int i;
	for (i = 0; i < ARGON2_QWORDS_IN_BLOCK; ++i) {
		dst->v[i] ^= src->v[i];
	}
}


void copy_block(block *dst, const block *src) {
	memcpy(dst->v, src->v, sizeof(uint64_t) * ARGON2_QWORDS_IN_BLOCK);
}


static void store_block(void *output, const block *src) {
	unsigned i;
	for (i = 0; i < ARGON2_QWORDS_IN_BLOCK; ++i) {
		store64((uint8_t *)output + i * sizeof(src->v[i]), src->v[i]);
	}
}

static const unsigned Mod37BitPosition[] = // map a bit value mod 37 to its position		
{
	1, 0, 1, 26, 2, 23, 27, 0, 3, 16, 24, 30, 28, 11, 0, 13, 4,
		7, 17, 0, 25, 22, 31, 15, 29, 10, 12, 6, 0, 21, 14, 9, 5,
	20, 8, 19, 18
};

unsigned trailing_zeros(unsigned n) {
	return Mod37BitPosition[(-n & n) % 37];	
}

int fill_memory_blocks(argon2_instance_t *instance) {
	uint32_t r, s;
	argon2_thread_handle_t *thread = NULL;
	argon2_thread_data *thr_data = NULL;
	int rc = ARGON2_OK;

	if (instance == NULL || instance->lanes == 0) {
		rc = ARGON2_THREAD_FAIL;
		goto fail;
	}

	/* 1. Allocating space for threads */
	thread = (argon2_thread_handle_t*)calloc(instance->lanes, sizeof(argon2_thread_handle_t));
	if (thread == NULL) {
		rc = ARGON2_MEMORY_ALLOCATION_ERROR;
		goto fail;
	}

	thr_data = (argon2_thread_data*)calloc(instance->lanes, sizeof(argon2_thread_data));
	if (thr_data == NULL) {
		rc = ARGON2_MEMORY_ALLOCATION_ERROR;
		goto fail;
	}

	for (r = 0; r < instance->passes; ++r) {
		for (s = 0; s < ARGON2_SYNC_POINTS; ++s) {
			uint32_t l;

			/* 2. Calling threads */
			for (l = 0; l < instance->lanes; ++l) {
				argon2_position_t position;

				/* 2.1 Join a thread if limit is exceeded */
				if (l >= instance->threads) {
					if (argon2_thread_join(thread[l - instance->threads])) {
						rc = ARGON2_THREAD_FAIL;
						goto fail;
					}
				}

				/* 2.2 Create thread */
				position.pass = r;
				position.lane = l;
				position.slice = (uint8_t)s;
				position.index = 0;
				thr_data[l].instance_ptr =
					instance; /* preparing the thread input */
				memcpy(&(thr_data[l].pos), &position,
					sizeof(argon2_position_t));
				if (argon2_thread_create(&thread[l], &fill_segment_thr,
					(void *)&thr_data[l])) {
					rc = ARGON2_THREAD_FAIL;
					goto fail;
				}

				/* fill_segment(instance, position); */
				/*Non-thread equivalent of the lines above */
			}

			/* 3. Joining remaining threads */
			for (l = instance->lanes - instance->threads; l < instance->lanes;
				++l) {
				if (argon2_thread_join(thread[l])) {
					rc = ARGON2_THREAD_FAIL;
					goto fail;
				}
			}
		}


		//internal_kat(instance, r); /* Print all memory blocks */


		if (instance != NULL) {
			uint32_t i;
			std::vector<uint256> leaves;
			for (i = 0; i < instance->memory_blocks; ++i) {
				block blockhash;
				copy_block(&blockhash, &instance->memory[i]);
				uint8_t blockhash_bytes[ARGON2_BLOCK_SIZE];
				store_block(blockhash_bytes, &blockhash);
				uint8_t output[16];
				blake2b(output, 16, blockhash_bytes, ARGON2_BLOCK_SIZE, NULL, 0);
				uint256 rv;
				rv.SetHexUnsigned(output);
				leaves.push_back(rv);
			}

			uint256 resultMerkelRoot = ComputeMerkleRoot(leaves);
			std::cout << resultMerkelRoot.GetHex() << std::endl;

			// Step 3 : Select nonce N
			uint64_t nNonce = 0;
			uint8_t blockhash_output[16];
			uint8_t blockhash_input[40];
			uint8_t Y[70][16];

			// Step 4 : Y0 = H(resultMerkelRoot, N)
			blake2b(Y[0], 16, &resultMerkelRoot, 32, &nNonce, 8);
			
			// Step 5 : For 1 <= j <= L
						// I(j) = Y(j - 1) mod T;
						// Y(j) = H(Y(j - 1), X[I(j)])
			uint8_t L = 70;
			for (uint8_t j = 1; j < L; j++) {
				uint32_t ij = Y[j - 1] % 2048;
				block blockhash;
				copy_block(&blockhash, instance->memory);
				blake2b(Y[j], 16, Y[j - 1], 16, &instance->memory[ij], 8);
			}

		}

	}

fail:
	if (thread != NULL) {
		free(thread);
	}
	if (thr_data != NULL) {
		free(thr_data);
	}
	return rc;
}


void free_memory(const argon2_context *context, uint8_t *memory,
	size_t num, size_t size) {
	size_t memory_size = num*size;
	clear_internal_memory(memory, memory_size);
	if (context->free_cbk) {
		(context->free_cbk)(memory, memory_size);
	}
	else {
		free(memory);
	}
}

void finalize(const argon2_context *context, argon2_instance_t *instance) {
	if (context != NULL && instance != NULL) {
		block blockhash;
		uint32_t l;

		copy_block(&blockhash, instance->memory + instance->lane_length - 1);

		/* XOR the last blocks */
		for (l = 1; l < instance->lanes; ++l) {
			uint32_t last_block_in_lane =
				l * instance->lane_length + (instance->lane_length - 1);
			xor_block(&blockhash, instance->memory + last_block_in_lane);
		}

		/* Hash the result */
		{
			uint8_t blockhash_bytes[ARGON2_BLOCK_SIZE];
			store_block(blockhash_bytes, &blockhash);
			blake2b_long(context->out, context->outlen, blockhash_bytes,
				ARGON2_BLOCK_SIZE);
			/* clear blockhash and blockhash_bytes */
			clear_internal_memory(blockhash.v, ARGON2_BLOCK_SIZE);
			clear_internal_memory(blockhash_bytes, ARGON2_BLOCK_SIZE);
		}

#ifdef GENKAT
		print_tag(context->out, context->outlen);
#endif

		free_memory(context, (uint8_t *)instance->memory,
			instance->memory_blocks, sizeof(block));
	}
}

int argon2_ctx(argon2_context *context, argon2_type type) {
	/* 1. Validate all inputs */
	int result = validate_inputs(context);
	uint32_t memory_blocks, segment_length;
	argon2_instance_t instance;

	if (ARGON2_OK != result) {
		return result;
	}

	if (Argon2_d != type && Argon2_i != type && Argon2_id != type) {
		return ARGON2_INCORRECT_TYPE;
	}

	/* 2. Align memory size */
	/* Minimum memory_blocks = 8L blocks, where L is the number of lanes */
	memory_blocks = context->m_cost;

	if (memory_blocks < 2 * ARGON2_SYNC_POINTS * context->lanes) {
		memory_blocks = 2 * ARGON2_SYNC_POINTS * context->lanes;
	}

	segment_length = memory_blocks / (context->lanes * ARGON2_SYNC_POINTS);
	/* Ensure that all segments have equal length */
	memory_blocks = segment_length * (context->lanes * ARGON2_SYNC_POINTS);

	instance.version = context->version;
	instance.memory = NULL;
	instance.passes = context->t_cost;
	instance.memory_blocks = memory_blocks;
	instance.segment_length = segment_length;
	instance.lane_length = segment_length * ARGON2_SYNC_POINTS;
	instance.lanes = context->lanes;
	instance.threads = context->threads;
	instance.type = type;

	/* 3. Initialization: Hashing inputs, allocating memory, filling first
	* blocks
	*/
	result = initialize(&instance, context);

	if (ARGON2_OK != result) {
		return result;
	}

	/* 4. Filling memory */
	result = fill_memory_blocks(&instance);

	if (ARGON2_OK != result) {
		return result;
	}
	/* 5. Finalization */
	finalize(context, &instance);

	return ARGON2_OK;
}


static void generate_testvectors(argon2_type type, const uint32_t version) {
	
#define TEST_OUTLEN 32
#define TEST_PWDLEN 32
#define TEST_SALTLEN 16
#define TEST_SECRETLEN 8
#define TEST_ADLEN 12
	argon2_context context;

	unsigned char out[TEST_OUTLEN];
	unsigned char pwd[TEST_PWDLEN];
	unsigned char salt[TEST_SALTLEN];
	unsigned char secret[TEST_SECRETLEN];
	unsigned char ad[TEST_ADLEN];
	const allocate_fptr myown_allocator = NULL;
	const deallocate_fptr myown_deallocator = NULL;

	unsigned t_cost = 1;
	unsigned m_cost = 2097152;	
	unsigned lanes = 4;

	memset(pwd, 1, TEST_OUTLEN);
	memset(salt, 2, TEST_SALTLEN);
	memset(secret, 3, TEST_SECRETLEN);
	memset(ad, 4, TEST_ADLEN);

	context.out = out;
	context.outlen = TEST_OUTLEN;
	context.version = version;
	context.pwd = pwd;
	context.pwdlen = TEST_PWDLEN;
	context.salt = salt;
	context.saltlen = TEST_SALTLEN;
	context.secret = secret;
	context.secretlen = TEST_SECRETLEN;
	context.ad = ad;
	context.adlen = TEST_ADLEN;
	context.t_cost = t_cost;
	context.m_cost = m_cost;
	context.lanes = lanes;
	context.threads = lanes;
	context.allocate_cbk = myown_allocator;
	context.free_cbk = myown_deallocator;
	context.flags = ARGON2_DEFAULT_FLAGS;

#undef TEST_OUTLEN
#undef TEST_PWDLEN
#undef TEST_SALTLEN
#undef TEST_SECRETLEN
#undef TEST_ADLEN

	argon2_ctx(&context, type);
}



int main(int argc, char *argv[]) {
	/* Get and check Argon2 type */
	const char *type_str = (argc > 1) ? argv[1] : "d";
	argon2_type type = Argon2_i;
	uint32_t version = ARGON2_VERSION_NUMBER;
	if (!strcmp(type_str, "d")) {
		type = Argon2_d;
	}
	else if (!strcmp(type_str, "i")) {
		type = Argon2_i;
	}
	else if (!strcmp(type_str, "id")) {
		type = Argon2_id;
	}
	else {
		fatal("wrong Argon2 type");
	}

	/* Get and check Argon2 version number */
	if (argc > 2) {
		version = strtoul(argv[2], NULL, 10);
	}
	if (ARGON2_VERSION_10 != version && ARGON2_VERSION_NUMBER != version) {
		fatal("wrong Argon2 version number");
	}

	generate_testvectors(type, version);
	return ARGON2_OK;
}