#include "merkle.h"
#include "hash.h"
#include "utilstrencodings.h"
#include "proof.h"
void proof()
{

	for (uint32_t i = 0; i < 5; ++i) {
		if (mt_verify(mt, test_values[i], HASH_LENGTH, i) == MT_ERR_ROOT_MISMATCH) {
			printf("Root mismatch error!\n");
			return;
		}
	}
}
mt_error mt_verify(const mt_t *mt, const uint8_t *tag, const size_t len,
		const uint32_t offset)
	{
		if (!(mt && tag && len <= HASH_LENGTH && (offset < mt->elems))) {
			return MT_ERR_ILLEGAL_PARAM;
		}
		uint8_t message_digest[HASH_LENGTH];
		mt_init_hash(message_digest, tag, len);
		uint32_t q = offset;
		uint32_t l = 0;         // level
		while (hasNextLevelExceptRoot(mt, l)) {
			if (!(q & 0x01)) { // left subtree
							   // If I am the left neighbor (even index), we need to check if a right
							   // neighbor exists.
				const uint8_t *right;
				if ((right = findRightNeighbor(mt, q + 1, l)) != NULL) {
					MT_ERR_CHK(mt_hash(message_digest, right, message_digest));
				}
			}
			else {           // right subtree
							 // In the right subtree, there must always be a left neighbor!
				uint8_t const * const left = mt_al_get(mt->level[l], q - 1);
				MT_ERR_CHK(mt_hash(left, message_digest, message_digest));
			}
			q >>= 1;
			l += 1;
		}
		//mt_print_hash(message_digest);
		int r = memcmp(message_digest, mt_al_get(mt->level[l], q), HASH_LENGTH);
		if (r) {
			return MT_ERR_ROOT_MISMATCH;
		}
		else {
			return MT_SUCCESS;
		}
	}