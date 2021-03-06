typedef enum mt_error {
	MT_SUCCESS = 0, /*!< Operation terminated successfully */
	MT_ERR_OUT_Of_MEMORY = -1, /*!< There was not enough memory to complete the operation */
	MT_ERR_ILLEGAL_PARAM = -2, /*!< At least one of the specified parameters was illegal */
	MT_ERR_ILLEGAL_STATE = -3, /*!< The operation reached an illegal state */
	MT_ERR_ROOT_MISMATCH = -4, /*!< Signals the failure of a root hash verification */
	MT_ERR_UNSPECIFIED = -5  /*!< A general error occurred */
} mt_error_t;

